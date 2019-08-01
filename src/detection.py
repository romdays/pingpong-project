import numpy as np
import cv2

from settings import (
    MIN_CIRCULARITY,
    MIN_COUNTOUR_AREA,
    MAX_COUNTOUR_AREA,
)

def detection(images):
    outframe = images[1]

    # MIN_COUNTOUR_AREA = (outframe.shape[0]*outframe.shape[1])//5000
    # MAX_COUNTOUR_AREA = (outframe.shape[0]*outframe.shape[1])//

    lower_color = np.array([0,0,200])
    upper_color = np.array([180,100,255])

    def extraction_hsv_color_space(image, lower, upper):
        # blur = cv2.GaussianBlur(image, ksize=(5,5),sigmaX=2)
        blur = cv2.blur(image, ksize=(5,5))
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        return cv2.bitwise_and(blur, blur, mask=mask)
        # return cv2.bitwise_and(image, image, mask=mask)

    prev_img = extraction_hsv_color_space(images[0], lower_color, upper_color)
    curr_img = extraction_hsv_color_space(images[1], lower_color, upper_color)
    next_img = extraction_hsv_color_space(images[2], lower_color, upper_color)

    prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

    # get t-s and t+s images
    p_c_diff = cv2.absdiff(prev_img, curr_img)
    c_n_diff = cv2.absdiff(curr_img, next_img)

    # binarization
    _, p_c_diff = cv2.threshold(p_c_diff,5,255, cv2.THRESH_BINARY)
    _, c_n_diff = cv2.threshold(c_n_diff,5,255, cv2.THRESH_BINARY)

    
    # get moving object
    AND_img = c_n_diff*p_c_diff*255

    # select candidates for ball object
    contours, hierarchy = cv2.findContours(AND_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs = []
    candidates = []
    for cnt in contours:
        arclen = cv2.arcLength(cnt, True) 
        if arclen < 1e-8: continue
        mu = cv2.moments(cnt)
        area = mu["m00"] # cv2.contourArea(cnt)
        circularity = 4*np.pi*area/(arclen**2)
        if (MIN_COUNTOUR_AREA<area and area<MAX_COUNTOUR_AREA \
            and MIN_CIRCULARITY<circularity and circularity<1.0):
            x,y= int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])

            objs.append((x,y))
            candidates.append(cnt)

    #         print('area: {}, circularity: {}, (x,y): {}'.format(area, circularity, (x,y)))

    bg = np.zeros(outframe.shape)
    if len(candidates)==1:
        cv2.drawContours(outframe,candidates,-1,(0,255,0),3)
        cv2.drawContours(bg,candidates,-1,(0,255,0),-1)
    elif len(candidates)>1:
        cv2.drawContours(outframe,candidates,-1,(0,0,255),3)
        cv2.drawContours(bg,candidates,-1,(0,0,255),-1)

    cv2.imshow('detection3', AND_img)
    cv2.imshow('detection2', bg)
    cv2.imshow('detection', outframe)

    AND_img=np.expand_dims(AND_img, axis=2)
    AND_img=np.concatenate((AND_img,AND_img,AND_img), axis=2)

    return objs, np.concatenate((outframe, bg, AND_img), axis=0).astype(np.uint8)


if __name__ == '__main__':
    FRAME_INTERVAL=1

    images = []

    # cap = cv2.VideoCapture('./data/videos/003.mp4')
    cap = cv2.VideoCapture('./data/videos/DCIM/100MEDIA/DJI_0022.MP4')

    for i in range(FRAME_INTERVAL*2):
        ret, frame = cap.read()
        images.append(frame)
        if frame is None:
            exit()

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('./data/output/output.mp4',fourcc, 30.0, (frame.shape[1], frame.shape[0]*3))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break

        images.append(frame)
        obj, img = detection(images[0::FRAME_INTERVAL])
        images.pop(0)

        out.write(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()