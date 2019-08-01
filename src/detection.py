import numpy as np
import cv2

from settings import (
    MIN_CIRCULARITY,
    MIN_COUNTOUR_AREA,
    MAX_COUNTOUR_AREA,
)

def detection(images):
    outframe = images[1]

    prev_img = cv2.blur(cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY), ksize=(5,5))
    curr_img = cv2.blur(cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY), ksize=(5,5))
    next_img = cv2.blur(cv2.cvtColor(images[2], cv2.COLOR_BGR2GRAY), ksize=(5,5))

    # get t-s and t+s images
    p_c_diff = cv2.absdiff(prev_img, curr_img)
    c_n_diff = cv2.absdiff(curr_img, next_img)

    # binarization
    _, p_c_diff = cv2.threshold(p_c_diff,5,255, cv2.THRESH_BINARY)
    _, c_n_diff = cv2.threshold(c_n_diff,5,255, cv2.THRESH_BINARY)

    # get moving object
    move_obj = c_n_diff*p_c_diff*255
    move_obj = cv2.bitwise_and(images[1], images[1], mask=move_obj)

    # extraction ping-pong color space
    lower_color = np.array([0,0,200])
    upper_color = np.array([180,100,255])
    hsv = cv2.cvtColor(move_obj, cv2.COLOR_BGR2HSV)
    moving_and_white_obj = cv2.inRange(hsv, lower_color, upper_color)

    #closing
    kernel = np.ones((5,5),np.uint8)
    moving_and_white_obj = cv2.morphologyEx(moving_and_white_obj, cv2.MORPH_CLOSE, kernel)

    # select candidates for ball object
    contours, hierarchy = cv2.findContours(moving_and_white_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

            print('area: {}, circularity: {}, (x,y): {}'.format(area, circularity, (x,y)))

    bg = np.zeros(outframe.shape)
    if len(candidates)==1:
        cv2.drawContours(outframe,candidates,-1,(0,255,0),3)
        cv2.drawContours(bg,candidates,-1,(0,255,0),-1)
    elif len(candidates)>1:
        cv2.drawContours(outframe,candidates,-1,(0,0,255),3)
        cv2.drawContours(bg,candidates,-1,(0,0,255),-1)

    cv2.imshow('detection3', bg)
    cv2.imshow('detection2', moving_and_white_obj)
    cv2.imshow('detection', outframe)

    moving_and_white_obj=np.expand_dims(moving_and_white_obj, axis=2)
    moving_and_white_obj=np.concatenate((moving_and_white_obj,moving_and_white_obj,moving_and_white_obj), axis=2)

    return objs, np.concatenate((outframe, bg, moving_and_white_obj), axis=0).astype(np.uint8)


if __name__ == '__main__':
    FRAME_INTERVAL=2

    images = []

    # cap = cv2.VideoCapture('./data/videos/003.mp4')
    cap = cv2.VideoCapture('./data/videos/DCIM/100MEDIA/DJI_0023.MP4')

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