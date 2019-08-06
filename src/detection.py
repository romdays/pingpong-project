import numpy as np
import cv2

from settings import Settings

def detection(images):
    prev_img = cv2.blur(cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY), ksize=(5,5))
    curr_img = cv2.blur(cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY), ksize=(5,5))
    next_img = cv2.blur(cv2.cvtColor(images[2], cv2.COLOR_BGR2GRAY), ksize=(5,5))

    # get t-s and t+s images
    p_c_diff = cv2.absdiff(prev_img, curr_img)
    c_n_diff = cv2.absdiff(curr_img, next_img)

    # binarization
    _, p_c_diff = cv2.threshold(p_c_diff,10,255, cv2.THRESH_BINARY)
    _, c_n_diff = cv2.threshold(c_n_diff,10,255, cv2.THRESH_BINARY)

    # get moving object
    move_obj = c_n_diff*p_c_diff*255

    # closing first
    kernel = np.ones((5,5),np.uint8)
    move_obj = cv2.morphologyEx(move_obj, cv2.MORPH_CLOSE, kernel)
    moving_and_white_obj = move_obj

    # # extraction ping-pong color space
    # move_obj = cv2.bitwise_and(images[1], images[1], mask=move_obj)
    # hsv = cv2.cvtColor(move_obj, cv2.COLOR_BGR2HSV)
    # moving_and_white_obj = cv2.inRange(hsv, Settings.get('LOWER_COLOR'), Settings.get('UPPER_COLOR'))

    # # closing second
    # moving_and_white_obj = cv2.morphologyEx(moving_and_white_obj, cv2.MORPH_CLOSE, kernel)

    # select candidates for ball object
    contours, hierarchy = cv2.findContours(moving_and_white_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs = []
    candidates = []
    info = []
    for cnt in contours:
        cnt = cv2.convexHull(cnt)
        arclen = cv2.arcLength(cnt, True)
        if arclen < 1e-8: continue
        mu = cv2.moments(cnt)
        area = mu["m00"] # cv2.contourArea(cnt)
        circularity = 4*np.pi*area/(arclen**2)
        if (Settings.get('MIN_CONTOUR_AREA')<area and area<Settings.get('MAX_CONTOUR_AREA') \
            and Settings.get('MIN_CIRCULARITY')<circularity and circularity<1.0):
            x,y= int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])

            objs.append((x,y))

            candidates.append(cnt)
            info.append((area, circularity, x, y))

    # debug
    outframe = images[1]
    moving_and_white_obj=np.expand_dims(moving_and_white_obj, axis=2)
    moving_and_white_obj=np.concatenate((moving_and_white_obj,moving_and_white_obj,moving_and_white_obj), axis=2)
    if len(candidates)==1:
        cv2.drawContours(outframe,candidates,-1,(0,255,0),3)
        cv2.drawContours(moving_and_white_obj,candidates,-1,(0,255,0),-1)
        # cv2.drawContours(move_obj,candidates,-1,(0,255,0),-1)
    elif len(candidates)>1:
        cv2.drawContours(outframe,candidates,-1,(0,0,255),3)
        cv2.drawContours(moving_and_white_obj,candidates,-1,(0,0,255),-1)
        # cv2.drawContours(move_obj,candidates,-1,(0,0,255),-1)

    for i, data in enumerate(info):
        text='area: {}, circularity: {}, (x,y): {}'.format(data[0], data[1], (data[2],data[3]))
        cv2.putText(moving_and_white_obj, text, (10, 50*(i+1)), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('detection2', moving_and_white_obj)
    cv2.imshow('detection', outframe)
    cv2.imshow('detection3', move_obj)

    return objs, np.concatenate((outframe, moving_and_white_obj), axis=0).astype(np.uint8)

def vsplit_ds_frame(image, shape):
    width, height = shape
    shape = image.shape
    image = image[:, 277:-272, :]
    top, btm = np.vsplit(image, 2)
    shape = top.shape

    _height = shape[0]*width//shape[1]
    top = cv2.resize(top,(width, _height))
    btm = cv2.resize(btm,(width, _height))

    margin = (height-_height)//2
    black_top = cv2.resize(np.zeros((1, 1, 3), np.uint8), (width, height))
    black_btm = cv2.resize(np.zeros((1, 1, 3), np.uint8), (width, height))
    black_top[margin:-margin] = top
    black_btm[margin:-margin] = btm

    return black_top, black_btm


if __name__ == '__main__':
    images = []

    # cap = cv2.VideoCapture('./data/videos/ds/13.mov')
    cap = cv2.VideoCapture('./data/videos/DCIM/100MEDIA/DJI_0022.MP4')

    for i in range(Settings.get('FRAME_INTERVAL')*2):
        ret, frame = cap.read()
        # frame, _ = vsplit_ds_frame(frame, (640, 480))
        images.append(frame)
        if frame is None:
            exit()

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('./data/output/output.mp4',fourcc, 60.0, (frame.shape[1], frame.shape[0]*2))

    while(cap.isOpened()):
        ret, frame = cap.read()
        # frame, _ = vsplit_ds_frame(frame, (640, 480))
        if frame is None:
            break

        images.append(frame)
        obj, img = detection(images[0::Settings.get('FRAME_INTERVAL')])
        images.pop(0)

        out.write(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()