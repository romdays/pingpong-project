import numpy as np
import cv2
import itertools

from settings import Settings
from selection import similar_vecs

def mask_move_obj(prev_img, curr_img, next_img):
    prev_img = cv2.blur(cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY), ksize=(5,5))
    curr_img = cv2.blur(cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY), ksize=(5,5))
    next_img = cv2.blur(cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY), ksize=(5,5))

    # get t-s and t+s images
    p_c_diff = cv2.absdiff(prev_img, curr_img)
    c_n_diff = cv2.absdiff(curr_img, next_img)

    # binarization
    _, p_c_diff = cv2.threshold(p_c_diff,10,255, cv2.THRESH_BINARY)
    _, c_n_diff = cv2.threshold(c_n_diff,10,255, cv2.THRESH_BINARY)

    # get moving object
    mask = c_n_diff*p_c_diff*255

    # closing
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    return mask

def mask_hsv_color_space(image, lower, upper):
    # extraction color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image, lower, upper)

    # closing
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    return mask

def detect_circular_obj_points_from_binary_image(image, min_circularity, min_contour_area, max_contour_ara):
    # line up candidates for ball object
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    info = []
    for cnt in contours:
        cnt = cv2.convexHull(cnt)
        arclen = cv2.arcLength(cnt, True)
        if arclen < 1e-8: continue
        mu = cv2.moments(cnt)
        area = mu["m00"] # cv2.contourArea(cnt)
        circularity = 4*np.pi*area/(arclen**2)
        if (min_contour_area<area and area<max_contour_ara \
            and min_circularity<circularity and circularity<1.0):
            x,y= int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])

            points.append(np.array((x,y), dtype=float))

            info.append((cnt, area, circularity, x, y))

    return points, info

def search_template_area(images, name=''):
    mask = mask_move_obj(images[0], images[1], images[2])
    masked_img = cv2.bitwise_and(images[1], images[1], mask=mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt = cv2.convexHull(cnt)
        arclen = cv2.arcLength(cnt, True)
        if arclen < 1e-8: continue
        mu = cv2.moments(cnt)
        area = mu["m00"] # cv2.contourArea(cnt)
        circularity = 4*np.pi*area/(arclen**2)
        if (Settings.get('TEMPLATE_MIN_CIRCULARITY')<circularity and circularity<1.0):
            x,y= int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])
            side = int(np.sqrt(area/np.pi)//1 + 1)
            Settings.update_template(name, masked_img[y-side:y+side,x-side:x+side])
    return Settings.get_template(name), masked_img

def template_matching_detection(images, name=''):
    points = []
    score = [0]
    
    temp_imgs, masked_img = search_template_area(images, name)
    if temp_imgs:
        max_values = []
        gray = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)
        for temp in temp_imgs:
            temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)
            side,_ = temp.shape

            match = cv2.matchTemplate(gray, temp, cv2.TM_CCOEFF_NORMED)

            min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
            points.append(np.array((max_pt[0]+side//2,max_pt[1]+side//2), dtype=float))
            max_values.append(max_value)
            # cv2.circle(images[1],(max_pt[0]+side//2,max_pt[1]+side//2), 5, (0,0,255), -1)

        similarity = Settings.get('TEMPLATE_MIN_SIMILARITY')
        for i,j in itertools.combinations(range(len(points)), 2):
            if len(score) <= j: score.append(0)
            if similar_vecs(points[i], points[j]):
                if max_values[i] > similarity: score[i] += 1
                if max_values[j] > similarity: score[j] += 1

    best_idx = score[::-1].index(max(score))
    best_point = points[::-1][best_idx:best_idx+1]
    
    while temp_imgs and len(Settings.get_template(name)) > 5:
        worst_idx = score.index(min(score))
        Settings.remove_template(name, worst_idx)
        score.pop(worst_idx)

    if best_point: cv2.circle(images[1],(int(best_point[0][0]),int(best_point[0][1])), 5, (0,0,255), -1)
    cv2.imshow('detection:'+name, images[1])

    return best_point

def detection(images, name=""):
    return template_matching_detection(images, name)
    
    mask = mask_move_obj(images[0], images[1], images[2])
    masked_img = cv2.bitwise_and(images[1], images[1], mask=mask)
    # mask = mask_hsv_color_space(masked_img, Settings.get('LOWER_COLOR'), Settings.get('UPPER_COLOR'))

    points, info = detect_circular_obj_points_from_binary_image(
        mask, Settings.get('MIN_CIRCULARITY'), Settings.get('MIN_CONTOUR_AREA'), Settings.get('MAX_CONTOUR_AREA')
        )

    # debug
    mask=np.expand_dims(mask, axis=2)
    mask=np.concatenate([mask]*3, axis=2)
    lst = [i[0] for i in info]
    if len(info)==1:
        cv2.drawContours(images[1],lst,-1,(0,255,0),-3)
        cv2.drawContours(mask,lst,-1,(0,255,0),-1)
    elif len(info)>1:
        cv2.drawContours(images[1],lst,-1,(0,0,255),-3)
        cv2.drawContours(mask,lst,-1,(0,0,255),-1)

    for i, data in enumerate(info):
        text='area: {}, circularity: {}, (x,y): {}'.format(data[1], data[2], (data[3],data[4]))
        cv2.putText(mask, text, (10, 50*(i+1)), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # cv2.imshow(name+'detection2', mask)
    # cv2.imshow(name+'detection', images[1])
    # cv2.imshow(name+'detection3', masked_img)

    return points#, np.concatenate((images[1], mask), axis=0).astype(np.uint8)

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
    black_top[margin:margin+top.shape[0]] = top
    black_btm[margin:margin+btm.shape[0]] = btm

    return black_top, black_btm


if __name__ == '__main__':
    images = []
    image_shape = (640, 480)
    Settings(image_shape)

    cap = cv2.VideoCapture('./data/videos/ds/13.mov')
    # cap = cv2.VideoCapture('./data/videos/DCIM/100MEDIA/DJI_0022.MP4')
    for i in range(330):
        ret, frame = cap.read()
        

    for i in range(Settings.get('FRAME_INTERVAL')*2):
        ret, frame = cap.read()
        frame, _ = vsplit_ds_frame(frame, image_shape)
        images.append(frame)
        if frame is None:
            exit()

    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('./data/output/output.mp4',fourcc, 60.0, (frame.shape[1], frame.shape[0]*2))
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame, _ = vsplit_ds_frame(frame, image_shape)
        if frame is None:
            break

        images.append(frame)
        # obj = detection(images[0::Settings.get('FRAME_INTERVAL')])
        # temp_match(images)
        template_matching_detection(images)
        images.pop(0)

        # out.write(img)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()