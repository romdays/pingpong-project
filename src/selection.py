import numpy as np
import cv2
import itertools

from settings import Settings

# def calc_near_point(point, point_list, distance):
#     if len(point_list)==0: return []
#     near_points = filter(lambda x:x<distance, [np.sqrt(np.sum((point-p)**2)) for p in point_list]) # max move distance[cm] par frame

#     return near_points

# def calc_uniformly_accelerated_model(points, ts):
#     dt1 = (ts[1]-ts[0])/60.
#     dt2 = (ts[2]-ts[1])/60.
#     acceleration = 2*(dt1*(points[2]-points[1])-dt2*(points[1]-points[0]))/(dt1*dt2*(dt1+dt2))
#     velocity = (points[1]-points[0])/dt1 - dt1*acceleration/2.

#     return acceleration, velocity


# def select_points_fit_model(a, v, point, init, points_seq):
#     fit_points = []
#     counter = []
#     for i in range(len(points_seq)):
#         dt = (i-init)/60
#         on_model = point + v*dt + a*(dt**2.)/2.
#         tmp = calc_near_point(on_model, points_seq[i], 30)
#         fit_points.append(tmp)
#         if tmp: counter.append(i)

#     if len(counter)<3: None # ２点しか見つからないとき、GGする

#     ts = [counter[0], counter[len(counter)/2], counter[-1]]

#     return [fit_points[i][0] for i in ts], ts



# def escape(points_seq):
#     mid = len(points_seq)/2
#     curr = points_seq[mid]
#     if curr and mid<3 : return []

#     for point in curr:
#         past = calc_near_point(point, points_seq[mid-1], 30)
#         future = calc_near_point(point, points_seq[mid+1], 30)
        
#         for prv in past:
#             for nxt in future:
#                 acceleration, velocity = calc_uniformly_accelerated_model(
#                     [prv, point,nxt], [mid-1, mid, mid+1]
#                     )

                # select_points_fit_model(
                #     acceleration, velocity, point, mid, points_seq
                #     )
def narrow_down_by_existence_area(points):
    def is_in_play_area(point):
        grad_x= 2.0 * Settings.get('TABLE_WIDTH') / Settings.get('TABLE_LENGTH') * point[0,0]
        intercept = Settings.get('TABLE_WIDTH') / 2.0
        return not(point[1,0] - grad_x - intercept > 0 and point[1,0] + grad_x - intercept > 0
                or point[1,0] - grad_x + intercept < 0 and point[1,0] + grad_x + intercept < 0)

    return [point for point in points if is_in_play_area(point)]


def calc_closest_point_nearby_prev_points(base_points, points):
    if len(points)==0: return []
    base_points = [x for x in base_points if x]
    if base_points: base_point = base_points[-1]
    else: base_point = np.array(([0],[0],[0]))
    distance = [np.sqrt(np.sum((point-base_point)**2)) for point in points]
    return [points[distance.index(min(distance))]]

def similar_vecs(v1, v2, similarlity=0.95):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if not v1_norm or not v2_norm: return False
    cos = np.dot(v1.flatten(), v2.flatten()) / (v1_norm * v2_norm)
    ratio = 1.0 - (np.abs(v1_norm - v2_norm) / (v1_norm + v2_norm))

    if cos * ratio  > similarlity:
        return True
    else: return False

def extract_points_similarly_movements(points_seq, holder, similarlity=0.95):
    length = len(holder)
    holder.append([])
    if length < 4: raise Exception('expect len(holder) >= 4. but len(holder): {}'.format(length))
    if len(points_seq)<length: return holder.pop(0), holder

    l = length-1
    for i,j,k in itertools.combinations(range(l), 3):
        for p1, p2, p3, p4 in itertools.product(points_seq[i], points_seq[j], points_seq[k], points_seq[l]):
            v1 = (p2-p1)/(j-i)
            v2 = (p3-p2)/(k-j)
            v3 = (p4-p3)/(l-k)
            if similar_vecs(v1, v2, similarlity) and similar_vecs(v2, v3, similarlity):
                for p, idx in zip((p1,p2,p3,p4), (i,j,k,l)):
                    if not any([np.array_equal(val,p) for val in holder[idx]]): holder[idx].append(p)

    return holder.pop(0), holder

if __name__ == '__main__':
    for i,j,k in itertools.combinations(range(5), 3):
        print(i,j,k)
