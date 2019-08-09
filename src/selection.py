import numpy as np
import cv2
import itertools

from settings import Settings

def similar_vecs(v1, v2, similarlity=0.8):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if not v1_norm or not v2_norm: return False
    cos = np.dot(v1.flatten(), v2.flatten()) / (v1_norm * v2_norm)
    ratio = 1.0 - (np.abs(v1_norm - v2_norm) / (v1_norm + v2_norm))

    if cos * ratio  > similarlity:
        return True
    else: return False

def selection(points_seq, holder):
    length = len(holder)
    if len(points_seq)<length: return holder
    foo = []
    for i,j,k in itertools.combinations(range(length), 3):
        for before, middle, after in itertools.product(points_seq[i], points_seq[j], points_seq[k]):
            v1 = (middle-before)/(j-i)
            v2 = (after-middle)/(k-j)
            if similar_vecs(v1, v2):
                for bar in foo:
                    if (i,j,k)==bar[1]:
                        continue
                    elif similar_vecs(v1, bar[2][0]) and similar_vecs(v2, bar[2][1]):
                        if not any([np.all(before==p) for p in holder[i]]): holder[i].append(before)
                        if not any([np.all(middle==p) for p in holder[j]]): holder[j].append(middle)
                        if not any([np.all(after==p) for p in holder[k]]): holder[k].append(after)

                foo.append(((before, middle, after), (i,j,k), (v1, v2)))

    return holder

if __name__ == '__main__':
    for i,j,k in itertools.combinations(range(5), 3):
        print(i,j,k)
    