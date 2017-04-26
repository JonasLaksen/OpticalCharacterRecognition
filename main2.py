import cv2
import random
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

fnames = [(i, fname) for i in range(26) for fname in glob.glob('chars/{}/*.jpg'.format(chr(ord('a') + i)))]
random.shuffle(fnames)
fnames_train = fnames[int(len(fnames) * 1.0 / 5.0):]
fnames_test = fnames[: int(len(fnames) * 1.0 / 5.0)]

def read_img(path):
    gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image

clf = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='brute')
clf.fit([read_img(fname).flatten() for (_, fname) in fnames_train], [i for (i, _) in fnames_train])
print clf.score([read_img(fname).flatten() for (_, fname) in fnames_test], [i for (i, _) in fnames_test])
print len(fnames_test)
print clf.predict_proba([np.array([0 for _ in range(20 * 20)])])

det_img = read_img('detection-images/detection-2.jpg')
(height, width) = det_img.shape
detections = []
squares = [(y,x,det_img[y:y + 20, x:x + 20].flatten()) for y in range(0, height, 5) for x in range(0, width, 5) if y <= height - 20 and x <= width - 20]
max_probabilities = filter(lambda x: x[2] >= 0.5, map(lambda x:(x[0][0],x[0][1],max(x[1])), zip(squares, clf.predict_proba(map(lambda x:x[2], squares)))))

def lol(L, i):
    if i == 0:
        return L
    first = L[0]
    in_same_region = filter(lambda x: x[0] < first[0] + 20 and x[0] > first[0] - 20 and x[1] < first[1] + 20 and x[1] > first[1] - 20, L)[:2]
    without = [x for x in L if x not in in_same_region]
    maxx = (0,None)
    for x in in_same_region:
        if x[2] > maxx[0]:
            maxx = (x[2], x)
    without.append(maxx[1])

    return lol(without, i - 1)


ok = lol(max_probabilities, len(max_probabilities))
im = cv2.imread('detection-images/detection-2.jpg')
for (x,y,_) in ok:
    cv2.rectangle(im, (y,x),(y+20,x+20), (0,255,0) , 2)
cv2.imwrite('okk.jpg', im)
print ok



print max_probabilities
