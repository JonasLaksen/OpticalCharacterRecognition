import cv2
from sklearn.decomposition import PCA
import random
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

fnames = [(i, fname) for i in range(26) for fname in glob.glob('chars/{}/*.jpg'.format(chr(ord('a') + i)))]
random.shuffle(fnames)
fnames_train = fnames[int(len(fnames) * 1.0 / 5.0):]
fnames_test = fnames[: int(len(fnames) * 1.0 / 5.0)]

def read_img(path):
    gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)
    return gray

pca = PCA(n_components=30)
X_train, y_train = [read_img(fname).flatten() for (_, fname) in fnames_train], [i for (i, _) in fnames_train]
X_test, y_test = [read_img(fname).flatten() for (_, fname) in fnames_test], [i for (i, _) in fnames_test]
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
clf = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='brute')
clf.fit(X_train, y_train)
print 'K Nearest Neighbor accuracy: {}'.format(clf.score(X_test, y_test))
svm = SVC(C=4,probability=True)
svm.fit(X_train,y_train)
print 'SVM accuracy: {}'.format(svm.score(X_test, y_test))
print len(X_test)


# Find the best squares when multiple squares are overlapping
def post_process(L, i):
    if i == 0:
        return L
    first = L[0]
    in_same_region = filter(
        lambda x: x[0] < first[0] + 20 and x[0] > first[0] - 20 and x[1] < first[1] + 20 and x[1] > first[1] - 20, L)[
                     :2]
    without = [x for x in L if x not in in_same_region]
    maxx = (0, None)
    for x in in_same_region:
        if x[2] > maxx[0]:
            maxx = (x[2], x)
    without.append(maxx[1])
    return post_process(without, i - 1)


def detect(path, p_threshold=0.6, step_size=5):
    det_img = read_img(path)
    (height, width) = det_img.shape
    squares = [(y, x, det_img[y:y + 20, x:x + 20].flatten()) for y in range(0, height, step_size) for x in
               range(0, width, step_size) if y <= height - 20 and x <= width - 20]
    max_probabilities = filter(lambda x: x[2] >= p_threshold,
                               map(lambda x: (x[0][0], x[0][1], max(x[1])),
                                   zip(squares, svm.predict_proba(
                                       pca.transform(map( lambda x: x[ 2], squares))))))

    ok = post_process(max_probabilities, len(max_probabilities))
    im = cv2.imread(path)
    for (x, y, _) in ok:
        cv2.rectangle(im, (y, x), (y + 20, x + 20), (0, 255, 0), 2)
    cv2.imwrite('out/{}'.format(path.split('/')[-1]), im)

for fname in glob.glob('detection-images/*.jpg'):
    detect(fname)
