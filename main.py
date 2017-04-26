import numpy as np
import cv2
import glob

fnames_train = []
fnames_test = []
for i in range(26):
    cap = 280
    for j, fname in enumerate(glob.glob('chars/{}/*.jpg'.format(chr(ord('a') + i)))):
        if j < cap:
            fnames_train.append((i, fname))
        else:
            fnames_test.append((i, fname))

def read_img(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gaus = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('out/{}'.format(path.split('/')[-1]), gaus)
    return gaus

sift = cv2.SIFT()
BOW = cv2.BOWKMeansTrainer(2000)
for (i, fname) in fnames_train:
    gray = read_img(fname)
    kp, des = sift.detectAndCompute(gray, None)
    if len(kp) > 0:
        BOW.add(des)
    else:
        fnames_train.remove((i, fname))

dictionary = BOW.cluster()

desc_ext = cv2.DescriptorExtractor_create("SIFT")
bow_img_ext = cv2.BOWImgDescriptorExtractor(desc_ext, cv2.BFMatcher(cv2.NORM_L2))
bow_img_ext.setVocabulary(dictionary)

def feature_extract(path):
    gray = read_img(path)
    return bow_img_ext.compute(gray, sift.detect(gray))

train_desc = []
train_labels = []

for (i, fname) in fnames_train:
    desc = feature_extract(fname)
    if desc != None:
        train_desc.extend(desc)
        train_labels.append(i)

svm = cv2.SVM()
svm.train(np.array(train_desc), np.array(train_labels))
knn = cv2.KNearest(np.array(train_desc), np.array(train_labels))
nb = cv2.NormalBayesClassifier()
nb.train(np.array(train_desc), np.array(train_labels))
correct = 0
wrong = 0
knn_correct = 0
knn_wrong = 0
nbcorrect = 0
nbwrong = 0
for (i,fname) in fnames_test:
    desc = feature_extract(fname)
    if desc != None:
        p, results, neighborResponses, dists = knn.find_nearest(desc, 15)
        if p == i:
            knn_correct += 1
        else:
            knn_wrong += 1
        c = svm.predict(desc)
        if c == i:
            correct += 1
        else:
            wrong += 1
        d = nb.predict(desc)
        if d == i:
            nbcorrect += 1
        else:
            nbwrong += 1

print 'Correct {}'.format(correct)
print 'Wrong {}'.format(wrong)
print 'KNNCorrect {}'.format(knn_correct)
print 'KNNWrong {}'.format(knn_wrong)
print 'NBCorrect {}'.format(nbcorrect)
print 'NBWrong {}'.format(nbwrong)
