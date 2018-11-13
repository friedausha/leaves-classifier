#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 14:36:49 2018

@author: dwi
"""
from imutils import paths
import cv2
import time
import numpy as np

if __name__ == "__main__":
    data = []
    labels = []
    imgPath = []
    t = time.time()
    detector = cv2.BRISK_create()
    for imagePath in paths.list_images("RGB1"):
        im = cv2.imread(imagePath)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        (kp, des) = detector.detectAndCompute(im, None)
        daun = imagePath.split("/")[-2]
        data.append(des)
        labels.append(daun)
        imgPath.append(imagePath)
    print("Jumlah class = %d daun %d foto" % (len(np.unique(labels)), len(data)))
    elapsed = time.time() - t
    print("Time Elpased = %g" % elapsed)

    scoreAll = []
    labelAll = []

    for i in range(len(data)):
        score, idx, bestScore = 0, 0, 0
        for j in range(len(data)):
            if i == j:
                continue

            des1 = data[i]

            des2 = data[j]

            #print("daun %s (%s)  -%s (%s)  des=%d %d" % (
            #labels[i], imgPath[i], labels[j], imgPath[j], len(des1), len(des2)))
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            #print(len(matches[0]))
            # Apply ratio test
            good = []
            print (len(matches[0]))
            for m,n in matches:
                if m.distance < 0.95 * n.distance:
                    good.append([m])

            if score < len(good):
                score = len(good)
                idx = j
        scoreAll.append(score)
        labelAll.append(idx)
    #            print("data ke %d [%s] terdeteksi %d[%s] dengan score %g"%(i,labels[i],idx,labels[idx],score))

    scoreAvg = []
    for i in range(0, len(data)):
        idx = labelAll[i]

        if labels[i] == labels[idx]:
            s = "VALID"
            scorenya = 1
        else:
            s = "UNVAL"
            scorenya = 0
        scoreAvg.append(scorenya)
        print("%s\tdata ke %d[%s] terdeteksi %d[%s] dengan score=%g" % (s, i, labels[i], idx, labels[idx], scoreAll[i]))
    print("rata2x %s score %g" % (np.mean(scoreAvg), np.mean(scoreAll)))
