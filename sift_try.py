# https://github.com/adumrewal/SIFTImageSimilarity/blob/master/SIFTSimilarityInteractive.ipynb
import cv2
from tries_utils import *


def computeSIFT(img):
    sift = cv2.xfeatures2d.SIFT_create()
    return sift.detectAndCompute(image, img)


def calculateMatches(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    topResults1 = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            topResults1.append([m])

    matches = bf.knnMatch(des2, des1, k=2)
    topResults2 = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            topResults2.append([m])

    topResults = []
    for match1 in topResults1:
        match1QueryIndex = match1[0].queryIdx
        match1TrainIndex = match1[0].trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2[0].queryIdx
            match2TrainIndex = match2[0].trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(match1)
    return topResults


def calculateScore(matches, keypoint1, keypoint2):
    return 100 * (matches /min(keypoint1, keypoint2))


def predict(target):
    keypoint_target, descriptor_target = computeSIFT(target)
    get_char_img(char, path_to_fonts)
    paths = get_char_lst(char, path_to_fonts)
    imgs = [cv2.imread(p) for p in paths]
    scores = []
    for i in range(7):
        keypoint_suspect, descriptor_suspect = computeSIFT(imgs[i])
        matches = calculateMatches(descriptor_target, descriptor_suspect)
        scores.append(calculateScore(len(matches), len(keypoint_target), len(keypoint_suspect)))
    return max(range(7), key=lambda index: scores[index])
