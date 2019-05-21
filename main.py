#coding:utf-8
import matplotlib.pyplot as plt
import cv2
import imutils
import Stitcher


imageA = cv2.imread("test1.png")
imageB = cv2.imread("test2.png")

imageA = cv2.cvtColor(imageA, cv2.COLOR_RGB2BGR)
imageB = cv2.cvtColor(imageB, cv2.COLOR_RGB2BGR)
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)
# stitch the images together to create a panorama
stitcher = Stitcher.Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# show the images

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1);
plt.imshow(imageA), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2);
plt.imshow(imageB), plt.xticks([]), plt.yticks([])
plt.show()
plt.figure(figsize=(10, 10))
plt.imshow(vis), plt.xticks([]), plt.yticks([])
plt.show()
plt.figure(figsize=(10, 10))
plt.imshow(result), plt.xticks([]), plt.yticks([])
plt.show()