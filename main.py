import matplotlib.pyplot as plt
import cv2             #opencvのimport
import imutils         #Matplotlib画像の表示などの基本的な画像処理機能のimport
import Stitcher        #Stitcherクラスのimport

#入力画像の読み込み
imageA = cv2.imread("test1.png")
imageB = cv2.imread("test2.png")

#RGBをBGRに変更
imageA = cv2.cvtColor(imageA, cv2.COLOR_RGB2BGR)
imageB = cv2.cvtColor(imageB, cv2.COLOR_RGB2BGR)

#画像のサイズを変更
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)

#Stitcherクラスのインスタンス生成　
stitcher = Stitcher.Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB])

# 画像の保存、表示
plt.figure(figsize=(5, 5))
plt.imshow(vis), plt.xticks([]), plt.yticks([])
plt.savefig("vis.png")
plt.show()
plt.figure(figsize=(5, 5))
plt.imshow(result), plt.xticks([]), plt.yticks([])
plt.savefig("result.png")
plt.show()