import numpy as np
import imutils
import cv2

# 特徴点をとり、画像を結合させるクラス
class Stitcher:

    # コンストラクタ
    def __init__(self):
        # opencvのバージョン確認
        self.isv3 = imutils.is_cv3()


    # 特徴量を調べ、画像を結合するメソッド
    def stitch(self, images, ratio=0.75, reprojThresh=4.0):

        # ローカル変数の設定
        (imageB, imageA) = images

        # それぞれの画像におけるキーポイントの座標と特徴量記述子を求める
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # キーポイントのマッチング
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # マッチが弱かった時はNULLを返して終わりにする
        if M is None:
            return None

        # ローカル変数の設定
        (matches, H, status) = M
        # アウトプット画像の生成：射影変換を行う
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageB.shape[0]))
        # アウトプット画像にImageBを代入
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # マッチ画像の生成
        vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)

        # マッチ画像とアウトプット画像を返す
        return (result, vis)


    # キーポイントを調べるメソッド
    def detectAndDescribe(self, image):

        # グレースケールに変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # OpenCV3系列なら以下の処理を行う
        if self.isv3:
            # ORB検出器を作る
            descriptor = cv2.ORB_create()
            # ORBでキーポイントと特徴記述子を求める
            (kps, features) = descriptor.detectAnCompute(image, None)

        # OpenCV2系列なら以下の処理
        else:
            # ORB検出器を作る
            detector = cv2.ORB_create()
            # ORBでキーポイントを計算
            kps = detector.detect(gray)
            # ORB検出器を作る
            extractor = cv2.ORB_create()
            # ORBでキーポイントと特徴記述子を求める
            (kps, features) = extractor.compute(gray, kps)

        # キーポイントの座標をNumPy配列に入れ直す
        kps = np.float32([kp.pt for kp in kps])

        # キーポイントと特徴量を返す
        return (kps, features)


    # キーポイントをマッチングさせるメソッド
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):

        # 2つの画像の特徴量を計算するクラスの作成
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        # 各特徴点に対して、上位2個のマッチング結果をrawMatchesに代入
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        # 変数の初期化
        matches = []

        # D.Loweが提唱したRatio Testで、マッチング結果を間引いて保存する
        for m in rawMatches:
            # 近い二つの特徴点が存在する際に、より大きい方の特徴点（m[1].distance）の距離を少し小さくしても、
            # 小さい方の特徴点の距離(m[0].distance)が小さければ採択する
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 4つ以上のマッチングが見つかった場合はホモグラフィ行列を見つける
        if len(matches) > 4:
            # マッチングした座標の保存
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # ホモグラフィ行列とマッチした点のステータスを保存
            # ステータスは{0,1}で表され、1の点は使用された点で，0の点は使用されていない点
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # マッチした点とホモグラフィ行列、そのステータスを返す
            return (matches, H, status)

        # 4つ以上のマッチング点がない場合はホモグラフィ変換できないためNoneを返す
        return None


    # マッチした点同士を結んだ画像を作る
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):

        # 入力画像のサイズを算出
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]

        # 縦幅のより大きい方の数値と、横幅の和をサイズにした３チャネルの画像を生成
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")

        # アウトプット画像にimageAとimaageBを並べる
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # matchのステータスが1ならば線を引く
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # マッチ画像を返す
        return vis
