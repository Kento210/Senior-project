#!/usr/bin/env python3
# coding: utf-8
import cv2
# ArUcoのライブラリを導入
aruco = cv2.aruco

# 4x4のマーカー、IDは50までの辞書を使用
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

cnt = 0

def recognizeArMarker():
    for i in range(cnt + 1):
        # 入力ファイル名
        input_file_nm = "test_id0.png"
        # 出力ファイル名
        output_file_nm = "ar_detection" + str(i) + ".png"
        # 入力ファイルの読み込み
        input_img = cv2.imread(input_file_nm)
        if input_img is None:
            print("画像ファイルが読み込めませんでした。")
            continue
        # ArUcoマーカの検出
        corners, ids, rejectedCandidates = aruco.detectMarkers(input_img, dictionary, parameters=parameters)
        # 検出されたマーカーのIDをコンソールに出力
        if ids is not None:
            print(f"検出されたマーカーのID: {ids.ravel()}")
        else:
            print("マーカーが検出されませんでした。")
        # ArUcoマーカの検出結果の描画
        ar_image = aruco.drawDetectedMarkers(input_img, corners, ids)
        # ArUcoマーカの検出結果をファイル出力
        cv2.imwrite(output_file_nm, ar_image)

if __name__ == "__main__":
    recognizeArMarker()
