# -*- coding: utf-8 -*-

import cv2, os, sys

def detect_face(img_path, cascade_path, dir, file, image_size):

    #ファイル読み込み
    image = cv2.imread(img_path)

    re_file_name = dir + "_" + file
    print("Detect face: ", re_file_name)

    #グレースケール変換
    if len(image.shape) == 3:
        height, width, channels = image.shape[:3]
    else:
        height, width = image.shape[:2]
        channels = 1

    if (channels == 3):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(cascade_path)

    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    #facerect = cascade.detectMultiScale(image_gray)

    if len(facerect) > 0:
        for rect in facerect:
            dst = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            dst = cv2.resize(dst, dsize=(image_size, image_size))
            cv2.imwrite("./face_all/" + re_file_name, dst)

def main():
    image_size = 50

    #HAAR分類器の顔検出用の特徴量
    cascade_path = "C:/Anaconda/envs/tf140/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml"

    # オリジナル画像のディレクトリ
    org_img_path = "./temp_image_data/"
    dirs = []
    for i in os.listdir(org_img_path):
        if os.path.isdir(org_img_path + i):
            dirs.append(i)
    if len(dirs) == 0:
        print("not exist original image direcotry")
        sys.exit()

    # フォルダごとのループ処理
    for dir in dirs:
        each_dir_path = org_img_path + dir + "/"
        # 画像ファイルごとのループ処理
        for file in os.listdir(each_dir_path):
            each_img_path = each_dir_path + file
            detect_face(each_img_path, cascade_path, dir, file, image_size)

    print("Finish detect face")

if __name__ == "__main__":
    main()
