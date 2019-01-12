from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ["exile", "johnnys"]
num_classes = len(classes)
image_size = 50
num_testdata = 100

X_train = []#画像の配列データ
X_test = []#ラベル
Y_train = []
Y_test = []

for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        if i >= 160:    break#収集した画像の最小数
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)

        if i < num_testdata:
            X_test.append(data)
            Y_test.append(index)
        else:
            for angle in range(-20, 20, 5):
                # 回転
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)

                # 反転
                img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)

# tensorflowが扱いやすいnumpy配列に変換
x_train = np.array(X_train)
x_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)

xy = (x_train, x_test, y_train, y_test)
np.save("./judge_exile_johnnys_aug.npy", xy)
