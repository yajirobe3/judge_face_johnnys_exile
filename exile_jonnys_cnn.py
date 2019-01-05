from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras
import numpy as np

classes = ["exile", "jonnys"]
num_classes = len(classes)
image_size = 50

# メイン関数の定義
def main():
    x_train, x_test, y_train, y_test = np.load("judge_exile_johnnys.npy")
    # データの正規化
    x_train = x_train.astype("float") / 256
    x_test = x_test.astype("float") / 256
    # ラベルのone-hot-vector(正解は1, 他は0となるような処理)
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    # モデルの生成
    model = model_train(x_train, y_train)

    # モデルの評価
    model_eval(model, x_test, y_test)

def model_train(x, y):
    #層の定義
    model = Sequential()
    model.add(Conv2D(32,(3,3), padding='same', input_shape=x.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))#クラスの数（今回は画像のクラスが2つ）
    model.add(Activation('softmax'))

    # 最適化の定義
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                    optimizer=opt, metrics=['accuracy'])

    model.fit(x, y, batch_size=32, epochs=25)

    # モデルの保存
    model.save('./exile_johnnys_cnn.h5')

    return model

def model_eval(model, x, y):
    scores = model.evaluate(x, y, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy: ', scores[1])

if __name__ == "__main__":
    main()
