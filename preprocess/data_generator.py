from pathlib import Path
import math
from skimage.io import imread
from PIL import Image
from keras.utils import Sequence
from keras.utils import np_utils
import numpy as np

'''
ImageSequence
pairs: 画像ファイルpathとラベルのタプルのリスト

pairs = [
    ('./data/train/image1.png', 0),
    ('./data/train/image2.png', 1),
    ...
]
(need to get label before ImageSequence)

self.x: list of path to the images
self.y: the associated classes.

Kerasはラベルを数値ではなく、0or1を要素に持つベクトルで扱うらしい
つまりあるサンプルに対するターゲットを「3」だとすると
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]みたいな感じにしなければならない。
---->np_utils.to_categorical

keras documentより
Sequenceは__getitem__と__len__メソッドを実装しなければなりません．
エポックの間にデータセットを修正したい場合にはon_epoch_endを実装すべきです．
__getitem__メソッドは完全なバッチを返すべきです．
ß
疑問
on_epoch_endがないとデータセットがランダムにオーギュメンテーショんされないのか？されるのか？
きになるので、on_epoch_endありなしで、画像を出して確認してみることにする。
それに対する仮説
オーギュメンテーションをその設定のまま行うのであれば、よくて、ランダム変数とかを変えたいならこれがいる？
画像を変更する時もこれがいる

__getitem__ の返り値はバッチの画像の配列とバッチの正解ラベル配列
'''

class ImageSequence(Sequence):
    def __init__(self, x_set, y_set, num_classes, root='.', batch_size=1):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size

    def __getitem__(self, idx):
        # バッチサイズ分取り出す
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # 画像を1枚づつ読み込んで、前処理をする
        b_x = []
        for file_name in batch_x:
            img = Image.open(file_name)
            imgdata = np.array(img)
            img.close()
            b_x.append(imgdata)
#        batch_x = np.array([self.preprocess(imread(file_name)) for file_name in batch_x])

        return np.array(b_x), np.array(batch_y)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    #def

    def preprocess(self, image):
        # いろいろ前処理
        #(image)
        return image


'''
How to USE

train_gen = ImageSequence(train_pairs, num_classes, batchsize)
valid_gen = ImageSequence(valid_pairs, num_classes, batchsize)
model.fit_generator(
    generator=train_gen,
    epochs=epoch,
    steps_per_epoch=len(train_gen),
    verbose=1,
    validation_data=valid_gen,
    validation_steps=len(valid_gen))

'''
