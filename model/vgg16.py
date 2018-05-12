import argparse
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D


parser = args.ArgumentParser()
parser.add_argument('--n', '--n_classes', type=int, help='the numpber of classes')

# create the base pre-trained model
base_model = VGG16(weights="imagenet", include_top=False)

x = base_model.outputs
x = GlobalAveragePooling2D()(x)
#いるのかどうかよくわからんからとりあえずつけてる実験してみてもいいかも
x = Dense(512, activation='relu')(x)
out = Dense(args.n_classes, activation='softmax')

model = Model(inputs=base_model.input, outputs=out)
