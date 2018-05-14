from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

def vgg16(n_classes, weights='imagenet'):
    # create the base pre-trained model
    base_model = VGG16(weights=weights, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #いるのかどうかよくわからんからとりあえずつけてる実験してみてもいいかも
    #x = Dense(1024, activation='relu')(x) #1024説もある。
    x = Dense(512, activation='relu')(x)

    if n_classes == 2:
        out = Dense(1, activation='sigmoid')(x)
    else:
        out = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=out)

    return model
