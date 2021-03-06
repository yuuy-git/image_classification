from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

def vgg19(n_classes, weights='imagenet'):
    base_model = VGG19(weights=weights, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #x = Dense(1024, activation='relu')(x) #1024説もある。
    x = Dense(512, activation='relu')(x)
    if n_classes == 1:
        out = Dense(1, activation='sigmoid')(x)
    else:
        out = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=out)

    return model, base_model
