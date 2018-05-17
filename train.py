import argparse
from pathlib import Path
from preprocess.tuple_list import make_path_and_class_tuple_list
from preprocess.data_generator import ImageSequence
from model.vgg16 import vgg16
from model.vgg19 import vgg19
from model.inceptionv3 import inceptionv3
from model.xception import xception
from utils.callback import cb
from utils.plot import plot_history
from keras import losses
from keras import optimizers


parser = argparse.ArgumentParser()
parser.add_argument('num_classes',type=int)
parser.add_argument('model', type=str, help='choose model vgg16, vgg19, rensnet50')
parser.add_argument('--data_path', type=str, default='./data/')
parser.add_argument('--file_type','--k',type=str,default='.jpg', help='.jpg, .png')
parser.add_argument('--batch_size','--b',type=int,default='16', help='batch_size')
parser.add_argument('--pretrained','--p',type=bool,default=True,
                    help='if use pretrained weights, set True. if not use pretrained weight, set False')
parser.add_argument('--config', '--c', type=str, default='default',
                    help = 'default or  manual, if manual, u can choose what layer is fixed')
parser.add_argument('--optimizer', '--o', type=str, default='sgd', help='sgd or Adam u can customize more')


#parser.add_argument('--data_aug')
args = parser.parse_args()
path = args.data_path
num_classes = args.num_classes
batch_size = args.batch_size
print(args.pretrained)
#############################################################
#データを使えるように整える
#train, valそれぞれに
#pathのリストとそれに対応するfrom 0 to n_classesの正解ラベルを用意。
#############################################################
train_paths, train_labels = make_path_and_class_tuple_list(n_classes=num_classes, path=path, mode='train')
valid_paths, valid_labels = make_path_and_class_tuple_list(n_classes=num_classes, path=path, mode='validation')

print('------------')
print(train_paths)
print('------------')
print(train_labels)
print('------------')
print(valid_paths)
print('-------------')
print(valid_labels)

##########################
#kearas.utils.dataloaderにそれを渡す。
##########################
train_gen = ImageSequence(train_paths, train_labels, num_classes, batch_size)
valid_gen = ImageSequence(valid_paths, valid_labels, num_classes, batch_size)

print(train_gen)
print(valid_gen)
#########################
#modelを定義する。
#重みを引き継ぐかを定義する　→　vgg16.pyの方かも
#重みを引くつぐのなら、どこの重みを固定するかを指定する

#loss関数などを定義する

########################


if args.model == 'vgg16':
    if args.pretrained == True:
        model, base_model = vgg16(num_classes, weights='imagenet')
    elif args.pretrained == False:
        model, base_model = vgg16(num_classes, weights=None)

if args.model == 'vgg19':
    if args.pretrained == True:
        model, base_model = vgg19(num_classes, weights='imagenet')
    elif args.pretrained == False:
        model, base_model = vgg19(num_classes, weights=None)

if args.model == 'xception':
    if args.pretrained == True:
        model, base_model = xception(num_classes, weights='imagenet')
    elif args.pretrained == False:
        model, base_model = xception(num_classes, weights=None)

if args.model == 'inceptionv3':
    if args.pretrained == True:
        model, base_model = inceptionv3(num_classes, weights='imagenet')
    elif args.pretrained == False:
        model, base_model = inceptionv3(num_classes, weights=None)



print(model.summary())



#fix weights before VGG16 14layers
if args.config == 'default':
    for layer in base_model.layers:
        layer.trainable=False
elif args.config == 'manual':
    l = input("何番目の層まで固定しますか。int型。＋１して入れてね。")
    for layer in base_model.layers[:l]:
        layer.trainable=False
    for layer in base_model.layers[l:]:
        layer.trainable=True

#model compile
#optimizers
#その他にも設定可能　https://keras.io/ja/optimizers/
if args.optimizer == 'sgd':
    optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
elif args.optimizer == 'Adam':
    optimizer =optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)




if args.num_classes == 1:
    model.compile(optimizer,
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy'])
else:
    model.compile(optimizer,
                  loss=losses.categorical_crossentropy,
                  metrics=['accuracy'])


####################
#fit_generatorを用いてトレインする。
#####################

cb, history = cb()


model.fit_generator(generator=train_gen,
                    steps_per_epoch=None,#Sequenceの方で定義
                    epochs=2, #後で引数に追加する。
                    verbose=1,
                    callbacks=cb,#後で設定
                    validation_data=valid_gen,
                    validation_steps=None,#Sequenceの方で定義
                    class_weight=None,#過小評価されたクラスのサンプルに「より注意を向ける」場合に有用です．
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    shuffle=True,
                    initial_epoch=0)#前回の学習を再開するのに便利です


#plot_history(history)




'''
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator(...)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(...)
'''
