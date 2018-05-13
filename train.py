import argparse
from pathlib import Path
from preprocess.tuple_list import make_path_and_class_tuple_list
from preprocess.data_generator import ImageSequence



parser = argparse.ArgumentParser()
parser.add_argument('num_classes',type=int)
parser.add_argument('model', type=str, help='choose model vgg16, vgg19, rensnet50')
parser.add_argument('--data_path', type=str, default='./data/')
parser.add_argument('--file_type','--k',type=str,default='.jpg', help='.jpg, .png')
parser.add_argument('--batch_size','--b',type=int,default='16', help='batch_size')

#parser.add_argument('--data_aug')
args = parser.parse_args()
path = args.data_path
num_classes = args.num_classes
batch_size = args.batch_size

#############################################################
#データを使えるように整える
#train, valそれぞれに
#pathのリストとそれに対応するfrom 0 to n_classesの正解ラベルを用意。
#############################################################
train_paths, train_labels = make_path_and_class_tuple_list(path=path, mode='train')
valid_paths, valid_labels = make_path_and_class_tuple_list(path=path, mode='validation')

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
train_gen = ImageSequence(train_pairs, num_classes, batch_size)
valid_gen = ImageSequence(valid_pairs, num_classes, batch_size)

#########################
#modelを定義する。
#loss関数などを定義する
#重みを引き継ぐかを定義する　→　vgg16.pyの方かも
#重みを引くつぐのなら、どこの重みを固定するかを指定する
########################


####################
#
#####################


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
