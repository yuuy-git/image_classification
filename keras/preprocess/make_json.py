import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path',type=str, default='../data/')
args = parser.parse_args()
path = args.path


############################
#Args of ImageDataGenerator#
############################
#https://keras.io/ja/preprocessing/image/
data_gen_args_train=dict(featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    zca_epsilon=1e-06,
                    rotation_range=0.0,
                    width_shift_range=0.0,
                    height_shift_range=0.0,
                    brightness_range=None,
                    shear_range=0.0,
                    zoom_range=0.0,
                    channel_shift_range=0.0,
                    fill_mode='nearest',
                    cval=0.0,
                    horizontal_flip=False,
                    vertical_flip=False,
                    rescale=None,
                    preprocessing_function=None,
                    data_format=None,
                    validation_split=0.0)


data_gen_args_test=dict(featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    zca_epsilon=1e-06,
                    rotation_range=0.0,
                    width_shift_range=0.0,
                    height_shift_range=0.0,
                    brightness_range=None,
                    shear_range=0.0,
                    zoom_range=0.0,
                    channel_shift_range=0.0,
                    fill_mode='nearest',
                    cval=0.0,
                    horizontal_flip=False,
                    vertical_flip=False,
                    rescale=None,
                    preprocessing_function=None,
                    data_format=None,
                    validation_split=0.0)


#############################################
#Args of train_generator.flow_from_directory#
#############################################
#https://keras.io/ja/preprocessing/image/
#classes --->list class subdirectory(['dogs','cats'])
#classes --->class_mode 'categorical', 'binary', 'sparse', 'input', 'None'
#            default categorical 2dims one-hot label
#                    binary      1dim  2values label
#                    sparse      1dim  int     label
#                    input       return input itself (for autoencoder)
#                    None        return no label (predictの時など)
#save_to_dir ----> Path, augmentationの画像を見たい時など
#save_prefix ----> save_to_dir時、画像を保存する時につけるプリフィックス(接頭辞)
#save_format ----> save_to_dir時、'png','jpeg'

train_generator_args = dict(
    directory= path,  #may need change
    target_size=(256, 256),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix='',
    save_format='png',
    follow_links=False,
    subset=None,
    interpolation='nearest'
    )



test_generator_args = dict(
    directory=path, #may need change
    target_size=(256, 256),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix='',
    save_format='png',
    follow_links=False,
    subset=None,
    interpolation='nearest'
    )

def make_json(dict, name):
    f = open(path+name+'.json','w')
    json.dump(dict, f, indent = 0)

make_json(data_gen_args_train, 'data_gen_args_train')
make_json(data_gen_args_test, 'data_gen_args_test')
make_json(train_generator_args, 'train_generator_args')
make_json(test_generator_args, 'test_generator_args')
