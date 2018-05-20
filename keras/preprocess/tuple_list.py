from pathlib import Path
import json
from keras.utils import np_utils
import numpy as np


def read_json(path):
    f = open(path, 'r')
    dict_classname = json.load(f)
    return dict_classname

def make_tuple_list(n_classes, files, dict_classname):
    paths = []
    labels = []
    for tf in files:
        name = tf.name
        c = name.split('_')[0]
        c = dict_classname[c]
        c = np_utils.to_categorical(c, n_classes)
        paths.append(tf)
        labels.append(c)
    return paths, labels

def make_path_and_class_tuple_list(n_classes, path='./data/', mode='train'):
    ################################################################
    # input : nothing
    #output : train_list  and valid_list
    #             like   [(train_path_0, train_0_class), ..., (train_path_n, train_n_class)] ,
    #                     [(val_path_0, val_0_class], ... , [(val_path_n, val_n_class) ]
    ################################################################

    d = path
    c = 'classname.json'
    t = mode
    classname = Path(d).joinpath(c)
    folder = Path(d).joinpath(t)

    #read_json function
    dict_classname = read_json(classname)

    #get files list
    files = list(folder.glob('*'))

    #make_tuple_list function
    paths, labels = make_tuple_list(n_classes, files, dict_classname)

    return paths, labels
