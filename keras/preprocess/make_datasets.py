import os
import glob
import shutil
import argparse
from pathlib import Path
import json

'''
usage
 python make_datasets.py dir_path num_class

This program is prepareation for datasets
including rename, move files

before execute this program, need under datasets
dir_path
  - train
  |   | - class_0 : the name is class name ex)dog
  |   |             put this class's images under this folder
  |   | - class_1 : the name is class name ex)cat
  |   |             put this class's images under this folder
  |   | - class_2 : the name is class name ex)horse
  |   |             put this class's images under this folder
  |   | - class_3 : the name is class name ex)rabbit
  |                 put this class's images under this folder
  - validation(have the same constitution of subfolders )
  -class.txt : class_name

'''

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('num_class',type=int)
parser.add_argument('--file_type','--k',type=str,default='.jpg', help='.jpg, .png')
args = parser.parse_args()

path = args.path
num_class = args.num_class
kakucho = args.file_type

'''
次、多分いる
ex) class.txt
dog
cat
horse
rabbit

#convert txt to list
txt_path = path + 'class.txt'
with open(txt_path, "r") as f:  #　txt形式の読み込み
    class_lists = f.read()                # ひとまとまりのデータとして読み込む
class_lists = data.split("\n")            #　改行コードで1行ずつに分割
'''

###################
#prepare directory#
###################
#assert len(class_lists) == num_class, 'not match the number of classes'

if not os.path.exists("../data"):
    os.mkdir('../data')
if not os.path.exists("../data/train"):
    os.mkdir('../data/train')
if not os.path.exists("../data/validation"):
    os.mkdir('../data/validation')

#for x in class_lists:
#    os.mkdir('../data/'+ x)

#################
#rename and move file#
#################

train_folder = Path(path).joinpath('train')
validations_folder = Path(path).joinpath('validation')

sub_folders = list(train_folder.glob('*'))

#check if t_subfolder = v_sub_folders
v_sub_folders = list(validations_folder.glob('*'))

assert len(sub_folders)==len(v_sub_folders), 'not match the number of trainsubfolders and valsubfolders'

#make class.json
classnames = {}
for i, t in enumerate(sub_folders):
    classname = Path(t).name
    classnames.setdefault(classname, i)
f = open('../data/classname.json', 'w')
json.dump(classnames, f, indent=0)

#move and rename files
for s in sub_folders:
    classname = Path(s).name
    val_folder = validations_folder.joinpath(classname)
    t_files = Path(s).glob('*')
    v_files = val_folder.glob('*')
    for i, f in enumerate(t_files):
        dst = '../data/train/'
        dst += classname
        dst += '_'
        dst += str(i)
        dst += kakucho
        shutil.copy(f, dst)
    for i, f in enumerate(v_files):
        dst = '../data/validation/'
        dst += classname
        dst += '_'
        dst += str(i)
        dst += kakucho
        shutil.copy(f, dst)
