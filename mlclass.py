import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("C:/Users/danie/Desktop/practicapython/MultiLC/MAFood121"))
from tqdm import tqdm, tqdm_notebook
# -------------------------------------------------------------------------------------------------------------------------------

import os
import sys
import cv2
import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.models as models
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(42) # try and make the results more reproducible
BASE_PATH = 'C:/Users/danie/Desktop/practicapython/MultiLC/MAFood121/'

# ------------------------------------------------------------------------------------------------------------------------------
#print(os.listdir("C:/Users/danie/Desktop/practicapython/MultiLC/MAFood121/images"))
epochs = 35
batch_size = 64
MICRO_DATA = True # very small subset (just 3 groups)
SAMPLE_TRAINING = False # make train set smaller for faster iteration
IMG_SIZE = (384, 384)
# ------------------------------------------------------------------------------------------------------------------------------

#Ingredients for each class
f = open(BASE_PATH + '/annotations/foodgroups.txt', "r")
ingredients = f.read().split('\n')
f.close()
# ------------------------------------------------------------------------------------------------------------------------------
#Classes
f = open(BASE_PATH + '/annotations/dishes.txt', "r")
classes = f.read().split('\n')
f.close()
# ------------------------------------------------------------------------------------------------------------------------------
#Base Ingredients
f = open(BASE_PATH + '/annotations/baseIngredients.txt', "r") # crear baseingredients para mafood
base_ing = f.read().split('\n')
f.close()
# ------------------------------------------------------------------------------------------------------------------------------
print(ingredients)
print("#######################################################################################")
print(classes)
print("#######################################################################################")
print(base_ing)

#print(base_ing, ingredients, classes)
# ------------------------------------------------------------------------------------------------------------------------------
new_ingredients = []
for arr in ingredients:
    arr = arr.split(",")
    new_ingredients.append(arr)
#print(new_ingredients)
# ------------------------------------------------------------------------------------------------------------------------------

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
#df = pd.DataFrame(mlb.fit_transform(new_ingredients),columns=mlb.classes_) #binary encode ingredients
# ------------------------------------------------------------------------------------------------------------------------------

#df["target"] = classes
#food_dict = df
# -------------------------------------------------------------------------------------------------------------------------------

#train
f = open(BASE_PATH + '/annotations/train.txt', "r")
train_images = f.read().split('\n')
f.close()
f = open(BASE_PATH + '/annotations/train_lbls_ff.txt', "r")
train_labels = f.read().split('\n')
f.close()

#val
f = open(BASE_PATH + '/annotations/val.txt', "r")
val_images = f.read().split('\n')
f.close()
f = open(BASE_PATH + '/annotations/val_lbls_ff.txt', "r")
val_labels = f.read().split('\n')
f.close()

#test
f = open(BASE_PATH + '/annotations/test.txt', "r")
test_images = f.read().split('\n')
f.close()
f = open(BASE_PATH + '/annotations/test_lbls_ff.txt', "r")
test_labels = f.read().split('\n')
f.close()

#train_images
len(train_labels)

train_images = ["C:/Users/danie/Desktop/PracticaTorch/multilalbel/MAFood121/images/" + s + ".jpg" for s in train_images]
all_img_df = pd.DataFrame({'path': train_images, 'class_id': train_labels})
val_images = ["C:/Users/danie/Desktop/PracticaTorch/multilalbel/MAFood121/images/" + s + ".jpg" for s in val_images]
val_img_df = pd.DataFrame({'path': val_images, 'class_id': val_labels})
test_images = ["C:/Users/danie/Desktop/PracticaTorch/multilalbel/MAFood121/images/" + s + ".jpg" for s in test_images]
test_img_df = pd.DataFrame({'path': test_images, 'class_id': test_labels})

all_img_df = all_img_df[:-1]
val_img_df = val_img_df[:-1]
test_img_df = test_img_df[:-1]
#print("-------------------------------------------------------------------------------------------------")

all_img_df['class_name'] = all_img_df['path'].map(lambda x: os.path.split(os.path.dirname(x))[-1])
print(all_img_df)
#print("-------------------------------------------------------------------------------------------------")

val_img_df['class_name'] = val_img_df['path'].map(lambda x: os.path.split(os.path.dirname(x))[-1])
print(val_img_df)
#print("-------------------------------------------------------------------------------------------------")

test_img_df['class_name'] = test_img_df['path'].map(lambda x: os.path.split(os.path.dirname(x))[-1])
print(test_img_df)
#print("-------------------------------------------------------------------------------------------------")


#food_dict = food_dict.drop('', 1)
#food_dict = food_dict.dropna(axis=1) #food_dict = food_dict.drop(1)

#print(food_dict)


#Dataframe for train images
new_data = []
for index, row in all_img_df.iterrows():
    #get binary encoding ingredients from lookup
    food = row["class_name"]
    path = row["path"]
    class_id = row["class_id"]
    #binary_encod = food_dict.loc[food_dict["target"] == food]
    #binary_encod["path"] = path
    #binary_encod["class_id"] = class_id
    #print(binary_encod)
    #print((list(binary_encod.columns.values)))
    #print(len(np.array(binary_encod)[0]))
    #new_data.append(np.array(binary_encod)[0])
    
#Dataframe for test images
test_data = []
for index, row in test_img_df.iterrows():
    #get binary encoding ingredients from lookup
    food = row["class_name"]
    path = row["path"]
    class_id = row["class_id"]
    #binary_encod = food_dict.loc[food_dict["target"] == food]
    #binary_encod["path"] = path
    #binary_encod["class_id"] = int(class_id)
    #test_data.append(np.array(binary_encod)[0])

test_df = pd.DataFrame(test_data, columns = col_names)

train_df.to_hdf('train_df.h5','df',mode='w',format='table',data_columns=True)
val_df.to_hdf('val_df.h5','df',mode='w',format='table',data_columns=True)
test_df.to_hdf('test_df.h5','df',mode='w',format='table',data_columns=True)
