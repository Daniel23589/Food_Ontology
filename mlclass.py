import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("MAFood121"))
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
BASE_PATH = 'MAFood121/'

# ------------------------------------------------------------------------------------------------------------------------------
#print(os.listdir("MAFood121/images"))
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
base_ing = f.read().strip().split(', ')
f.close()
# ------------------------------------------------------------------------------------------------------------------------------
print(ingredients)
print("#######################################################################################")
print(classes)
print("#######################################################################################")
print(base_ing)

'''
#print(base_ing, ingredients, classes)
# ------------------------------------------------------------------------------------------------------------------------------
new_ingredients = []
for arr in ingredients:
    arr = arr.split(",")
    new_ingredients.append(arr)
print(new_ingredients)
# ------------------------------------------------------------------------------------------------------------------------------

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
#df = pd.DataFrame(mlb.fit_transform(new_ingredients),columns=mlb.classes_) #binary encode ingredients
# ------------------------------------------------------------------------------------------------------------------------------

#df["target"] = classes
#food_dict = df
# -------------------------------------------------------------------------------------------------------------------------------
'''

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

train_images = ["MAFood121/images/" + s  for s in train_images]
all_img_df = pd.DataFrame({'path': train_images, 'class_id': train_labels})
val_images = ["MAFood121/images/" + s  for s in val_images]
val_img_df = pd.DataFrame({'path': val_images, 'class_id': val_labels})
test_images = ["MAFood121/images/" + s  for s in test_images]
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



from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

train_ingredients = []
train_classid = []
with open(BASE_PATH + '/annotations/train_lbls_ff.txt') as f1:
    for line in f1:
       idx_ingredients = []
       classid = int(line)
       train_classid.append(classid)
       for ing in ingredients[classid].strip().split(","):
           idx_ingredients.append(str(base_ing.index(ing)))
       train_ingredients.append(idx_ingredients)
df_train = pd.DataFrame(mlb.fit_transform(train_ingredients),columns=mlb.classes_) #binary encode ingredients
df_train["path"] = all_img_df['path']
df_train["class_id"] = train_classid 
food_dict_train = df_train
print(df_train)


#Dataframe for train images
new_data = []
for index, row in all_img_df.iterrows():
    #get binary encoding ingredients from lookup
    food = row["class_name"]
    path = row["path"]
    class_id = row["class_id"]
    
    binary_encod = food_dict_train.loc[food_dict_train["path"] == path]
    #binary_encod["path"] = path
    #binary_encod["class_id"] = class_id
    #print(binary_encod)
    #print((list(binary_encod.columns.values)))
    #print(len(np.array(binary_encod)[0]))
    print(np.array(binary_encod)[0])
    new_data.append(np.array(binary_encod)[0])

col_names = list(binary_encod.columns.values)
train_df = pd.DataFrame(new_data, columns = col_names)

print(train_df)

test_ingredients = []
test_classid = []
with open(BASE_PATH + '/annotations/test_lbls_ff.txt') as f1:
    for line in f1:
       idx_ingredients = []
       classid = int(line)
       test_classid.append(classid)
       for ing in ingredients[classid].strip().split(","):
           idx_ingredients.append(str(base_ing.index(ing)))
       test_ingredients.append(idx_ingredients)
df_test = pd.DataFrame(mlb.fit_transform(test_ingredients),columns=mlb.classes_) #binary encode ingredients
df_test["path"] = test_img_df['path']
df_test["class_id"] = test_classid
food_dict_test = df_test
print(df_test)

#Dataframe for test images
test_data = []
for index, row in test_img_df.iterrows():
    #get binary encoding ingredients from lookup
    food = row["class_name"]
    path = row["path"]
    class_id = row["class_id"]
    binary_encod = food_dict_test.loc[food_dict_test["path"] == path]
    #binary_encod = food_dict.loc[food_dict["target"] == food]
    #binary_encod["path"] = path
    #binary_encod["class_id"] = int(class_id)
    test_data.append(np.array(binary_encod)[0])

col_names = list(binary_encod.columns.values)
test_df = pd.DataFrame(test_data, columns = col_names)

train_df.to_hdf('train_df.h5','df',mode='w',format='table',data_columns=True)
#val_df.to_hdf('val_df.h5','df',mode='w',format='table',data_columns=True)
test_df.to_hdf('test_df.h5','df',mode='w',format='table',data_columns=True)
# -------------------------------------------------------------------------------------------------------------------------------

#START HERE
import torchvision.models as models
from tqdm import tqdm, tqdm_notebook, tnrange
#use_cuda = False
#device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(42) # try and make the results more reproducible
BASE_PATH = 'MAFood121/'

print(os.listdir("MAFood121/images"))
epochs = 8
batch_size = 2
SMALL_DATA = False
IMG_SIZE = (384, 384)



train_df = pd.read_hdf("train_df.h5")
#val_df = pd.read_hdf("../input/preloaded/test_df.h5")
test_df = pd.read_hdf("test_df.h5")

if SMALL_DATA:
    train_df = train_df[:128]
    #val_df = test_df[:128]
    test_df = actual_test_df[:128]

col_names = list(train_df.columns.values)

ing_names = col_names[:-3]
targets = ing_names

# -------------------------------------------------------------------------------------------------------------------------------

class DataWrapper(data.Dataset):
    ''' Data wrapper for pytorch's data loader function '''
    def __init__(self, image_df, resize):
        self.dataset = image_df
        self.resize = resize

    def __getitem__(self, index):
        c_row = self.dataset.iloc[index]
        target_arr = []
        for item in c_row[targets].values:
            target_arr.append(item)
        #print(target_arr)
        image_path, target = c_row['path'], torch.from_numpy(np.array(target_arr)).float()  #image and target
        #read as rgb image, resize and convert to range 0 to 1
        image = cv2.imread(image_path, 1)
        if self.resize:
            image = cv2.resize(image, IMG_SIZE)/255.0 
        else:
            image = image/255.0
        image = (torch.from_numpy(image.transpose(2,0,1))).float() #NxCxHxW
        return image, target

    def __len__(self):
        return self.dataset.shape[0]
    
# -------------------------------------------------------------------------------------------------------------------------------

model = models.resnet50(pretrained=True)
# #freeze layers
# for param in model.parameters():
#      param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(targets))

ct = 0
for name, child in model.named_children():
    ct += 1
    if ct < 8:
        for name2, params in child.named_parameters():
            params.requires_grad = False
# -------------------------------------------------------------------------------------------------------------------------------

import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

train_dataset = DataWrapper(train_df, True)
train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=True, batch_size=batch_size, pin_memory=False)

#val_dataset = DataWrapper(val_df, True)
#val_loader = torch.utils.data.DataLoader(val_dataset,shuffle=True, batch_size=batch_size, pin_memory=False)

test_dataset = DataWrapper(test_df, True)
test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=True, batch_size=batch_size, pin_memory=False)

# -------------------------------------------------------------------------------------------------------------------------------

#TRY LATER
def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)

# -------------------------------------------------------------------------------------------------------------------------------

from collections import defaultdict
train_results = defaultdict(list)
train_iter, test_iter, best_acc = 0,0,0
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10, 10))
ax1.set_title('Train Loss')
ax2.set_title('Train Accuracy')
ax3.set_title('Test Loss')
ax4.set_title('Test Accuracy')

f1_scores = defaultdict(list)

for i in tnrange(epochs, desc='Epochs'):
    print("Epoch ",i)
    ## Train Phase
    #Model switches to train phase
    model.train() 
    
    all_outputs = []
    all_targets = []
    # Running through all mini batches in the dataset
    count, loss_val, correct, total = train_iter, 0, 0, 0
    for img_data, target in tqdm_notebook(train_loader, desc='Training'):    
        img_data, target = img_data.to(device), target.to(device)
        
        output = model(img_data) #FWD prop

        loss = criterion(output, target) #Cross entropy loss
        c_loss = loss.data.item()
        ax1.plot(count, c_loss, 'r.')
        loss_val += c_loss

        optimizer.zero_grad() #Zero out any cached gradients
        loss.backward() #Backward pass
        optimizer.step() #Update the weights

        total_batch = (target.size(0) * target.size(1))
        total += total_batch
        output_data = torch.sigmoid(output)>=0.5
        target_data = (target==1.0)
        for arr1,arr2 in zip(output_data, target_data):
            all_outputs.append(list(arr1.cpu().numpy()))
            all_targets.append(list(arr2.cpu().numpy()))
        c_acc = torch.sum((output_data == target_data.to(device)).to(torch.float)).item()
        ax2.plot(count, c_acc/total_batch, 'r.')
        correct += c_acc
        count +=1
        
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    f1score_samples = f1_score(y_true=all_targets, y_pred=all_outputs, average='samples')
    f1score_macro = f1_score(y_true=all_targets, y_pred=all_outputs, average='macro')
    f1score_weighted = f1_score(y_true=all_targets, y_pred=all_outputs, average='weighted')
    recall = recall_score(y_true=all_targets, y_pred=all_outputs, average='samples')
    prec = precision_score(y_true=all_targets, y_pred=all_outputs, average='samples')
    hamming = hamming_score(y_true=all_targets, y_pred=all_outputs)
    
    f1_scores["samples_train"].append(f1score_samples)
    f1_scores["macro_train"].append(f1score_macro)
    f1_scores["weighted_train"].append(f1score_weighted)
    f1_scores["hamming_train"].append(hamming)
    
    train_loss_val, train_iter, train_acc = loss_val/len(train_loader.dataset), count, correct/float(total)
    
    print("Training loss: ", train_loss_val, " train acc: ",train_acc)    
    ## Test Phase
    
    #Model switches to test phase
    model.eval()
    
    all_outputs = []
    all_targets = []
    '''
    #Running through all mini batches in the dataset
    count, correct, total, lost_val = test_iter, 0, 0, 0
    for img_data, target in tqdm_notebook(val_loader, desc='Testing'):
        img_data, target = img_data.to(device), target.to(device)
        output = model(img_data)
        loss = criterion(output, target) #Cross entropy loss
        c_loss = loss.data.item()
        ax3.plot(count, c_loss, 'b.')
        loss_val += c_loss
        #Compute accuracy
        #predicted = output.data.max(1)[1] #get index of max
        total_batch = (target.size(0) * target.size(1))
        total += total_batch
        output_data = torch.sigmoid(output)>=0.5
        target_data = (target==1.0)
        #print("Predictions: ", output_data)
        #print("Actual: ", target_data)
        for arr1,arr2 in zip(output_data, target_data):
            all_outputs.append(list(arr1.cpu().numpy()))
            all_targets.append(list(arr2.cpu().numpy()))
        c_acc = torch.sum((output_data == target_data.to(device)).to(torch.float)).item()
        ax4.plot(count, c_acc/total_batch, 'b.')
        correct += c_acc
        count += 1
    '''
    #print("Outputs: ", len(all_outputs), " x ", len(all_outputs[0]))
    #print("Targets: ", len(all_targets), " x ", len(all_targets[0]))
    
    #F1 Score
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    f1score_samples = f1_score(y_true=all_targets, y_pred=all_outputs, average='samples')
    f1score_macro = f1_score(y_true=all_targets, y_pred=all_outputs, average='macro')
    f1score_weighted = f1_score(y_true=all_targets, y_pred=all_outputs, average='weighted')
    recall = recall_score(y_true=all_targets, y_pred=all_outputs, average='samples')
    prec = precision_score(y_true=all_targets, y_pred=all_outputs, average='samples')
    hamming = hamming_score(y_true=all_targets, y_pred=all_outputs)
    
    f1_scores["samples_test"].append(f1score_samples)
    f1_scores["macro_test"].append(f1score_macro)
    f1_scores["weighted_test"].append(f1score_weighted)
    f1_scores["hamming_test"].append(hamming)
    
    #Accuracy over entire dataset
    test_acc, test_iter, test_loss_val = correct/float(total), count, loss_val/len(test_loader.dataset)
    print("Test set accuracy: ",test_acc)
    
    train_results['epoch'].append(i)
    train_results['train_loss'].append(train_loss_val)
    train_results['train_acc'].append(train_acc)
    train_results['train_iter'].append(train_iter)
    
    train_results['test_loss'].append(test_loss_val)
    train_results['test_acc'].append(test_acc)
    train_results['test_iter'].append(test_iter)
    
    #Save model with best accuracy
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth') 
fig.savefig('train_curves.png')

# -------------------------------------------------------------------------------------------------------------------------------

print("TRAIN")
print("F1 Samples: ", f1_scores["samples_train"])
print("F1 Weighted: ", f1_scores["weighted_train"])
print("Hamming: ", f1_scores["hamming_train"])
print()
print("==============")
print("VALIDATION")
print("F1 Samples: ", f1_scores["samples_test"])
print("F1 Weighted: ", f1_scores["weighted_test"])
print("Hamming: ", f1_scores["hamming_test"])

# -------------------------------------------------------------------------------------------------------------------------------

#Inference on test
model_path = "best_model.pth"
model.load_state_dict(torch.load(model_path))
model.to(device)
for param in model.parameters():
    param.requires_grad = False
model.eval()

#Run predictions
all_outputs = []
all_targets = []
for img_data, target in tqdm_notebook(test_loader, desc='Testing'):
    img_data, target = img_data.to(device), target.to(device)
    output = model(img_data)
    loss = criterion(output, target) #Cross entropy loss
    c_loss = loss.data.item()
    ax3.plot(count, c_loss, 'b.')
    loss_val += c_loss
    #Compute accuracy
    #predicted = output.data.max(1)[1] #get index of max
    total_batch = (target.size(0) * target.size(1))
    total += total_batch
    output_data = torch.sigmoid(output)>=0.5
    target_data = (target==1.0)
    #print("Predictions: ", output_data)
    #print("Actual: ", target_data)
    for arr1,arr2 in zip(output_data, target_data):
        all_outputs.append(list(arr1.cpu().numpy()))
        all_targets.append(list(arr2.cpu().numpy()))
    c_acc = torch.sum((output_data == target_data.to(device)).to(torch.float)).item()
    ax4.plot(count, c_acc/total_batch, 'b.')
    correct += c_acc
    count += 1


#F1 Score
all_outputs = np.array(all_outputs)
all_targets = np.array(all_targets)
f1score_samples = f1_score(y_true=all_targets, y_pred=all_outputs, average='samples')
f1score_macro = f1_score(y_true=all_targets, y_pred=all_outputs, average='macro')
f1score_weighted = f1_score(y_true=all_targets, y_pred=all_outputs, average='weighted')
recall = recall_score(y_true=all_targets, y_pred=all_outputs, average='samples')
prec = precision_score(y_true=all_targets, y_pred=all_outputs, average='samples')
hamming = hamming_score(y_true=all_targets, y_pred=all_outputs)

# -------------------------------------------------------------------------------------------------------------------------------

print("TEST")
print("F1 Samples: ", f1score_samples)
print("F1 Weighted: ", f1score_weighted)
print("Hamming: ", hamming)
