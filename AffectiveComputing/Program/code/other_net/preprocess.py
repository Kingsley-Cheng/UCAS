import numpy as np
import pandas as pd
from PIL import Image
import os

train_path = './data/train/'
valid_path = './data/valid/'
data_path = './fer2013.csv'

def make_dir():
    for i in range(7):
        train = os.path.join(train_path,str(i))
        valid = os.path.join(valid_path, str(i))
        if not os.path.exists(train):
            os.makedirs(train)
        if not os.path.exists(valid):
            os.makedirs(valid)
            
def CreateDataset():
    data = pd.read_csv(data_path)
    num = len(data)
    train_index = [1 for i in range(7)]
    valid_index = [1 for i in range(7)]
    for idx in range(num):
        emotion = data.iloc[idx][0]
        image = data.iloc[idx][1]
        usage = data.iloc[idx][2]
        data_array = list(map(float, image.split()))
        data_array = np.asarray(data_array)
        image = np.reshape(data_array,(48,48))
        im = Image.fromarray(image)
        im = im.convert('L')
        if usage=='Training':
            image_path = os.path.join(train_path,str(emotion),"{}.jpg".format(train_index[emotion]))
            train_index[emotion] +=1
        else:
            image_path = os.path.join(valid_path,str(emotion),"{}.jpg".format(valid_index[emotion]))
            valid_index[emotion] +=1
        im.save(image_path)
make_dir()
CreateDataset()