from PIL import Image
from os import listdir
import numpy as np
from sklearn.model_selection import train_test_split

def make(folder_path, *, resize: int=None, test_size=None):
    classes = listdir(folder_path)
    labels = []
    datas = []
    for i, imgs in enumerate(classes):
        for img_path in listdir(folder_path + "/" + imgs):
            try:
                img = Image.open(folder_path + "/" + imgs + "/" + img_path)
            except:
                continue
            if resize != None:
                img = img.resize((resize, resize))
            datas.append(np.array(img))
            labels.append(np.array(i))
    labels = np.array(labels)
    datas = np.array(datas)
    return train_test_split(datas, labels)
