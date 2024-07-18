import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
from obj_tools import *
import torch
import torch.optim as optim
import numpy as np
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F 
import csv
import pandas as pd

def get_config():
    config = {
        "alpha": 0.1,
        "resize_size": 224,
        "crop_size": 224,
        "batch_size": 32,
        "dataset": "coco",
        "device": torch.device("cuda:0"),
        "hash_bit":64,
        "lambda": 0.01,
    }
    config = config_dataset(config)
    return config

config = get_config()
device=config["device"]

train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)

obj_model= torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained="True")
obj_model=obj_model.to(device)


def one_hot_label(a):
    a=torch.tensor(a)
    a=a-1
    if len(a)==0:
        b_one=torch.zeros(90)
    else:
        b=F.one_hot(a,90)
        b_sum=sum(b)
        b_one=torch.where(b_sum > 0, 1, b_sum)
    return b_one


def faster_rcnn_detection(imgs,model):
    mybox=[]
    batch_labels=[]
    model.eval()
    for img in imgs:
        mylabel=[]
        _,height,weight=img.shape
        # 将图像输入神经网络模型中，得到输出
        output = model(img.unsqueeze(0))
        scores = output[0]['scores'].cpu().detach().numpy()     # 预测每一个obj的得分
        bboxes = output[0]['boxes'].cpu().detach().numpy()
        labels=output[0]['labels'].cpu().detach().numpy()     # 预测每一个obj的边框
        obj_index = np.argwhere(scores>0.7).squeeze(axis=1).tolist() 
        for i in obj_index:
            mylabel.append(labels[i])

        one_hot=one_hot_label(mylabel)
        #b_one=one_hot.numpy().tolist()
        batch_labels.append(one_hot.unsqueeze(0))
    mybatch_labels=torch.cat(batch_labels)
    return mybatch_labels

def train_val(config):
    with open('obj_data/coco_obj.csv', 'w') as file:
    # 创建csv.writer对象
        writer = csv.writer(file)
        
        for image, label, ind in train_loader:
            
            image = image.to(device)
            
            label = label.to(device)
        
            im_label=faster_rcnn_detection(image,obj_model)
            
            label_csv=im_label.tolist()
            for data in label_csv:
                writer.writerow(data)


if __name__ == "__main__":
    config = get_config()
    print(config)
    train_val(config)