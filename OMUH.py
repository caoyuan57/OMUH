import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from tools import *
import torch
import torch.optim as optim
import time
import numpy as np
from torchvision import transforms
import os
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F 
import pandas as pd
from model import Vit
from util import *

def get_config():
    config = {
        "alpha": 0.5,
        "beta":0.3,
        "gamma":0.04,
        "delta":0.5,
        "eta":0.6,
        "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.01, "weight_decay": 1e-4}},
        #"optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5, "weight_decay": 1e-4}},
        "info": "[OMUH]",
        "resize_size": 224,
        "batch_size": 32,
        "dataset": "mirflickr",
        "epoch": 50,
        "test_map": 5,
        "save_path": "save/OMUH",
        "device": torch.device("cuda:0"),
        "bit_list": [16],
    }
    config = config_dataset(config)
    return config
config = get_config()
device=config["device"]

#generate dataloader
train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)

#read object pseudo-label
data = pd.read_csv("obj_data/flickr_obj.csv",header=None)
label_data = np.array(data,dtype=np.float)

#generate object similarity matrix
def s_o(batch_label):
    ll=len(batch_label)
    s_obj = np.ones((ll,ll))
    for i in range(ll):
        for j in range(ll):
            aij=batch_label[i]@batch_label[j]
            bij=batch_label[i]+batch_label[j]
            bij=torch.where(bij > 0, 1, bij)
            bij=bij.sum()
            if bij==0:
                s_obj[i][j]=0
            else:
                s_obj[i][j]= aij/bij
    return s_obj

#Optimize the similarity matrix
def new_s(s_o,im_fea):  
    Ss=s_o
    for i in range(len(Ss)):
        for j in range(len(Ss)):
            if (Ss[i][j]!=0 and Ss[i][j]!=1):
                Ss[i][j]=im_fea[i][j]
            if(Ss[i][j]== 0):
                Ss[i][j]=-1
    return Ss

#Optimize positive and negative samples
def pos_neg(ss,s_hash_rotation):
    pos_sample=[]
    neg_sample=[]
    for i in range(len(s_hash_rotation)):
        for j in range(len(s_hash_rotation)):
            if ss[i][j]==1:
                pos_sample.append(s_hash_rotation[i][j])
            if ss[i][j]==0:
                neg_sample.append(s_hash_rotation[i][j])
    pos_sample=torch.tensor(pos_sample)
    neg_sample=torch.tensor(neg_sample)
    return pos_sample,neg_sample

new_trans=transforms.Compose([ 
                     transforms.RandomHorizontalFlip(),
                     transforms.RandomCrop(224),
                     transforms.Resize([224,224]),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                               ])

new_rotation=transforms.Compose([ 
                     transforms.RandomRotation(45),
                     transforms.RandomCrop(224),
                     transforms.Resize([224,224]),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                               ])

def train_val(config, bit):
    criterion = ObjCenteralLoss(config,bit=bit)
    criterion2 = nn.MSELoss()
    criterion3=ObjContrastiveLoss()
    net=Vit(hash_bit=bit)
    net=net.to(device)
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    Best_mAP = 0
    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")
        net.train()
        train_loss = 0
        for image, label, ind in train_loader:
            im_label=[]
            image = image.to(device)
            for i in ind:
                im_label.append(label_data[i])
            
            im_label=torch.tensor(np.array(im_label))
            im_label=im_label.to(device)
            s_obj=s_o(im_label.float())

            im_fea,im_hash=net(new_trans(image))
            rotation_fea,rotation_hash=net(new_rotation(image))

            s_fea=F.cosine_similarity(im_fea.unsqueeze(1),im_fea.unsqueeze(0),dim=-1)
            s_hash=F.cosine_similarity(im_hash.unsqueeze(1), im_hash.unsqueeze(0), dim=-1)

            s_hash_rotation=F.cosine_similarity(im_hash.unsqueeze(1),rotation_hash.unsqueeze(0),dim=-1)
            
            S_op=new_s(s_obj,s_fea)
            S_op = torch.Tensor(S_op).to(device)
            s_poshash,s_neghash=pos_neg(s_obj,s_hash_rotation)

            optimizer.zero_grad()
            loss_pair=criterion2(s_hash,S_op)
            loss_cen=criterion(im_hash, im_label.float(), ind, config)
            loss_con=criterion3(s_poshash.float(),s_neghash.float())
            loss_rec=criterion2(im_hash,rotation_hash)+criterion2(im_fea,rotation_fea)
            loss_q = (im_hash.abs() - 1).pow(2).mean()
    
            loss= config['alpha']*loss_pair+ config['beta']*loss_cen + config['gamma']*loss_con+ config['delta']*loss_rec+ config['eta']*loss_q
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
    
            tst_binary, tst_label = compute_result(test_loader, net, device=device)

            trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

            # print("calculating map.......")
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])

            if mAP > Best_mAP:
                Best_mAP = mAP
                if "save_path" in config:
                    if not os.path.exists(config["save_path"]):
                        os.makedirs(config["save_path"])
                    print("save in ", config["save_path"])

                    np.save(os.path.join(config["save_path"], config["dataset"] + str(bit) + "-"  + "trn_binary.npy"),
                            trn_binary.numpy())
                    np.save(os.path.join(config["save_path"], config["dataset"] + str(bit) + "-"  + "tst_binary.npy"),
                            tst_binary.numpy())
                    np.save(os.path.join(config["save_path"], config["dataset"] + str(bit) + "-"  + "trn_label.npy"),
                            trn_label.numpy())
                    np.save(os.path.join(config["save_path"], config["dataset"] + str(bit) + "-"  + "tst_label.npy"),
                            tst_label.numpy())
                    np.save(os.path.join(config["save_path"], config["dataset"] + str(bit) +"-" + "map.npy"),
                            mAP)

            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (config["info"], epoch + 1, bit, config["dataset"], mAP, Best_mAP))
            print(config)

if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)