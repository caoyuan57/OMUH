import torch
from scipy.linalg import hadamard
import numpy as np
import random


class ObjContrastiveLoss(torch.nn.Module):  
    def __init__(self, temperature=0.1):  
        super(ObjContrastiveLoss, self).__init__()   
        self.temperature = temperature 
  
    def forward(self, positive, negative): 
           
        loss = -torch.log((torch.exp(positive / self.temperature).sum()) / (torch.exp(positive / self.temperature).sum()+torch.exp(negative / self.temperature).sum())).mean()  
        return loss
    

           
        
class ObjCenteralLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(ObjCenteralLoss, self).__init__()
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco", "mirflickr","voc2012"}
        self.hash_targets = self.get_hash_targets(90, bit).to(config["device"])
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
        self.criterion = torch.nn.BCELoss().to(config["device"])

    def forward(self, u, y,ind, config):
        #u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))
        return center_loss 

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:

            # to get sign no need to use mean, use sum here
            center_sum = y @ self.hash_targets
            
            random_center = self.multi_label_random_center*2-1
            random_center=random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets
    



