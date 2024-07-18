import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets


def config_dataset(config):
    if "cifar" in config["dataset"]:
        config["topK"] = -1
        config["n_class"] = 10
    elif config["dataset"] in ["nuswide_21", "nuswide_21_m"]:
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_81_m":
        config["topK"] = 5000
        config["n_class"] = 81
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "imagenet":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "mirflickr":
        config["topK"] = 5000
        config["n_class"] = 38
    elif config["dataset"] == "voc2012":
        config["topK"] = 1000
        config["n_class"] = 20

    config["data_path"] = "/dataset/" + config["dataset"] + "/"
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "/hdd/public/datasets/nuswide/NUS-WIDE/"
    if config["dataset"] in ["nuswide_21_m", "nuswide_81_m"]:
        config["data_path"] = "/dataset/nus_wide_m/"
    if config["dataset"] == "coco":
        config["data_path"] = "/hdd/public/datasets/coco/"
    if config["dataset"] == "voc2012":
        config["data_path"] = "/home/"
    if config["dataset"] == "mirflickr":
        config["data_path"] = "/hdd/public/datasets/flickr/img/mirflickr/"
    
    config["data"] = {
        "train_set": {"list_path": "/home/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "/home/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "/home/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    return config


class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

'''
def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose(#[transforms.Resize(resize_size)]
                              #+ step +
                              [transforms.Resize([416,416]),
                               transforms.ToTensor()
                               #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    #std=[0.229, 0.224, 0.225])
                               ])
'''
def obj_transform(resize_size,data_set):
    return transforms.Compose([transforms.Resize([416,416]),
                               transforms.ToTensor()
                               #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    #std=[0.229, 0.224, 0.225])
                               ])


def get_data(config):

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=obj_transform(config["resize_size"],data_set))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle=True, num_workers=4)

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])


def compute_result(dataloader, net, device):
    new_trans=transforms.Compose([ 
                     transforms.Resize([224,224]),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                               ])
   
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        img=new_trans(img)
        img=img.to(device)
        clses.append(cls)
        _,hashma=net(img)
        bs.append(hashma.data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap
