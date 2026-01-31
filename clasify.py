# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 19:37:00 2026

"""
import argparse
import os
import sys
# sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
from torchvision.transforms import transforms
from torchvision import datasets
import datetime
# import functools
from dataset import Indices2Dataset
# from scipy import linalg
# from dataset import build_non_iid_by_dirichlet, partition_balance
import torch
# from model.model_init import model_init
from torch.utils.data import TensorDataset, DataLoader
# from model.ddpm_conditional import UNet_conditional, ConditionalDiffusion1D
# from copy import deepcopy
# import torch.optim as optim
import random
# from opacus import PrivacyEngine
# import dataset_utils
# import ujson
import json
import logging
from config_parse import config_setup
from fine_tune_tr import fine_tune_fc



parser = argparse.ArgumentParser()
path_dir = os.path.dirname(__file__)
parser.add_argument('--path_cifar10', type=str, default=os.path.join(path_dir, 'data\\CIFAR10\\'))
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_clients', type=int, default=100)
parser.add_argument('--imb_factor', default=0.1, type=float, help='imbalance factor')
parser.add_argument('--dataset', default="CIFAR10", type=str, help='dataset name it may be CIFAR10, CIFAR100 or ImageNet')
parser.add_argument('--datapath', type=str, default="./data/CIFAR10_LT01")
parser.add_argument('--checkpoint', default=None, type=str, help='model path to resume previous training, default None')
parser.add_argument('--config', default="./config/cifar10/cifar10_LSC_Mixup.txt", help='path to config file')
parser.add_argument('--batch_size_fc', default=1024, type=int, help='CNN fully connected layer batch size')
parser.add_argument('--learning_rate_fc', default=0.001, type=float, help='CNN fully connected layer learning rate')
parser.add_argument('--eval', default=None, type=str, help='evaluate the model performance')
parser.add_argument('--epoch', default=301, type=int, help='epoch number to train')
parser.add_argument('--model_fixed', default="./pretrained_models/resnet32_cifar10_lt001.checkpoint", type=str, help='the encoder model path')


# parser.add_argument('--diffusion_epoch', default=201, type=int, help='diffusion epoch to train')
# parser.add_argument('--feature_ratio', default=0.20, type=float, help='The ratio of generating feature')
# parser.add_argument('--diffusion_step', default=1000, type=int, help='The steps of diffusion')
# parser.add_argument('--is_diffusion_pretrained', default = None, help='pre-trained diffusion model path. Training from scratch if None')
# parser.add_argument('--generation_mmf', default=None, type=str, help='CNN fully connected layer batch size')
args = parser.parse_args()

log_save_dir = os.path.dirname(os.path.abspath(__file__))
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join(log_save_dir, "CIFAR10" + "_client" + str(args.num_clients) + f"_{current_time}.log")
# logging.basicConfig(filename=log_filename, level=logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        # logging.StreamHandler(sys.stdout)
    ],
    force=True 
)
logging.info("--client: {}  --imb_factor:{}  ".format(args.num_clients,  args.imb_factor))


transform_all = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


data_test = datasets.CIFAR10(args.path_cifar10, train=False, download=False, transform=transform_all)
test_index_per_class = [[] for _ in range(args.num_classes)]
for idx, datum in enumerate(data_test):
    test_index_per_class[datum[1]].append(idx)
    
test_img_num_per_class = [1000 for _ in range(args.num_classes)]   

test_list_label_indices = []
classes = list(range(args.num_classes))
# 按照样本数随机取样
for _class, _img_num in zip(classes, test_img_num_per_class):
    indices = test_index_per_class[_class]
    np.random.shuffle(indices)
    idx = indices[:_img_num]
    test_list_label_indices.append(idx)



test_indices2targets = []
for label, index in enumerate(test_list_label_indices):
    for idx in index:
        # 每个样本与标签成对保存
        test_indices2targets.append((idx, label))

test_list_indices = []
for i in test_indices2targets:
    test_list_indices.append(i[0])






test_images = [0 for _ in range(len(test_list_indices))]
test_labels = [0 for _ in range(len(test_list_indices))]



test_indices2data = Indices2Dataset(data_test)
# testset
test_indices2data.load(test_list_indices)

for i in range(len(test_indices2data.indices)):
     image, label = test_indices2data.__getitem__(i)
     test_images[i] = image
     test_labels[i] = label
test_images_tsr = torch.stack(test_images, dim=0)
test_labels_tsr = torch.tensor(test_labels)
test_dataset = TensorDataset(test_images_tsr, test_labels_tsr)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle = False)



features_tensor = [0 for _ in range(len(test_list_indices))]
labels_tensor = [0 for _ in range(len(test_list_indices))]
new_dataset = [0 for _ in range(len(test_list_indices))]
for client in range(args.num_clients):
    # 构建文件路径
    file_name = f"client_{client}.pt"
    file_path = os.path.join('F:\\codes\\FLDM\\0122FLDP\\train', file_name)
    
    # 检查文件是否存在，防止报错
    if os.path.exists(file_path):
        # 加载对象
        checkpoint = torch.load(file_path, map_location='cpu') # 建议先加载到CPU
        
        # 提取数据并存入列表
        features_tensor[client] = checkpoint['features']
        labels_tensor[client] = checkpoint['labels']
        
        print(f"Successfully loaded client {client}: {features_tensor[client].shape}")
    else:
        print(f"Warning: File {file_path} not found.")
        
        

with open('F:\\codes\\FLDM\\0122FLDP\\config.json', 'r', encoding='utf-8') as f:
    config_data = json.load(f)
dict_per_client = config_data.get("dict_per_client")
img_num_per_class = config_data.get("img_num_per_class")

clientDM = torch.load("F:\\codes\\FLDM\\0122FLDP\\fldm_CIFAR10_0124_round_600_fid_4.0886.pth",weights_only=False)
# 记得切换到评估模式（如果你是用来做推理/测试）
clientDM.eval()

def Custom_dataset(args):
    if args.dataset == "CIFAR10":
        class_num = 10
    if args.dataset == "CIFAR100":
        class_num = 100

    dataset = {"name" : args.dataset,
               "class_num" : class_num,
               "imb_factor" : args.imb_factor,
                "path" : args.datapath,
                "batch_size": 64,
                "sampler": None,
                "number_worker": 0,
                "pin_memory": True}
    return dataset
dataset_info = Custom_dataset(args)

dset_info = {'class_num': args.num_classes,
             # 'per_class_loc': class_loc_list,
             'per_class_img_num': img_num_per_class}


cfg, finish = config_setup(args.config,
                           args.checkpoint,
                           args.datapath,
                           update=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for client in range(args.num_clients):
    print(f"client {client}")
    # if client>79:
    logging.info("client: {}: ".format(client))
    max_class_num = max(dict_per_client[client])
    class_num = [max_class_num-dict_per_client[client][i] for i in range(args.num_classes)]
    
    print(f"生成类别数量{class_num}")
    fake_classes = [i for i, count in enumerate(class_num) for _ in range(count)]
    print(f"Client {client}: the number of generated features is {len(fake_classes)}")
    
    random.shuffle(fake_classes)
    fake_classes_tensor = torch.tensor(fake_classes)
    fake_classes_tensor = fake_classes_tensor.to(device)
    with torch.no_grad():
        clientDM = clientDM.to(device)
    generated_features = clientDM.sample(fake_classes_tensor)
    generated_features = generated_features.cpu()
    fake_classes_tensor = fake_classes_tensor.cpu()
    print(f"FLDM生成的样本特征：{generated_features.shape}")
    
    new_dataset[client] = TensorDataset(features_tensor[client], labels_tensor[client])
    
    fine_tune_fc(generated_features, fake_classes_tensor, new_dataset[client], test_dataloader, dataset_info, args, dset_info, cfg)
    
print(" ------------Finish--------------")


    
    