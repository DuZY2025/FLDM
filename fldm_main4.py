"""
Created on Thu Oct 30 22:23:02 2025
"""

import argparse
import os
import numpy as np
from torchvision.transforms import transforms
from torchvision import datasets
import datetime
import functools
from dataset import Indices2Dataset
from scipy import linalg
from dataset import build_non_iid_by_dirichlet, partition_balance
import torch
from model.model_init import model_init
from torch.utils.data import TensorDataset, DataLoader
from model.ddpm_conditional import UNet_conditional, ConditionalDiffusion1D
from copy import deepcopy
import torch.optim as optim
import random
from opacus import PrivacyEngine
# import dataset_utils
import ujson
import sys
# import json
import logging
from config_parse import config_setup
from fine_tune_tr import fine_tune_fc
    
    
parser = argparse.ArgumentParser()
path_dir = os.path.dirname(__file__)
parser.add_argument('--path_cifar10', type=str, default=os.path.join(path_dir, 'data\\CIFAR10\\'))
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--dataset', default="CIFAR10", type=str, help='dataset name it may be CIFAR10, CIFAR100 or ImageNet')
parser.add_argument('--datapath', type=str, default="./data/CIFAR10_LT001")

parser.add_argument('--partition', type=str, default='dir')
parser.add_argument('--non_iid_alpha', type=float, default=0.1)

parser.add_argument('--num_clients', type=int, default=100)
parser.add_argument('--num_online_clients', type=int, default=30)

parser.add_argument('--num_rounds', type=int, default=600)
parser.add_argument('--num_epochs_local_training', type=int, default=5)  
parser.add_argument('--num_epochs_global_training', type=int, default=30)
parser.add_argument('--warmup_epochs', type=int, default=200)

parser.add_argument('--glb_lr', type=int, default=5e-5)

parser.add_argument('--fl_feature_ratio', type=float, default=0.1)
parser.add_argument('--auxiliary_dataset_size', type=int, default=12000)


parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=7)
# parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')


parser.add_argument('--checkpoint', default=None, type=str, help='model path to resume previous training, default None')
parser.add_argument('--config', default="./config/cifar10/cifar10_LSC_Mixup.txt", help='path to config file')
parser.add_argument('--batch_size_fc', default=1024, type=int, help='CNN fully connected layer batch size')
parser.add_argument('--learning_rate_fc', default=0.001, type=float, help='CNN fully connected layer learning rate')
parser.add_argument('--eval', default=None, type=str, help='evaluate the model performance')
parser.add_argument('--epoch', default=401, type=int, help='epoch number to train')
parser.add_argument('--model_fixed', default="./pretrained_models/resnet32_cifar10_lt001.checkpoint", type=str, help='the encoder model path')




args = parser.parse_args()


print(
    'num_clients:{nc}\n'
    'non_iid_alpha:{non_iid}\n'
    'warmup_epochs:{wue}\n'
    'num_rounds{nr}\n'
    'device:{dv}\n'
    .format(
        nc=args.num_clients,
        non_iid=args.non_iid_alpha,
        wue=args.warmup_epochs,
        nr=args.num_rounds,
        dv=args.device
        ))



# 定义保存路径
save_dir = os.path.join(args.path_cifar10, f"processed_{args.partition}_{args.num_clients}clients")
train_path = os.path.join(save_dir, "train/")
test_path = os.path.join(save_dir, "test/") # 如果你有划分测试集的话
config_path = os.path.join(save_dir, "config.json")
# 创建文件夹
if not os.path.exists(train_path): os.makedirs(train_path)


log_save_dir = os.path.dirname(os.path.abspath(__file__))
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join(log_save_dir, "CIFAR10" + "_client" + str(args.num_clients) + f"_{current_time}.log")
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




# 加载数据集
# 定义图像处理操作
transform_all = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# 使用datasets库函数加载数据集
data_training = datasets.CIFAR10(args.path_cifar10, train=True, download=False, transform=transform_all)

# 按类别保存索引
index_per_class = [[] for _ in range(args.num_classes)]
for idx, datum in enumerate(data_training):
    index_per_class[datum[1]].append(idx)
    

# 计算每个类别需要保留的样本数
img_max = len(data_training) / args.num_classes
img_num_per_class = []    
for _classes_idx in range(args.num_classes):
    num = img_max * (args.imb_factor**(_classes_idx / (args.num_classes - 1.0)))
    img_num_per_class.append(int(num))
print('img_num_class')
print(img_num_per_class)
print('All img num')
print(sum(img_num_per_class))

mul_per_class = []
for class_idx in range(args.num_classes):
    if class_idx == 0:
        mul_per_class.append(1)
    else:
        mul_per_class.append(img_num_per_class[0]/img_num_per_class[class_idx])
print('mul_per_class')
print(mul_per_class)
print("\n")


list_label_indices = []
classes = list(range(args.num_classes))
# 按照样本数随机取样
for _class, _img_num in zip(classes, img_num_per_class):
    indices = index_per_class[_class]
    np.random.shuffle(indices)
    idx = indices[:_img_num]
    list_label_indices.append(idx)

indices2targets = []
for label, index in enumerate(list_label_indices):
    for idx in index:
        # 每个样本与标签成对保存
        indices2targets.append((idx, label))
batch_indices = build_non_iid_by_dirichlet(seed=args.seed,
                                           indices2targets=indices2targets,
                                           non_iid_alpha=args.non_iid_alpha,
                                           num_classes=args.num_classes,
                                           num_indices=len(indices2targets),
                                           n_workers=args.num_clients)
# 展成一维列表
# indices_dirichlet = functools.reduce(lambda x, y: x + y, batch_indices)
# 平衡分配
# list_client_indices = partition_balance(indices_dirichlet, args.num_clients)
list_client_indices = batch_indices


# 展示客户端数据
dict_per_client = []
for client, indices in enumerate(batch_indices):
    nums_data = [0 for _ in range(args.num_classes)]
    for idx in indices:
        label = data_training[idx][1]
        nums_data[label] += 1
    dict_per_client.append(nums_data)
    print(f'{client}: {nums_data}')
# 由于伪随机所以各个客户端每类数量每次都是一样的
# 但是其中每类的数据是不一样的


# 将训练好的模型去掉全连接层，得到特征编码器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
premodel_path = './pretrained_models/resnet32_cifar10_lt001.checkpoint'
config = dict(torch.load(premodel_path))
if 'model' in config['state_dict'].keys():
    model_state_dict = config['state_dict']['model']
model = model_init(config['model'],
                       config['dataset']['name'],
                       model_state_dict)
model.to(device)
modules = list(model.children())[:-1]
modules.append(torch.nn.AvgPool2d(kernel_size=8))
model = torch.nn.Sequential(*modules)
model.to(device)

# 每个客户端加载数据
# 将各个客户端原始图像通过特征编码器得到图像特征
m = len(list_client_indices)
images = [[0 for _ in range(len(list_client_indices[i]))] for i in range(m)]
labels = [[0 for _ in range(len(list_client_indices[i]))] for i in range(m)]
feature_set = [[] for i in range(m)]
label_set = [[] for i in range(m)]
features_tensor = [0 for _ in range(m)]
labels_tensor = [0 for _ in range(m)]
new_dataset = [0 for _ in range(m)]
dataloader_phase1 = [0 for _ in range(m)]
dataloader_phase2 = [0 for _ in range(m)]
indices2data = Indices2Dataset(data_training)
for client in range(args.num_clients):
    # trianset
    indices2data.load(list_client_indices[client])
    data_client = indices2data
    for i in range(len(data_client.indices)):
         image, label = data_client.__getitem__(i)
         images[client][i] = image
         labels[client][i] = label
    images_tsr = torch.stack(images[client], dim=0)
    labels_tsr = torch.tensor(labels[client])
    dataset = TensorDataset(images_tsr, labels_tsr)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    for i, (x, y) in enumerate(dataloader):
        img = x.cuda()
        lb = y.cuda()
        features = model(img)
        # Reshape the features to 2D
        features = features.view(features.size(0), -1)
        feature_set[client].append(features.detach().cpu())
        label_set[client].append(lb.detach().cpu())
    
    features_tensor[client] = torch.cat(feature_set[client], dim=0)
    labels_tensor[client] = torch.cat(label_set[client], dim=0)
    client_save_obj = {
        'features': features_tensor[client],
        'labels': labels_tensor[client]
    }
    torch.save(client_save_obj, os.path.join(train_path, f"client_{client}.pt"))
    new_dataset[client] = TensorDataset(features_tensor[client], labels_tensor[client])
    dataloader_phase1[client] = DataLoader(new_dataset[client], batch_size=64)
    dataloader_phase2[client] = DataLoader(new_dataset[client], batch_size=128)
    
# 按每个类别加载数据
# 按类别计算FID需要
images = [[0 for _ in range(len(list_label_indices[i]))] for i in range(args.num_classes)]
labels = [[0 for _ in range(len(list_label_indices[i]))] for i in range(args.num_classes)]
feature_set = [[] for i in range(args.num_classes)]
label_set = [[] for i in range(args.num_classes)]
features_tensor_class = [0 for _ in range(args.num_classes)]
labels_tensor_class = [0 for _ in range(args.num_classes)]
indices2data_class = Indices2Dataset(data_training)
for class_index in range(args.num_classes):
    indices2data_class.load(list_label_indices[class_index])
    for i in range(len(indices2data_class.indices)):
         image, label = indices2data_class.__getitem__(i)
         images[class_index][i] = image
         labels[class_index][i] = label
    images_tsr = torch.stack(images[class_index], dim=0)
    labels_tsr = torch.tensor(labels[class_index])
    dataset = TensorDataset(images_tsr, labels_tsr)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    for i, (x, y) in enumerate(dataloader):
        img = x.cuda()
        lb = y.cuda()
        features = model(img)
        # Reshape the features to 2D
        features = features.view(features.size(0), -1)
        feature_set[class_index].append(features.detach().cpu())
        label_set[class_index].append(lb.detach().cpu())
    
    features_tensor_class[class_index] = torch.cat(feature_set[class_index], dim=0)
    labels_tensor_class[class_index] = torch.cat(label_set[class_index], dim=0)
    class_save_obj = {
        'features': features_tensor_class[class_index],
        'labels': labels_tensor_class[class_index]
    }
    torch.save(class_save_obj, os.path.join(train_path, f"class_{class_index}.pt"))


config1 = {
    'num_clients': args.num_clients,
    'num_classes': args.num_classes,
    'partition': args.partition,
    'non_iid_alpha': getattr(args, 'non_iid_alpha', 'N/A'),
    'dict_per_client': dict_per_client, 
    'img_num_per_class': img_num_per_class
}
with open(config_path, 'w') as f:
    ujson.dump(config1, f, indent=4)
print(f"\n--- 数据集已保存至: {save_dir} ---\n")






def calculate_fid(real_features, fake_features):
    # 计算FID
    # 这里使用Fréchet Inception Distance (FID)的计算方式
    # real_features 和 fake_features 是两个样本的特征集合，计算它们的均值和协方差
    real_mu = np.mean(real_features, axis=0)
    real_cov = np.cov(real_features, rowvar=False)
    fake_mu = np.mean(fake_features, axis=0)
    fake_cov = np.cov(fake_features, rowvar=False)
    
    # 计算Fréchet距离
    diff = real_mu - fake_mu
    # covmean = np.sqrt(real_cov @ fake_cov)
    cov_mean, _ = linalg.sqrtm(real_cov @ fake_cov, disp=False)
    # 检查复数结果 (理论上协方差矩阵乘积应该是非负定的，但浮点误差可能导致微小的复数部分)
    if np.iscomplexobj(cov_mean):
        if not np.allclose(np.imag(cov_mean), 0, atol=1e-8):
            print("警告: 协方差矩阵乘积的平方根存在显著的虚部，取其实部。")
        cov_mean = np.real(cov_mean)
        
    fid = np.dot(diff, diff) + np.trace(real_cov + fake_cov - 2 * cov_mean)
    return fid

from opacus.grad_sample import GradSampleModule
def collect_grad_norms(
    model,
    dataloader,
    device,
    t_range="high",
    s_cutoff=200,
    max_batches=20  # 只统计前 N 个 batch，避免太慢
):
    # 统计 Phase I（无 DP）下的 per-sample gradient L2 norm
    model = GradSampleModule(model)  # 关键
    model.train()

    grad_norms = []

    for batch_idx, (x, y) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        x = x.unsqueeze(1).to(device)
        y = y.to(device)

        model.zero_grad(set_to_none=True)

        loss, t = model(x, y, t_range=t_range, s_cutoff=s_cutoff)
        loss.backward()

        # 计算 per-sample grad norm
        for p in model.parameters():
            if hasattr(p, "grad_sample") and p.grad_sample is not None:
                # p.grad_sample shape: [batch_size, ...]
                gs = p.grad_sample.view(p.grad_sample.size(0), -1)
                norm = torch.norm(gs, dim=1)  # per-sample
                grad_norms.append(norm.detach().cpu())

        # 清理 grad_sample（非常重要）
        for p in model.parameters():
            if hasattr(p, "grad_sample"):
                p.grad_sample = None

    grad_norms = torch.cat(grad_norms)
    return grad_norms.numpy()



# 联邦学习训练扩散模型

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# scaler = torch.cuda.amp.GradScaler()
print('Using device:', device)
torch.manual_seed(123)
random_state = np.random.RandomState(args.seed)
total_clients = list(range(args.num_clients))
model_u = UNet_conditional(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=1,
    num_classes=args.num_classes
)
globalDM = ConditionalDiffusion1D(
    model_u,
    seq_length=64,
    timesteps=1000,
    objective='pred_x0'
)
globalDM = globalDM.to(device)

generated_features_list_0 = []
fake_classes_list_0 = []
generated_features_tensor_list = []
fake_classes_tensor_list = []

best_globalDM_state = deepcopy(globalDM.state_dict())
best_fid = float('inf')
current_fid = float('inf')


T = globalDM.num_timesteps
S = 200
print(f"Using dp-promise cutoff S = {S} (T = {T})")

for i in range(args.num_rounds+1):
    if i==0:
        print("\n=== WarmUp ===")
        online_clients = np.array(range(args.num_clients))
        client_epochs = args.warmup_epochs
        lr1 = 1e-4
        lr2 = 1e-4
    else:
        print(f"=== Round {i} ===")
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        client_epochs = args.num_epochs_local_training
        lr1 = 1e-4
        lr2 = 1e-4
        print(f"num of online clients {len(online_clients)} ")
        print(f"clients :{online_clients} ")
        
    
    local_models = []
    

    for client in online_clients:
        # 全局模型参数赋值给客户端
        print(f"Round {i}, Client {client} training...")
        clientDM = deepcopy(globalDM)
        clientDM = clientDM.to(device)
        # optimizer = optim.SGD(clientDM.parameters(), lr=1e-4)
        opt_phase1 = optim.Adam(clientDM.parameters(), lr=lr1)  
        
        # for epoch in range(client_epochs):
        #     total_loss = 0
        #     for feature, feature_label in new_dataloader[client]:
        #         feature  = feature.unsqueeze(1)
        #         feature = feature.to(device)
        #         feature_label = feature_label.to(device)
                
        #         optimizer.zero_grad()
        #         with torch.cuda.amp.autocast():
        #             loss = clientDM(feature, feature_label)
        #         scaler.scale(loss).backward()
        #         scaler.step(optimizer)
        #         scaler.update()

        #         total_loss += loss.item()
        #     print(f"Round {i}, Client {client}, epoch {epoch}, loss={total_loss:.4f}")
        
        
        
        # local training
        # PHASE I : high-noise timesteps -> no DP
        for epoch in range(client_epochs):
            total_loss_phase1 = 0.0
            clientDM.train()
            for feature, feature_label in dataloader_phase1[client]:
                feature  = feature.unsqueeze(1)
                feature = feature.to(device)
                feature_label = feature_label.to(device)
                
                opt_phase1.zero_grad()
                # forward: 使用 t_range='high'，传入 s_cutoff=S
                loss, t = clientDM(feature, feature_label, t_range='high', s_cutoff=S)
                loss.backward()
                opt_phase1.step()
                

                total_loss_phase1 += loss.item()
            print(f"Round {i}, Client {client}, epoch {epoch}, Phase I loss={total_loss_phase1:.5f}")
        
        # tmp_model = deepcopy(clientDM).to(device)

        # grad_norms = collect_grad_norms(
        #     model=tmp_model,
        #     dataloader=client_dataloader,
        #     device=device,
        #     t_range="high",
        #     s_cutoff=S,
        #     max_batches=10
        # )
        
        
        # # ... 在 client 训练循环内部 ...
        # for epoch in range(client_epochs):
        #     for batch_idx, (feature, feature_label) in enumerate(client_dataloader):
        #         opt_phase1.zero_grad()
        #         feature  = feature.unsqueeze(1)
        #         feature = feature.to(device)
        #         feature_label = feature_label.to(device)
        #         # 正常前向传播
        #         loss,t = clientDM(feature, feature_label, t_range='high', s_cutoff=S)
        #         loss.backward()
        
        #         # --- 开始统计梯度范数 ---
        #         total_norm = 0.0
        #         # 遍历模型所有参数
        #         parameters = [p for p in clientDM.parameters() if p.grad is not None and p.requires_grad]
        #         if len(parameters) == 0:
        #             total_norm = 0.0
        #         else:
        #             # 计算整个模型的梯度 L2 范数 (与 Opacus 裁剪逻辑一致)
        #             device = parameters[0].grad.device
        #             total_norm = torch.norm(
        #                 torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2
        #             ).item()
        
        #         # 打印建议
        #         if batch_idx % 10 == 0: # 没必要每个 batch 都打，隔几个打一次
        #             print(f"DEBUG: Round {i}, Client {client}, Batch {batch_idx}")
        #             print(f" >>> Current Gradient L2 Norm: {total_norm:.6f}")
        
        #         opt_phase1.step()
            
        # PHASE II pass: low-noise timesteps -> DP-SGD
        # 使用 Opacus 来包装 model/optimizer/dataloader 一次（方便在 Phase II 中执行 DP）
        # 注意：make_private 可能会返回包装后的 model 对象（model._module 为原始 module）
        privacy_engine = PrivacyEngine()
        # You can tune these DP hyperparams:
        dp_max_grad_norm = 1e-2
        # dp_noise_multiplier = 7.89
        # dp_noise_multiplier = 6.0
        # sigma; 增大提升隐私但降低 utility
        # noise_multiplicity: 论文里 phase2 里每个样本会被重复采样 K 次来稳定梯度
        # 这里我们把 noise_multiplicity 设为 1 或可配置
        noise_multiplicity = 4
        
        # make_private 会返回包装后的 (model, optimizer, dataloader)
        clientDM_phase2 = deepcopy(clientDM)
        opt_phase2 = optim.Adam(clientDM_phase2.parameters(), lr=lr2)
        
        # clientDM_priv, opt_phase2_priv, dataloader_priv = privacy_engine.make_private(
        #     module=clientDM_phase2,
        #     optimizer=opt_phase2,
        #     data_loader=client_dataloader,
        #     max_grad_norm=dp_max_grad_norm,
        #     noise_multiplier=dp_noise_multiplier,
        #     noise_multiplicity=noise_multiplicity
        # )
        
        clientDM_priv, opt_phase2_priv, dataloader_priv = privacy_engine.make_private_with_epsilon(
            module=clientDM_phase2,
            optimizer=opt_phase2,
            data_loader=dataloader_phase2[client],
            target_epsilon=1.0,  # epsilon (例如 1.0)
            target_delta=1e-5,      # delta (例如 1e-5)
            epochs=client_epochs, # 告诉它本地要跑多少个 epoch
            max_grad_norm=dp_max_grad_norm, # 推荐 1.0
            noise_multiplicity=noise_multiplicity
        )
                
        # Training loop for Phase II with DP
        for epoch in range(client_epochs):
            total_loss_phase2 = 0.0
            clientDM_priv.train()
            for X, y in dataloader_priv:
                X = X.unsqueeze(1).to(device)
                y = y.to(device)
                
                opt_phase2_priv.zero_grad(set_to_none=True)
                # forward: t_range='low' -> sample t in [0, S-1]
                loss_dp, t_dp = clientDM_priv(X, y, t_range='low', s_cutoff=S)
                loss_dp.backward()
                opt_phase2_priv.step()
                
                total_loss_phase2 += float(loss_dp.item())
            print(f"Round {i}, Client {client}, epoch {epoch}, Phase II loss={total_loss_phase2:.5f}")
            
            # 若 make_private 返回的 model 是包装类型，请将它恢复为原始 module 以便后续联邦聚合（取底层模块 state）
            # 许多 make_private 实现会将包装后的模型放在 model._module / model.module 中。
            # 我们在聚合时使用 clientDM 的 state_dict，如果 clientDM_priv._module 存在，则同步原始 module。
            if hasattr(clientDM_priv, "_module"):
                # 将包装后底层 module 的参数赋回 clientDM（原始对象）
                clientDM.load_state_dict(clientDM_priv._module.state_dict())
            elif hasattr(clientDM_priv, "module"):
                clientDM.load_state_dict(clientDM_priv.module.state_dict())
            else:
                # 如果 make_private 已经原地修改 clientDM，也无需动作
                pass
        
        # local end: 把训练好的 clientDM 加入本地模型列表
        local_models.append(deepcopy(clientDM).cpu())
        
        if i==0:
            # warmup生成伪特征
            class_num = []
            for index, num in enumerate(dict_per_client[client]):
                if num<5:
                    class_num.append(0)
                    continue
                class_num.append(round(num * args.fl_feature_ratio * mul_per_class[index]))
            print(f"生成类别数量{class_num}")
            fake_classes = [i for i, count in enumerate(class_num) for _ in range(count)]
            print(f"Client {client}: the number of generated features is {len(fake_classes)}")
            random.shuffle(fake_classes)
            fake_classes_tensor = torch.tensor(fake_classes)
            fake_classes_tensor = fake_classes_tensor.to(device)
            clientDM = clientDM.to(device)
            generated_features = clientDM.sample(fake_classes_tensor)
            generated_features = generated_features.cpu()
            fake_classes_tensor = fake_classes_tensor.cpu()
            print(f"生成的样本特征：{generated_features.shape}")
            
            fo_list = [0 for i in range(args.num_classes)]
            df_list = [0 for i in range(args.num_classes)]
            # ========== 首先计算每个类的 fo 和 df ==========
            for j, num in enumerate(dict_per_client[client]):
                temporary_feature = []
                if num<5:
                    fo_list[j] = 0
                    df_list[j] = float('inf')
                    continue
                for k in range(len(features_tensor[client])):
                    if labels_tensor[client][k] == j:
                        if len(temporary_feature) == 0:
                            temporary_feature = features_tensor[client][k].unsqueeze(0)
                        else:
                            temporary_feature = torch.cat((temporary_feature,features_tensor[client][k].unsqueeze(0)),0)
                            
                # ========== 1. 计算真实样本中心 fo ==========
                fo_list[j] = temporary_feature.mean(dim=0)
                # ========== 2. 计算 df （真实样本内部最大距离） ==========
                # 扩展维度计算 pairwise distance
                f1 = temporary_feature.unsqueeze(1)  # (m, 1, 64)
                f2 = temporary_feature.unsqueeze(0)  # (1, m, 64)
                dist_matrix = torch.norm(f1 - f2, dim=2)  # (m, m)
                df_list[j] = dist_matrix.max().item()
                print(f"类别{j}, 数量:{len(temporary_feature)}, df:{df_list[j]}")
                
            # ========== 3. 计算生成样本与中心 fo 的距离 di ==========
            generated_features = generated_features.squeeze(1)
            di = []
            for j in range(len(generated_features)):
                distance = torch.norm(generated_features[j] - fo_list[fake_classes_tensor[j]])
                di.append(distance.item())
            # ========== 4. 剔除 OOD 样本 ==========
            # 条件：di > 2*df
            mask_not_ood = []
            for j in range(len(generated_features)):
                if (di[j] < 2*df_list[fake_classes_tensor[j]]):
                    mask_not_ood.append(True)
                else:
                    mask_not_ood.append(False)
            filtered_generated = generated_features[mask_not_ood]
            filtered_generated = filtered_generated.unsqueeze(1)
            filtered_classes = fake_classes_tensor[mask_not_ood]
            print(f"剔除OOD之后的样本特征:{filtered_generated.shape}")
            print(f"剔除OOD之后的样本标签:{filtered_classes.shape}")
            
            generated_features_list_0.append(filtered_generated)
            fake_classes_list_0.append(filtered_classes) 
                         
        print("\n")
        torch.cuda.empty_cache()
    
     
    new_state = {}
    # 1. 先聚合可训练参数（weights & bias）
    for name, param in globalDM.named_parameters():
        avg_param = sum(
            local_models[i].state_dict()[name]
            for i in range(len(local_models))
        ) / len(local_models)
        new_state[name] = avg_param
    # 2. buffers 不做平均（保持全局模型原始值）
    for name, buf in globalDM.named_buffers():
        new_state[name] = buf
    # 3. 加载聚合后的参数
    globalDM.load_state_dict(new_state)
    
    print(f"=== Round {i} Done ===\n")      
    
    
    if i==0:
        # 处理生成的数据集
        
        generated_features_tensor_0 = torch.cat(generated_features_list_0, dim=0)
        fake_classes_tensor_0 = torch.cat(fake_classes_list_0, dim=0)
        
        # 将初始生成的数据添加到辅助数据集用以以后更新
        generated_features_tensor_list = generated_features_tensor_0
        fake_classes_tensor_list = fake_classes_tensor_0
        print(f"辅助数据集{generated_features_tensor_list.shape}")
        
        # dataset_0 = TensorDataset(generated_features_tensor_0, fake_classes_tensor_0)
        # dataloader_0 = DataLoader(dataset_0, batch_size=64)
        # optimizer = optim.Adam(globalDM.parameters(), lr=1e-4)
        
        # # globalDM train
        # for epoch in range(args.num_epochs_global_training):
        #     total_loss = 0
        #     for feature, feature_label in dataloader_0:
                
        #         feature = feature.to(device)
        #         feature_label = feature_label.to(device)
        #         optimizer.zero_grad()
        #         # with torch.cuda.amp.autocast():
        #         loss, _ = globalDM(feature, feature_label)
        #         # scaler.scale(loss).backward()
        #         # scaler.step(optimizer)
        #         # scaler.update()
        #         loss.backward()
        #         optimizer.step()

        #         total_loss += loss.item()
            
        #     print(f"global, epoch {epoch}, loss={total_loss:.4f}")
            
        # torch.cuda.empty_cache()
        
    torch.cuda.empty_cache()
    # 每20次跟之前比较
    if i!=0 and i%20==0 :
        print("判断全局模型的性能")
        logging.info("round: {}  ".format(i))
        total_fid = 0
        for c in range(args.num_classes):
            globalDM = globalDM.to(device)
            class_num = []
            for index, num in enumerate(img_num_per_class):
                if index==c:
                    class_num.append(5000)
                else:
                    class_num.append(0)
            fake_classes = [i for i, count in enumerate(class_num) for _ in range(count)]
            print(f"Class {c} 生成样本")
            # random.shuffle(fake_classes)
            fake_classes_tensor = torch.tensor(fake_classes)
            fake_classes_tensor = fake_classes_tensor.to(device)
            
            generated_features4fid = clientDM.sample(fake_classes_tensor)
            generated_features4fid = generated_features4fid.cpu()
            generated_features4fid = generated_features4fid.squeeze(1)
            classfid = calculate_fid(features_tensor_class[c].cpu().numpy(),generated_features4fid.numpy())
        
            print(f"class {c} fid: {classfid}")
            logging.info("class: {}  fid：{} ".format(c, classfid))
            total_fid = total_fid + classfid
            
        # 求平均fid然后判断是否更好
        avg_fid = total_fid / args.num_classes
        current_fid = avg_fid
        print(f"epoch {i} fid: {avg_fid},  best fid:{best_fid}")
        logging.info("epoch {} fid: {},  best fid:{}".format(i, avg_fid, best_fid))
        print("\n")
        if avg_fid < best_fid:
            best_fid = avg_fid
            if i>=50:
                globalDM = globalDM.to(device)
                class_num = []
                class_num = [round(args.fl_feature_ratio*img_num_per_class[0]) for _ in img_num_per_class]
                fake_classes = [i for i, count in enumerate(class_num) for _ in range(count)]
                print("生成新的辅助数据集")
                random.shuffle(fake_classes)
                fake_classes_tensor = torch.tensor(fake_classes)
                fake_classes_tensor = fake_classes_tensor.to(device)
                new_generated_features = clientDM.sample(fake_classes_tensor)
                new_generated_features = new_generated_features.cpu()
                
                generated_features_tensor_list = torch.cat((generated_features_tensor_list, new_generated_features),0)
                fake_classes_tensor_list = torch.cat((fake_classes_tensor_list, fake_classes_tensor.cpu()),0)
                print(f"合并的辅助数据特征{generated_features_tensor_list.shape}")
                print(f"合并的辅助数据标签{fake_classes_tensor_list.shape}")
                
                # 更新辅助数据集，保存当前模型参数
                s = args.auxiliary_dataset_size
                l = len(generated_features_tensor_list)
                if l>s:
                    generated_features_tensor_list = generated_features_tensor_list[l-s:]
                    fake_classes_tensor_list = fake_classes_tensor_list[l-s:]
                    best_globalDM_state = deepcopy(globalDM.state_dict())
                print(f"辅助数据集中特征{generated_features_tensor_list.shape}")
                print(f"辅助数据集中标签{fake_classes_tensor_list.shape}")
        else:
            print("使用辅助数据集来训练")
            
            auxiliary_dataset = TensorDataset(generated_features_tensor_list, fake_classes_tensor_list)
            auxiliary_dataloader = DataLoader(auxiliary_dataset, batch_size=64)
            # 加载上次更好的模型的参数
            globalDM.load_state_dict(best_globalDM_state)
            optimizer = optim.Adam(globalDM.parameters(), lr=args.glb_lr)
            for epoch in range(args.num_epochs_global_training):
                total_loss = 0
                for feature, feature_label in auxiliary_dataloader:
                    feature = feature.to(device)
                    feature_label = feature_label.to(device)
                    optimizer.zero_grad()
                    # with torch.cuda.amp.autocast():
                    loss,_ = globalDM(feature, feature_label)
                    # scaler.scale(loss).backward()
                    # scaler.step(optimizer)
                    # scaler.update()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"global, epoch {epoch}, loss={total_loss:.4f}")
                
        torch.cuda.empty_cache()
    
    #保存模型
    if i%20==0 or i==args.num_rounds:
        filename = 'fldm_CIFAR10_{}_round_{}_fid_{:.4f}.pth'.format(datetime.date.today().strftime('%m%d'), i, current_fid)
        file_path = os.path.join(filename)
        torch.save(globalDM, file_path)
            
                    
            


# 处理测试集
data_test = datasets.CIFAR10(args.path_cifar10, train=False, download=False, transform=transform_all)
test_index_per_class = [[] for _ in range(args.num_classes)]
for idx, datum in enumerate(data_test):
    test_index_per_class[datum[1]].append(idx)
    
test_img_num_per_class = [1000 for _ in range(args.num_classes)]   

test_list_label_indices = []
classes = list(range(args.num_classes))

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


clientDM = globalDM
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

# 训练分类器
for client in range(args.num_clients):
    print(f"client {client}")
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