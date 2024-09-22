import os
import random
import shutil
import sys

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from mpl_toolkits import axisartist
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch import nn
from torchvision.transforms import transforms
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from util import *

device = torch.device("cpu")
sys.path.append('../../')
from CL.build.models import *

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

data_dir = '../../../../mnt/e/Cifar_100/vtest'
# data_dir = '../../../datasets/Cifar_100/vtest'

class_list = os.listdir(data_dir)

mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)

data_set = torchvision.datasets.ImageFolder(
    root=data_dir,
    transform=transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
)
dataloader = torch.utils.data.DataLoader(data_set, batch_size=512, num_workers=8, drop_last=False)
# path = '../simclr_cifar100_rcrop_singlestepdecovering1_2023-10-14_19/pretrain/checkpoint/'
# path = '../simclr_cifar100_rcrop_2023-10-12_00/pretrain/checkpoint/'
# path = '../simclr_cifar100_rcrop_2023-10-25_16/pretrain/checkpoint/'

# path = '../simclr_cifar100_rcrop_2023-10-30_18/pretrain/checkpoint/'

# path = '../moco_cifar100_rcrop_2023-11-22_23/pretrain/checkpoint/'
# path = '../moco_cifar100_rcrop_singlestepdecovering_nt2_2023-11-24_06/pretrain/checkpoint/'

path = '../f1'
# moco_cifar100_rcrop_singlestepdecovering_nt1_2023-11-24_03
# moco_cifar100_rcrop_singlestepdecovering_nt2_2023-11-24_06
# path = '../simclr_cifar100_ccrop_singlestepdecovering1_2023-10-15_18/pretrain/checkpoint/'
# path = '../simclr_cifar100_ccrop_2023-10-14_19/pretrain/checkpoint/'
w_list = os.listdir(path)
try:
    w_list.remove('last_weights.pth')
except:
    print('no last_weights.pth')
try:
    w_list.remove('v_pics')
    shutil.rmtree(os.path.join(path, 'v_pics'))
except:
    print('no v_pics')
try:
    w_list.remove('last_model.pth')
except:
    print('no last_model.pth')

savepath = os.path.join(path, 'v_pics')
if not os.path.exists(savepath):
    os.makedirs(savepath)

# *************************************
# parm
m = TSNE(n_components=2, learning_rate=150, random_state=245)
# m = PCA(n_components=2, random_state=245)

# model_type = "moco"
model_type = "simclr"
normal = True
fit_e = 500

# x = 200
# simclr:1 => ccrop,2 => rcrop
# 绘图XY轴
xy_lim = 0
# *************************************
model = ResNet(depth=18, features_dim=128, maxpool=False)
dim_mlp = model.fc.weight.shape[1]
if model_type == 'simclr':
    model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
elif model_type == 'moco':
    model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)

model.to(device)

colors = plt.cm.rainbow(np.linspace(0, 1, len(class_list)))

if fit_e != 0:
    load_weights(os.path.join(path, 'weights_epoch' + str(fit_e) + '.pth'), model)
    # model.load_state_dict(torch.load(os.path.join(path, 'weights_epoch' + str(fit_e) + '.pth')))
    w_list.remove('weights_epoch' + str(fit_e) + '.pth')

res_0 = [torch.tensor([]), torch.tensor([])]
for images, labels in tqdm(dataloader, leave=False):
    images = images.to(device)
    labels = labels
    preds = model(images)

    f1 = res_0.pop(0)
    l1 = res_0.pop(0)

    res_0.append(torch.cat([f1, preds.cpu()], 0))
    res_0.append(torch.cat([l1, labels], 0))

m.fit(res_0[0].detach().numpy())

# print(colors)
tsne_0 = m.fit_transform(res_0[0].detach().numpy())
if normal:
    tsne_0 = (tsne_0 - tsne_0.min(0)) / (tsne_0.max(0) - tsne_0.min(0))

fig = plt.figure(figsize=(12, 6))
ax = axisartist.Subplot(fig, 1, 1, 1)
fig.add_axes(ax)
ax.axis["bottom"].set_axisline_style("->", size = 1.5)
ax.axis["left"].set_axisline_style("->", size=1.5)
ax.axis["top"].set_visible(False)
ax.axis["right"].set_visible(False)

if xy_lim == 1:
    plt.xlim(-1, 6)
    plt.ylim(-0.6, 1)

cl = [colors[int(res_0[1][x])] for x in range(len(res_0[1]))]

plt.scatter(tsne_0[:, 0], tsne_0[:, 1], color=cl)
plt.savefig(os.path.join(savepath, 'weights_epoch' + str(fit_e) + '.png'),dpi=300, bbox_inches='tight')
plt.close()

# w_list.append('weights_epoch0.pth')
wl = tqdm(w_list, leave=True)
for wp in wl:

    if wp.split('.')[1] != 'pth':
        continue

    w_path = os.path.join(path, wp)

    load_weights(w_path, model)

    res = [torch.tensor([]), torch.tensor([])]

    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)
        labels = labels
        preds = model(images)

        f1 = res.pop(0)
        l1 = res.pop(0)
        res.append(torch.cat([f1, preds.cpu()], 0))
        res.append(torch.cat([l1, labels], 0))

    tsne = m.fit_transform(res[0].detach().numpy())
    if normal:
        tsne = (tsne - tsne.min(0)) / (tsne.max(0) - tsne.min(0))

    # plt.figure(figsize=(12, 6))
    if xy_lim == 1:
        if wp == 'weights_epoch50.pth':
            plt.xlim(-1 * 160, 160)
            plt.ylim(-1 * 160, 160)
        elif wp == 'weights_epoch100.pth':
            plt.xlim(-1 * 50, 70)
            plt.ylim(-1 * 50, 50)
        elif wp == 'weights_epoch150.pth':
            plt.xlim(-1 * 20, 40)
            plt.ylim(-1 * 20, 40)
        elif wp == 'weights_epoch200.pth':
            plt.xlim(-1 * 8, 11)
            plt.ylim(-1 * 8, 11)
    elif xy_lim == 2:
        if wp == 'weights_epoch50.pth':
            plt.xlim(-1 * 200, 350)
            plt.ylim(-1 * 250, 250)
        elif wp == 'weights_epoch100.pth':
            plt.xlim(-1 * 60, 60)
            plt.ylim(-1 * 50, 100)
        elif wp == 'weights_epoch150.pth':
            plt.xlim(-1 * 60, 60)
            plt.ylim(-1 * 60, 60)
        elif wp == 'weights_epoch200.pth':
            plt.xlim(-1 * 30, 50)
            plt.ylim(-1 * 30, 30)
    cl = [colors[int(res_0[1][x])] for x in range(len(res_0[1]))]


    # ax = plt.gca()
    # ax.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.spines['right'].set_color('none')
    # , fc = 'blue', ec = 'blue'
    # ax.arrow(0, 0, 0, 0, head_width=0.01, head_length=0.01, fc = 'blue', ec = 'blue')
    # ax.arrow(0, 0, 0, 0, head_width=0.01, head_length=0.01, fc = 'blue', ec = 'blue')



    fig = plt.figure(figsize=(12, 6))
    ax = axisartist.Subplot(fig, 1, 1, 1)
    fig.add_axes(ax)
    ax.axis["bottom"].set_axisline_style("->", size = 1.5)
    ax.axis["left"].set_axisline_style("->", size=1.5)
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)

    plt.scatter(tsne[:, 0], tsne[:, 1], c=cl)
    # plt.axis('off')
    # , pad_inches = 0.0
    plt.savefig(os.path.join(savepath, wp.split('.')[0] + '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()


