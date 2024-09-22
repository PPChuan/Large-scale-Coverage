import os
import random
import shutil
import sys

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch import nn
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.nn.functional as F
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from CL.build.datasets.transform import MultiViewTransform, DiffSCMultiViewTransform
from util import *

sys.path.append('../../')
from CL.build.models import *


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    for un in range(len(data)):
        # 去除[],这两行按数据不同，可以选择
        s = str(data[un]).replace('[', '').replace(']', '')
        # 每行末尾追加换行符
        s = s + '\n'
        file.write(s)
    # file.write(data)
    file.close()
    print("保存成功")

class SSD(object):
    def __init__(self, cover_interval=50, division_num=4, step_decover=14, cover_num=14):
        self.epoch = 0
        self.cover_interval = cover_interval
        self.division_num = division_num
        self.step_decover = step_decover
        self.s_cover_num = cover_num

    def __call__(self, image):
        if self.s_cover_num - (int((self.epoch - 1) / self.cover_interval) * self.step_decover) <= 0:
            return image
        else:
            img = np.array(image)
            h, w, c = img.shape
            if h != w:
                raise Exception('Error, RandomCovering From Covering.py, Line 85.')
            step_num = int((self.epoch - 1) / self.cover_interval)
            cover_num = self.s_cover_num - (step_num * self.step_decover)
            numbers = range(0, self.division_num * self.division_num)
            random_number = random.sample(numbers, cover_num)
            scale = w / self.division_num
            mask = torch.ones(h, w, c).type(torch.int16)
            for i in random_number:
                x = i % self.division_num
                y = (i - x) / self.division_num
                mask[int(x * scale):int((x + 1) * scale) if x < self.division_num - 1 else w,
                int(y * scale):int((y + 1) * scale) if y < self.division_num - 1 else w, :] = 0
            img = np.uint8(np.multiply(img, mask.numpy()))
            image = Image.fromarray(img)
            return image

class dataset_ssd(datasets.ImageFolder):
    def __init__(self, root, transform, epoch=0, **kwargs):
        super().__init__(root=root, **kwargs)
        self.transform = transform
        self.epoch = epoch

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        self.transform.epoch = self.epoch
        img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

device = torch.device("cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# //////////////////////////////////////////////////////////////
# md = 'rcrop'
md = 'ssd'

data_dir = '../../../../mnt/e/Cifar_100/vtest'
# data_dir = '../../../datasets/Cifar_100/vtest'

class_list = os.listdir(data_dir)

mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)

if md == 'rcrop':
    data_set = torchvision.datasets.ImageFolder(
        root=data_dir,
        transform=MultiViewTransform(transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]), num_views=2)
    )
elif md == 'ssd':
    data_set = dataset_ssd(
        root=data_dir,
        transform=DiffSCMultiViewTransform(transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            [
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                SSD(cover_interval=50, division_num=4, step_decover=14, cover_num=14),
                # savepic(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
    )
else:
    raise 'error'

dataloader = torch.utils.data.DataLoader(data_set, batch_size=512, num_workers=8, drop_last=True, shuffle=True)
# path = '../simclr_cifar100_rcrop_singlestepdecovering1_2023-10-14_19/pretrain/checkpoint/'
# path = '../moco_cifar100_rcrop_2023-11-22_23/pretrain/checkpoint/'
# path = '../moco_cifar100_rcrop_singlestepdecovering_nt2_2023-11-24_06/pretrain/checkpoint/'
path = '../f1'
w_list = os.listdir(path)
try:
    w_list.remove('last_weights.pth')
except:
    print('no last_weights.pth')
try:
    w_list.remove('last_model.pth')
except:
    print('no last_model.pth')

# model_type = "moco"
model_type = "simclr"

model = ResNet(depth=18, features_dim=128, maxpool=False)
dim_mlp = model.fc.weight.shape[1]
model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
# if model_type == 'simclr':
#     model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
#     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
# elif model_type == 'moco':
#     model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)

model.to(device)
r_list = []
# w_list.append('weights_epoch0.pth')
wl = tqdm(w_list, leave=True)
for wp in wl:
    if wp.split('.')[1] != 'pth':
        continue

    w_path = os.path.join(path, wp)

    param = torch.load(w_path)
    model.load_state_dict(param)
    # load_weights(w_path, model)

    n = 0
    a_p = .0
    p_n = .0
    # res = [torch.tensor([]), torch.tensor([])]
    if md == 'ssd':
        data_set.epoch = int(wp.split('.')[0].replace('weights_epoch', ''))
    for images, labels in tqdm(dataloader, leave=False):

        bsz = images[0].shape[0]

        images = torch.cat([images[0], images[1]], dim=0)
        images.to(device)
        features = model(images)  # (2*bsz, C)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        # 正样本相似度
        a_p = a_p + (torch.mean(F.cosine_similarity(f1, f2, dim=1))).item()
        # 负样本相似度
        a, _ = f1.shape
        # b = 2 * a - 2
        ne_s = 0
        for i in range(a):
            x2 = torch.cat([f1[0:i, :], f1[i + 1:a, :], f2[0:i, :], f2[i + 1:a, :]], dim=0)
            x1 = f2[i, :].repeat(x2.shape[0], 1)

            ne_s = ne_s + (torch.mean(F.cosine_similarity(x1, x2, dim=1))).item()
        p_n = p_n + ne_s / a
        n = n + 1
    r = {
        'epoch': int(wp.split('.')[0].replace('weights_epoch', '')),
        'a_n': a_p / n,
        'p_n': p_n / n
    }
    r_list.append(r)
    print('epoch' + wp.split('.')[0].replace('weights_epoch', '') + ' p_s: ' + str(a_p / n) + ' n_s: ' + str(p_n / n))
r_list = sorted(r_list, key = lambda item:item['epoch'])
print(r_list)
text_save(os.path.join(path, md + '_similarity.txt'), r_list)
