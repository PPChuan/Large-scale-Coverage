import os
import random
import shutil

path = '../../../../mnt/e/Cifar_100/train'
savepath = '../../../../mnt/e/Cifar_100/vtest'
class_list = os.listdir(path)[:20]
if not os.path.exists(savepath):
    os.makedirs(savepath)
for i in class_list:
    d_path = os.path.join(path, i)
    d_list = os.listdir(d_path)
    r = random.sample(range(0, len(d_list)), 100)
    for j in r:
        src_path = os.path.join(d_path, d_list[j])
        ds_path = os.path.join(savepath, i)
        if not os.path.exists(ds_path):
            os.makedirs(ds_path)
        dst_path = os.path.join(ds_path, d_list[j])
        shutil.copy2(src_path, dst_path)



