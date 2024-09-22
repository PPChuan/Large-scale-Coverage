import os
import sys

import matplotlib.pyplot as plt
from mpl_toolkits import axisartist

sys.path.append('../../')

path = '../f1'

# x = [10 ,20 ,30 ,40, 50, 60, 70, 80, 90, 100]

file_ssd = open(os.path.join(path, 'ssd_similarity.txt'), 'r')
file_rcrop = open(os.path.join(path, 'rcrop_similarity.txt'), 'r')

ssd_p_list = []
ssd_n_list = []
for l in file_ssd.readlines():
    p_s = float(l.split(',')[1][8:])
    n_s = float(l.split(',')[2][8:len(l.split(',')[2])-2])

    # print(l)

    ssd_p_list.append(p_s)
    ssd_n_list.append(n_s)

    # print(p_s)
    # print(n_s)

file_ssd.close()

# print(ssd_p_list)
# print(ssd_n_list)

rcrop_p_list = []
rcrop_n_list = []
for l in file_rcrop.readlines():
    p_s = float(l.split(',')[1][8:])
    n_s = float(l.split(',')[2][8:len(l.split(',')[2]) - 2])

    # print(l)

    rcrop_p_list.append(p_s)
    rcrop_n_list.append(n_s)

    # print(p_s)
    # print(n_s)

file_ssd.close()

# print(rcrop_p_list)
# print(rcrop_n_list)

x = []
di = int(100/len(rcrop_n_list))
for i in range(len(rcrop_n_list)):
    x.append((i+1)*di)

plt.figure(figsize=(12, 6))
plt.xlabel('Training epochs', fontsize=18, labelpad=15)
plt.xticks(x[::1],x[::1], fontsize=14)
plt.ylabel('Similarity(a_p)', fontsize=18, labelpad=15)
plt.yticks(fontsize=14)
plt.plot(x, rcrop_p_list, marker='o', linewidth=2, label='Backbone')
plt.plot(x, ssd_p_list, marker='o', linewidth=2, label='Large-scale Coverage')
plt.legend(loc='best', fontsize=18)
plt.grid(linestyle='--', alpha=0.8, linewidth=1)
plt.tight_layout()

plt.savefig(os.path.join(path, 'ap_s.png'),dpi=300)
# plt.show()

plt.close()


plt.figure(figsize=(12, 6))
plt.xlabel('Training epochs', fontsize=18, labelpad=15)
plt.xticks(x[::1],x[::1], fontsize=14)
plt.ylabel('Similarity(p_n)', fontsize=18, labelpad=15)
plt.yticks(fontsize=14)
plt.plot(x, rcrop_n_list, marker='o', linewidth=2, label='Backbone')
plt.plot(x, ssd_n_list, marker='o', linewidth=2, label='Large-scale Coverage')
plt.legend(loc='best', fontsize=18)
plt.grid(linestyle='--', alpha=0.8, linewidth=1)
plt.tight_layout()

plt.savefig(os.path.join(path, 'pn_s.png'),dpi=300)
# plt.show()

plt.close()


