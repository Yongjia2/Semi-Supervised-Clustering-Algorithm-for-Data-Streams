import pandas as pd
import os
import numpy as np
import math
import random

OrgData = pd.read_csv('./dataset/dataset_org/letter.txt', sep=',', index_col=None, header=None)

#============================================================
#divide the dataset into blocks

# row = np.size(OrgData, 0)
# kblocks = 5
# data = np.array(OrgData)
# piece_num = math.floor(row/kblocks)

# for i in range(1, kblocks+1):
#     if i < kblocks:
#         piece = data[(i-1)*piece_num: i*piece_num]
#         np.savetxt('./dataset/letter/%d.txt' % i, piece, fmt='%g', delimiter=',')
#     else:
#         piece = data[(i-1)*piece_num:]
#         np.savetxt('./dataset/letter/%d.txt' % i, piece, fmt='%g', delimiter=',')

# ========================================================
# # get the MUst-Link data of each block
# data = pd.read_csv('./dataset/letter/datainput/5.txt', sep=',', index_col=None, header=None)
# data = np.array(data)
# percent_mustlink = 0.01
# num_mustlink = math.floor(np.size(data, 0) * percent_mustlink)
# mustlink = np.zeros((num_mustlink, 2), dtype=int)
# label_mustlink = np.empty_like(mustlink)
#
#
# label = data[:, -1]
# index = np.array([sample for sample in range(1, np.size(data, 0)+1)])  # 创建原始数据下标
# num = 0
#
# for i in range(0, 1000*num_mustlink):
#     if num >= num_mustlink:
#         break
#     index_1 = random.choice(index)
#     index_2 = random.choice(index)
#     if index_1 != index_2:
#         if label[index_1-1] == label[index_2-1]:
#             if num == 0:
#                 mustlink[num, 0] = index_1
#                 mustlink[num, 1] = index_2
#                 label_mustlink[num, 0] = label[index_1-1]
#                 label_mustlink[num, 1] = label[index_2 - 1]
#                 num += 1
#             if num > 0:
#                 flag = True
#                 while flag:
#                     for j in range(0, num):
#                         if index_1 in mustlink[j, :] and index_2 in mustlink[j, :]:
#                             flag = False
#                             break
#                     if flag:
#                         mustlink[num, 0] = index_1
#                         mustlink[num, 1] = index_2
#                         label_mustlink[num, 0] = label[index_1 - 1]
#                         label_mustlink[num, 1] = label[index_2 - 1]
#                         num += 1
#
# np.savetxt('./dataset/letter/MustLink/5.txt', mustlink, fmt='%g', delimiter=',')

#=============================================================
# get the Cannot-Link data of each block
data = pd.read_csv('./dataset/letter/datainput/5.txt', sep=',', index_col=None, header=None)
data = np.array(data)
percent_cannotlink = 0.01
num_cannotlink = math.floor(np.size(data, 0) * percent_cannotlink)
cannotlink = np.zeros((num_cannotlink, 2), dtype=int)
label_cannotlink = np.empty_like(cannotlink)

label1 = data[:, -1]
index1 = np.array([sample for sample in range(1, np.size(data, 0)+1)])  # 创建原始数据下标
num1 = 0

for i in range(0, 1000*num_cannotlink):
    if num1 >= num_cannotlink:
        break
    index1_1 = random.choice(index1)
    index1_2 = random.choice(index1)
    if index1_1 != index1_2:
        if label1[index1_1-1] != label1[index1_2-1]:
            if num1 == 0:
                cannotlink[num1, 0] = index1_1
                cannotlink[num1, 1] = index1_2
                label_cannotlink[num1, 0] = label1[index1_1-1]
                label_cannotlink[num1, 1] = label1[index1_2 - 1]
                num1 += 1
            if num1 > 0:
                flag = True
                while flag:
                    for j in range(0, num1):
                        if index1_1 in cannotlink[j, :] and index1_2 in cannotlink[j, :]:
                            flag = False
                            break
                    if flag:
                        cannotlink[num1, 0] = index1_1
                        cannotlink[num1, 1] = index1_2
                        label_cannotlink[num1, 0] = label1[index1_1 - 1]
                        label_cannotlink[num1, 1] = label1[index1_2 - 1]
                        num1 += 1

np.savetxt('./dataset/letter/CannotLink/5.txt', cannotlink, fmt='%g', delimiter=',')


print('hello')