
import pandas as pd
import os
import numpy as np
import time
from dgm import dgm

from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, normalized_mutual_info_score
from collections import Counter
from sklearn import datasets
# import jqmcvi
# from jqmcvi import base

"""
time: 2023-06-05
author: Yongjia Yuan
note: this is semi-supervised cluster algorithm on a example data set
"""



class IncSemisupervisedClusteringAlgo():



    def __init__(self, num_clusters=10):
        self.dataset_name = 'letter'
        self.num_clusters = num_clusters
        self.num_clusters_max = 100
        self.tnorm = 0
        self.tlimit = 7.2 * 10 ** 4
        self.label = 'Ture'

    def para_choose(self):
        if self.nrecord <= 200:
            self.gamma1 = 1.0
            self.gamma2 = 1.0
            self.gamma3 = 1.0
        elif (self.nrecord > 200) and (self.nrecord <= 1000):
            self.gamma1 = 8.0e-1
            self.gamma2 = 5.0e-1
            self.gamma3 = 4.0e-1
        elif (self.nrecord > 1000) and (self.nrecord <= 5000):
            self.gamma1 = 7.0e-1
            self.gamma2 = 4.0e-1
            self.gamma3 = 3.0e-1
        elif (self.nrecord > 5000) and (self.nrecord <= 15000):
            self.gamma1 = 5.0e-1
            self.gamma2 = 3.0e-1
            self.gamma3 = 2.0e-1
        elif (self.nrecord > 15000) and (self.nrecord <= 50000):
            self.gamma1 = 4.0e-1
            self.gamma2 = 2.0e-1
            self.gamma3 = 1.0e-1
        elif self.nrecord > 50000:
            self.gamma1 = 3.0e-1
            self.gamma2 = 1.0e-1
            self.gamma3 = 1.0e-1

        return self.gamma1, self.gamma2, self.gamma3

    def purity(self, result, label):
        total_num = len(label)
        cluster_counter = Counter(result)
        original_counter = Counter(label)

        t = []
        for k in cluster_counter:
            p_k = []
            for j in original_counter:
                count = 0
                for i in range(len(result)):
                    if result[i] == k and label[i] == j:  # 求交集
                        count += 1
                p_k.append(count)
            temp_t = max(p_k)
            t.append(temp_t)

        pur = sum(t) / total_num

        return pur

    def input(self):
        if self.dataset_name == 'letter':
            Org_data = pd.read_csv('./dataset/letter/datainput/1.txt', sep=',', index_col=None, header=None)
            self.data = np.array(Org_data.iloc[:, :-1])
            if self.label:
                self.Classlabel = np.array(Org_data.iloc[:, -1])
            CL_index = pd.read_csv('./dataset/letter/CannotLink/1.txt', sep=',', index_col=None, header=None)
            self.CL_index = np.array(CL_index)
            self.ncan = self.CL_index.shape[0]
            ML_index = pd.read_csv('./dataset/letter/MustLink/1.txt', sep=',', index_col=None, header=None)
            self.ML_index = np.array(ML_index)
            self.nmust = self.ML_index.shape[0]
        if self.dataset_name == 'example':
            Org_data = pd.read_csv('./dataset/example/datainput.txt', sep=',', index_col=None, header=None)
            self.data = np.array(Org_data.iloc[:, :-1])
            if self.label:
                self.Classlabel = np.array(Org_data.iloc[:, -1])
            CL_index = pd.read_csv('./dataset/example/cannotlink.txt', sep=' ', index_col=None, header=None)
            self.CL_index = np.array(CL_index)
            self.ncan = self.CL_index.shape[0]
            ML_index = pd.read_csv('./dataset/example/mustlink.txt', sep=' ', index_col=None, header=None)
            self.ML_index = np.array(ML_index)
            self.nmust = self.ML_index.shape[0]

    def step1(self, w):
        x = np.dot(self.data.T, w / np.sum(w))
        distance = np.linalg.norm(self.data - x.reshape((1, x.size)), axis=1) ** 2
        f = np.dot(distance, w)
        self.tnorm = self.tnorm + self.nrecord
        x = x.T
        return f, x

    def _step2(self, toler, lcand, min_dis_sample, w):
        fmin1 = np.zeros((np.size(lcand, 0), 1), dtype=float)
        x2 = np.zeros((0, self.num_feature))

        if self.nrecord < 200:
            x2 = self.data[lcand, :]

        for i, sample in enumerate(lcand):
            distance_3 = np.array([np.linalg.norm(self.data[sample] - tmp) ** 2 for tmp in self.data])
            differ_1 = distance_3 - min_dis_sample
            index = np.where(differ_1 < 0)
            fmin1[i] = np.dot(differ_1[index], w[index])

        fmin1 = fmin1.flatten()
        i_min, fmin = fmin1.argmin(), fmin1.min()
        i_max, fmax = fmin1.argmax(), fmin1.max()
        #tnorm = tnorm + np.size(lcand, 0) * self.nrecord

        # with open('./results/test.txt', 'a') as file:
        #  file.write(f'\t step2\t i_min\t fmin\t i_max\t fmax\t {i_min}\t {fmin}\t {i_max}\t {fmax}\t\n')

        fmin2 = fmin + self.gamma1 * (fmax-fmin)
        index1 = np.where(fmin1 <= fmin2)
        lcand = np.array(lcand)
        lcand = lcand[index1[0]]

        nstart = 0

        for i, sample in enumerate(lcand):
            distance = [np.linalg.norm(self.data[sample] - tmp) ** 2 for tmp in self.data]
            index2 = np.where(distance < min_dis_sample)
            if len(index2[0]) == 0:
                continue
            else:
                w1 = w[index2].sum()
                x4 = np.dot(self.data[index2].T, w[index2] / w1).T
                if nstart == 0:
                    nstart += 1
                    x2 = np.vstack((x2, x4))
                    continue
                else:
                    #tnorm += nstart
                    distance_1 = [np.linalg.norm(x4 - tmp) ** 2 for tmp in x2]
                    if (np.array(distance_1) <= toler).any():
                        continue
                    else:
                        nstart += 1
                        x2 = np.vstack((x2, x4))


        #tnorm = tnorm + nstart * self.nrecord
        distance_nstart = np.array([np.linalg.norm(tmp1 - tmp2) ** 2 for tmp1 in x2 for tmp2 in self.data]).reshape(
            nstart, self.nrecord)
        differ_2 = distance_nstart - min_dis_sample
        new_distance_start = np.minimum(differ_2, 0)
        decrease_d21 = np.dot(new_distance_start, w).flatten()
        max_dec = decrease_d21.min()
        d6 = decrease_d21.max()

        # with open('./results/test.txt', 'a') as file:
        #  file.write(f'\t step2\t max_dec\t d6\t toler\t {max_dec }\t {d6}\t {toler}\t\n')

        d2 = max_dec + self.gamma2 * (d6-max_dec)
        index_d2 = np.where(decrease_d21 <= d2)
        nstart = len(index_d2[0])
        x2 = x2[index_d2]

        # with open('./results/results_test.txt', 'a') as file:
        #    file.write(f'\tstep2_x2\t {x2}\t\n')

        return nstart, x2 #tnorm


    def cluster(self,x,nc,f):
        # with open('./results/results_test.txt', 'a') as file:
        #    file.write(f'\t cluster_input_x\t {x}\t\n')
        nel = np.zeros((nc, ), dtype=int)
        cluster_radius = np.zeros((nc, ), dtype=float)  # 计算每个簇的半径
        cluster_nk_OrgIndex = {}
        rad_sum = np.zeros((nc,))
        radmax = np.zeros((nc,))
        radmax = np.zeros((nc, 1), dtype=float)
        list1 = np.zeros((self.nrecord,), dtype=int)
        min_dis_sample = np.zeros((self.nrecord, ))

        # with open('./results/results_test.txt', 'a') as file:
        #     file.write(f'\t cluster\t {x}\t\n')

        # cluster normal
        list_normalIndex = []
        for i in range(1, self.nrecord + 1):
            if (i not in self.CL_index) and (i not in self.ML_index):
                list_normalIndex.append(i)
        list_normalIndex1 = np.array(list_normalIndex) - 1   # txt 文档的下标从1开始，python从0开始

        # list_nrecord = np.array([sample for sample in range(0, self.nrecord)]) # 创建原始数据下标
        # list_pair = np.r_[self.CL_index[:,0], self.CL_index[:,1], self.ML_index[:,0], self.ML_index[:,0]] - 1  #约束数据点所在下标
        # print(len(set(list_pair)))
        # list_normalIndex = [sample for sample in list_nrecord if sample not in list_pair]   #正常数据点所在下标
        # nbar = np.size(list_normalIndex)
        #list_normalIndex1 = np.array(list_normalIndex)  # txt 文档的下标从1开始，python从0开始

        normal_data = self.data[list_normalIndex1]
        cluster_plan_index, cluster_plan_nk, min_dis_sample_1, cluster_plan = self._assign_data(normal_data, x, nc)
        list1[list_normalIndex1] = cluster_plan_index  #向量：每个数据点所属的cluster
        min_dis_sample[list_normalIndex1] = min_dis_sample_1  # 将normal数据的中心距离输入到min_dis_sample中

        for i in range(0, nc):  # Fortran算法
            rad_sum[i] = min_dis_sample_1[cluster_plan_nk[i]].sum()
            radmax[i] = max(min_dis_sample_1[cluster_plan_nk[i]])

        if nc == 5:
            print('stop')


        # cluster cl-links
        for i in range(0, self.ncan):
            f_sum = 1.0e+22
            sample1, sample2 = self.data[self.CL_index[i][0] - 1, :], self.data[self.CL_index[i][1] - 1, :]
            for j in range(0, nc):
                for jj in range(0, nc):
                    if j != jj:
                        distance_cl_1 = np.linalg.norm(sample1 - x[j, :]) ** 2
                        distance_cl_2 = np.linalg.norm(sample2 - x[jj, :]) ** 2
                        distance_sum = distance_cl_1 + distance_cl_2
                        if f_sum > distance_sum:
                            f_sum = distance_sum
                            list1[self.CL_index[i][0] - 1] = j      # 把该点所属的cluster加到list1中
                            list1[self.CL_index[i][1] - 1] = jj
                            min_dis_sample[self.CL_index[i][0] - 1] = distance_cl_1  # 该点到该簇中心的距离
                            min_dis_sample[self.CL_index[i][1] - 1] = distance_cl_2
            rad_sum[list1[self.CL_index[i][0] - 1]] += min_dis_sample[self.CL_index[i][0] - 1] #Fortran算法
            rad_sum[list1[self.CL_index[i][1] - 1]] += min_dis_sample[self.CL_index[i][1] - 1]
            if min_dis_sample[self.CL_index[i][0] - 1] > radmax[list1[self.CL_index[i][0] - 1]]:  # 该簇的最大距离
                radmax[list1[self.CL_index[i][0] - 1]] = min_dis_sample[self.CL_index[i][0] - 1]
            if min_dis_sample[self.CL_index[i][1] - 1] > radmax[list1[self.CL_index[i][1] - 1]]:
                radmax[list1[self.CL_index[i][1] - 1]] = min_dis_sample[self.CL_index[i][1] - 1]

            # cluster_radius_sum[list1[self.CL_index[i][0] - 1]] += min_dis_sample[self.CL_index[i][0] - 1]  # 该簇到中心点的距离和
            # cluster_radius_sum[list1[self.CL_index[i][1] - 1]] += min_dis_sample[self.CL_index[i][1] - 1]
            # cluster_nk_OrgIndex[list1[self.CL_index[i][0] - 1]] = np.hstack(
            #     (cluster_nk_OrgIndex[list1[self.CL_index[i][0] - 1]], (self.CL_index[i][0] - 1)))  #把该点加到所属簇nk中
            # cluster_nk_OrgIndex[list1[self.CL_index[i][1] - 1]] = np.hstack(
            #     (cluster_nk_OrgIndex[list1[self.CL_index[i][1] - 1]], (self.CL_index[i][1] - 1)))
            # if min_dis_sample[self.CL_index[i][0] - 1] > radmax[list1[self.CL_index[i][0] - 1]]:  # 该簇的最大距离
            #     radmax[list1[self.CL_index[i][0] - 1]] = min_dis_sample[self.CL_index[i][0] - 1]
            # if min_dis_sample[self.CL_index[i][1] - 1] > radmax[list1[self.CL_index[i][1] - 1]]:  # 该簇的最大距离
            #     radmax[list1[self.CL_index[i][1] - 1]] = min_dis_sample[self.CL_index[i][1] - 1]

        # cluster Ml-links
        for i in range(0, self.nmust):
            f_sum = 1.0e+22
            sample1, sample2 = self.data[self.ML_index[i][0] - 1, :], self.data[self.ML_index[i][1] - 1, :]
            for j in range(0, nc):
                distance_cl_1 = np.linalg.norm(sample1 - x[j, :]) ** 2
                distance_cl_2 = np.linalg.norm(sample2 - x[j, :]) ** 2
                distance_sum = distance_cl_1 + distance_cl_2
                if f_sum > distance_sum:
                    f_sum = distance_sum
                    list1[self.ML_index[i][0] - 1] = j  # 把该点所属的cluster加到list1中
                    list1[self.ML_index[i][1] - 1] = j
                    min_dis_sample[self.ML_index[i][0] - 1] = distance_cl_1  # 该点到该簇中心的距离
                    min_dis_sample[self.ML_index[i][1] - 1] = distance_cl_2
            rad_sum[list1[self.ML_index[i][0] - 1]] += min_dis_sample[self.ML_index[i][0] - 1]  #Fortran算法
            rad_sum[list1[self.ML_index[i][1] - 1]] += min_dis_sample[self.ML_index[i][1] - 1]
            if min_dis_sample[self.ML_index[i][0] - 1] > radmax[list1[self.ML_index[i][0] - 1]]:  # 该簇的最大距离
                radmax[list1[self.ML_index[i][0] - 1]] = min_dis_sample[self.ML_index[i][0] - 1]
            if min_dis_sample[self.ML_index[i][1] - 1] > radmax[list1[self.ML_index[i][1] - 1]]:
                radmax[list1[self.ML_index[i][1] - 1]] = min_dis_sample[self.ML_index[i][1] - 1]

            # cluster_radius_sum[list1[self.ML_index[i][0] - 1]] += min_dis_sample[self.ML_index[i][0] - 1]  # 该簇到中心点的距离和
            # cluster_radius_sum[list1[self.ML_index[i][1] - 1]] += min_dis_sample[self.ML_index[i][1] - 1]
            # cluster_nk_OrgIndex[list1[self.ML_index[i][0] - 1]] = np.hstack(
            #     (cluster_nk_OrgIndex[list1[self.ML_index[i][0] - 1]], (self.ML_index[i][0] - 1)))  # 把该点加到所属簇nk中
            # cluster_nk_OrgIndex[list1[self.ML_index[i][1] - 1]] = np.hstack(
            #     (cluster_nk_OrgIndex[list1[self.ML_index[i][1] - 1]], (self.ML_index[i][1] - 1)))
            # if min_dis_sample[self.ML_index[i][0] - 1] > radmax[list1[self.ML_index[i][0] - 1]]:  # 该簇的最大距离
            #     radmax[list1[self.ML_index[i][0] - 1]] = min_dis_sample[self.ML_index[i][0] - 1]
            # if min_dis_sample[self.ML_index[i][1] - 1] > radmax[list1[self.ML_index[i][1] - 1]]:
            #     radmax[list1[self.ML_index[i][1] - 1]] = min_dis_sample[self.ML_index[i][1] - 1]


        # link_points = []
        # # cluster CL-links
        # if nc > 1:
        #     distance_cl = np.zeros((nc, nc))
        #     m1 = 9e26
        #     for i in range(0, self.ncan):
        #         sample1, sample2 = self.data[self.CL_index[i][0] - 1, :], self.data[self.CL_index[i][1] - 1, :]
        #         distance_cl_1 = np.array([np.linalg.norm(sample1 - tmp) ** 2 for tmp in x])
        #         distance_cl_2 = np.array([np.linalg.norm(sample2 - tmp) ** 2 for tmp in x])
        #         for k in range(0, nc):
        #             for j in range(0, nc):
        #                 if k == j:
        #                     distance_cl[k][j] = m1
        #                 else:
        #                     distance_cl[k][j] = distance_cl_1[k] + distance_cl_2[j]
        #         min_value = np.min(distance_cl)  # 找到最小值
        #         index = np.where(distance_cl == min_value)  # 返回最小值下标，形式为tuple
        #         index_1, index_2 = index[0][0], index[1][0]  # 提取两个点对应所在的簇
        #         if (self.CL_index[i][0] - 1) not in cluster_nk_OrgIndex[index_1]:  # 分配第一个点
        #             if (self.CL_index[i][0] - 1) not in link_points:
        #                 link_points.append(self.CL_index[i][0] - 1)
        #                 min_dis_sample[self.CL_index[i][0] - 1] = distance_cl_1[index_1]  # 该点到该簇中心的距离
        #                 cluster_radius_sum[index_1] += distance_cl_1[index_1]    # 该簇到中心点的距离和
        #                 list1[self.CL_index[i][0] - 1] = index_1     # 把该点所属的cluster加到list1中
        #                 cluster_nk_OrgIndex[index_1] = np.hstack(
        #                     (cluster_nk_OrgIndex[index_1], (self.CL_index[i][0] - 1)))  #把该点加到所属簇nk中
        #                 if distance_cl_1[index_1] > radmax[index_1]:  # 该簇的最大距离
        #                     radmax[index_1] = distance_cl_1[index_1]
        #         if (self.CL_index[i][1] - 1) not in cluster_nk_OrgIndex[index_2]:# 分配第二个点
        #             if (self.CL_index[i][1] - 1) not in link_points:
        #                 link_points.append(self.CL_index[i][1] - 1)
        #                 min_dis_sample[self.CL_index[i][1] - 1] = distance_cl_2[index_2]   # 该点到该簇中心的距离
        #                 cluster_radius_sum[index_2] += distance_cl_2[index_2]  # 该簇到中心点的距离和
        #                 list1[self.CL_index[i][1] - 1] = index_2  # 把该点所属的cluster加到list1中
        #                 cluster_nk_OrgIndex[index_2] = np.hstack(
        #                     (cluster_nk_OrgIndex[index_2], (self.CL_index[i][1] - 1)))  # 把该点加到所属簇nk中
        #                 if distance_cl_2[index_2] > radmax[index_2]:
        #                     radmax[index_2] = distance_cl_2[index_2]
        # if nc == 2:
        #     print('stop')
        # # cluster Ml-links
        # # distance_sum_must = np.zeros(self.nmust,)
        # # distance_each_must1 = np.zeros((self.nmust,))
        # # distance_each_must2 = np.zeros((self.nmust,))
        # for i in range(0, self.nmust):
        #     sample1, sample2 = self.data[self.ML_index[i][0]-1, :], self.data[self.ML_index[i][1]-1, :]
        #     distance_each_must1 = np.array([np.linalg.norm(sample1 - tmp) ** 2 for tmp in x])
        #     distance_each_must2 = np.array([np.linalg.norm(sample2 - tmp) ** 2 for tmp in x])
        #     distance_each_must = distance_each_must1 + distance_each_must2
        #     index, min_dis = np.argmin(distance_each_must), min(distance_each_must)
        #     radius_1, radius_2 = distance_each_must1[index], distance_each_must2[index]
        #     if (self.ML_index[i][0] - 1) not in cluster_nk_OrgIndex[index]:  # 分配第一个点
        #         if (self.ML_index[i][0] - 1) not in link_points:
        #             link_points.append(self.ML_index[i][0] - 1)
        #             min_dis_sample[self.ML_index[i][0] - 1] = radius_1  # 该点到该簇中心的距离
        #             cluster_radius_sum[index] += radius_1     # 该点到簇中心点的距离和
        #             list1[self.ML_index[i][0] - 1] = index     # 把该点所属的cluster加到list1中
        #             cluster_nk_OrgIndex[index] = np.hstack((cluster_nk_OrgIndex[index],
        #                                                     (self.ML_index[i][0] - 1)))   # 把该点加到所属簇nk中
        #             if radius_1 > radmax[index]:    # 该簇的最大距离
        #                 radmax[index] = radius_1
        #     if (self.ML_index[i][1] - 1) not in cluster_nk_OrgIndex[index]:   # 分配第二个点
        #         if (self.ML_index[i][1] - 1) not in link_points:
        #             link_points.append(self.ML_index[i][1] - 1)
        #             min_dis_sample[self.ML_index[i][1] - 1] = radius_2
        #             cluster_radius_sum[index] += radius_2
        #             list1[self.ML_index[i][1]-1] = index
        #             cluster_nk_OrgIndex[index] = np.hstack((cluster_nk_OrgIndex[index],
        #                                                     (self.ML_index[i][1]-1)))
        #             #cluster_nk_OrgIndex[index] = np.concatenate(cluster_nk_OrgIndex[index], self.ML_index[i][0]-1, self.ML_index[i][1]-1)
        #             if radius_2 > radmax[index]:
        #                 radmax[index] = radius_2



        for i in range(0, nc):
            cluster_nk_OrgIndex[i] = []

        for j, sample in enumerate(list1):
            cluster_nk_OrgIndex[sample].append(j)

        for i in range(0, nc):
            nel[i] = len(cluster_nk_OrgIndex[i])

        cluster_radius_sum = np.zeros((nc,))  # 计算每个簇到中心点的距离和及离簇中心最远的点的距离

        for i in range(0, nc):
            if len(cluster_nk_OrgIndex[i]) == 0:
                continue
            else:
                cluster_radius[i] = rad_sum[i] / nel[i]  # fortran 算法
                # cluster_radius_sum[i] = min_dis_sample[cluster_nk_OrgIndex[i]].sum() # 我的写法
                # cluster_radius[i] = min_dis_sample[cluster_nk_OrgIndex[i]].mean()
                # radmax[i] = max(min_dis_sample[cluster_nk_OrgIndex[i]])



        f = min_dis_sample.sum()

        if nc == 1:
            list1[:] = 0


        with open('./results/results_test.txt', 'a') as file:
            file.write(f'\t radius_nel\t {cluster_radius}\t {nel}\t\n')

        if self.nrecord < 500:
            for i in range(0, nc):
                cluster_radius[i] = 0.0

        if nc > 5 and self.nrecord > 500:
            ratio = np.zeros((nc,), dtype=float)
            for i in range(0, nc):
                ratio[i] = radmax[i]/cluster_radius[i]
            ratmin = min(ratio)
            for j in range(0, nc):
                step1 = 5.0e-1 * ratmin / ratio[j]
                cluster_radius[j] = cluster_radius[j] + step1 * (radmax[j] - cluster_radius[j])

        # lcand = np.zeros((10000,), dtype=int)
        # lcand1 = np.zeros((10000,), dtype=int)
        # if nc < self.num_clusters_max:
        #     ncand = 0
        #     for i in range(0, nc):
        #         if nel[i] > 2:
        #             toler3 = 5.0e-1 * cluster_radius[i]
        #             ncand1 = 0
        #             for j in range(0, len(cluster_nk_OrgIndex[i])):
        #                 if ncand1 == 0:
        #                     if min_dis_sample[cluster_nk_OrgIndex[i][j]] > cluster_radius[i]:
        #                         ncand = ncand+1
        #                         ncand1 = ncand1 + 1
        #                         lcand[ncand] = cluster_nk_OrgIndex[i][j]
        #                         lcand1[ncand1] = cluster_nk_OrgIndex[i][j]
        #                 if ncand1 > 0:
        #                     if min_dis_sample[cluster_nk_OrgIndex[i][j]] > cluster_radius[i]:
        #                         flag_cluster = False
        #                         while True:
        #                             for k in range(0, ncand1):
        #                                 j1 = lcand1[k]
        #                                 dis4 = np.linalg.norm(self.data[cluster_nk_OrgIndex[i][j], :] - self.data[j1, :]) ** 2
        #                                 if dis4 <= toler3:
        #                                     flag_cluster = True
        #                                     break
        #                             if flag_cluster:
        #                                 break
        #                             ncand = ncand+1
        #                             ncand1 = ncand1 + 1
        #                             lcand[ncand] = cluster_nk_OrgIndex[i][j]
        #                             lcand1[ncand1] = cluster_nk_OrgIndex[i][j]
        #
        #     lcand = lcand[0:ncand]
        ncand = 0
        lcand = []
        if nc < self.num_clusters_max:
            lcand = []
            ncand = 0
            for i in range(0, nc):
                if nel[i] > 2:
                    toler3 = 5.0e-1 * cluster_radius[i]
                    ncand1 = 0
                    lcand1 = []
                    for j in range(0, len(cluster_nk_OrgIndex[i])):
                        if min_dis_sample[cluster_nk_OrgIndex[i][j]] > cluster_radius[i]:
                            if ncand1 == 0:
                                ncand += 1
                                ncand1 += 1
                                lcand.append(cluster_nk_OrgIndex[i][j])
                                lcand1.append(cluster_nk_OrgIndex[i][j])
                            else:
                                Flag_ncand = False
                                while True:
                                    data_points = self.data[lcand1]
                                    sample_point = self.data[cluster_nk_OrgIndex[i][j]]
                                    for k in range(0, ncand1):
                                        distance_points = np.linalg.norm(sample_point - self.data[lcand1[k]]) ** 2
                                        if distance_points <= toler3:
                                            Flag_ncand = True
                                            break
                                    if Flag_ncand:
                                        break
                                    ncand1 += 1
                                    ncand += 1
                                    lcand.append(cluster_nk_OrgIndex[i][j])
                                    lcand1.append(cluster_nk_OrgIndex[i][j])
                                    break
                                    # distance_points = np.array([np.linalg.norm(sample_point
                                    #                                            - tmp) ** 2 for tmp in data_points])
                                    # if (distance_points > toler3).all():
                                    #     ncand1 += 1
                                    #     lcand.append(cluster_nk_OrgIndex[i][j])

                #ncand += ncand1

        with open('./results/results_test.txt', 'a') as file:
            file.write(f'\t ncand\t {ncand}\t\n')

        if nc == 3:
            print('hello')
        return f, list1, ncand, lcand, min_dis_sample

    # def cluster(self,x,nc,f):
    #     nel = np.zeros((nc, ), dtype=int)
    #     cluster_radius = np.zeros((nc, ), dtype=float)  # 计算每个簇的半径
    #     radmax = np.zeros((nc, 1), dtype=float)
    #     cluster_nk_OrgIndex = []
    #     list1 = np.zeros((self.nrecord,), dtype=int)
    #     min_dis_sample = np.zeros((self.nrecord, ))
    #     if nc == 1:
    #         cluster_plan_index, cluster_plan_nk, min_dis_sample_1, cluster_plan = self._assign_data(self.data, x, nc)
    #         min_dis_sample = min_dis_sample_1
    #         cluster_nk_OrgIndex.append(cluster_plan_nk[0][0])
    #         nel[nc-1] = self.nrecord
    #         cluster_radius[nc-1] = f/nel[nc-1]
    #         radmax[nc-1] = max(min_dis_sample_1)
    #     else:
    #         #cluster normal
    #         list_nrecord = np.array([sample for sample in range(0, self.nrecord)]) # 创建原始数据下标
    #         list_pair = np.r_[self.CL_index[:,0], self.CL_index[:,1], self.ML_index[:,0], self.ML_index[:,0]] - 1 #约束数据点所在下标
    #         list_normalIndex = [sample for sample in list_nrecord if sample not in list_pair] #正常数据点所在下标
    #         nbar = np.size(list_normalIndex)
    #         list_normalIndex1 = np.array(list_normalIndex) # txt 文档的下标从1开始，python从0开始
    #         normal_data = self.data[list_normalIndex1]
    #         cluster_plan_index, cluster_plan_nk, min_dis_sample_1, cluster_plan = self._assign_data(normal_data, x, nc)
    #         list1[list_normalIndex1] = cluster_plan_index  #向量：每个数据点所属的cluster
    #
    #         for i in range(0, len(cluster_plan_nk)):  # 计算簇中每个点的原index
    #             OrdIndex = list_normalIndex1[cluster_plan_nk[i][0]]
    #             cluster_nk_OrgIndex.append(OrdIndex)
    #
    #         cluster_radius_sum = np.zeros((nc,)) # 计算每个簇到中心点的距离和及离簇中心最远的点的距离
    #         for i, cluster in enumerate(cluster_plan_nk):
    #             if cluster[0].size == 0:
    #                 continue
    #             else:
    #                 cluster_radius_sum[i] = min_dis_sample_1[cluster[0]].sum()
    #                 radmax[i] = max(min_dis_sample_1[cluster[0]])
    #
    #         min_dis_sample[list_normalIndex1] = min_dis_sample_1 # 将normal数据的中心距离输入到min_dis_sample中
    #
    #
    #         # cluster CL-links
    #         distance_cl = np.zeros((nc,nc))
    #         m1 = 9e26
    #         for i in range(0, self.ncan):
    #             sample1, sample2 = self.data[self.CL_index[i][0] - 1, :], self.data[self.CL_index[i][1] - 1, :]
    #             distance_cl_1 = np.array([np.linalg.norm(sample1 - tmp) ** 2 for tmp in x])
    #             distance_cl_2 = np.array([np.linalg.norm(sample2 - tmp) ** 2 for tmp in x])
    #             for k in range(0, nc):
    #                 for j in range(0, nc):
    #                     if k == j:
    #                         distance_cl[k][j] = m1
    #                     else:
    #                         distance_cl[k][j] = distance_cl_1[k] + distance_cl_2[j]
    #             min_value = np.min(distance_cl)  # 找到最小值
    #             index = np.where(distance_cl == min_value)  # 返回最小值下标，形式为tuple
    #             index_1, index_2 = index[0][0], index[1][0]  # 提取两个点对应所在的簇
    #             if (self.CL_index[i][0] - 1) not in cluster_nk_OrgIndex[index_1]:  # 分配第一个点
    #                 min_dis_sample[self.CL_index[i][0] - 1] = distance_cl_1[index_1]  # 该点到该簇中心的距离
    #                 cluster_radius_sum[index_1] += distance_cl_1[index_1]    # 该簇到中心点的距离和
    #                 list1[self.CL_index[i][0] - 1] = index_1     # 把该点所属的cluster加到list1中
    #                 cluster_nk_OrgIndex[index_1] = np.hstack(
    #                     (cluster_nk_OrgIndex[index_1], (self.CL_index[i][0] - 1)))  #把该点加到所属簇nk中
    #                 if distance_cl_1[index_1] > radmax[index_1]:  # 该簇的最大距离
    #                     radmax[index_1] = distance_cl_1[index_1]
    #             if (self.CL_index[i][1] - 1) not in cluster_nk_OrgIndex[index_2]: # 分配第二个点
    #                 min_dis_sample[self.CL_index[i][1] - 1] = distance_cl_2[index_2]   # 该点到该簇中心的距离
    #                 cluster_radius_sum[index_2] += distance_cl_2[index_2]  # 该簇到中心点的距离和
    #                 list1[self.CL_index[i][1] - 1] = index_2  # 把该点所属的cluster加到list1中
    #                 cluster_nk_OrgIndex[index_2] = np.hstack(
    #                     (cluster_nk_OrgIndex[index_2], (self.CL_index[i][1] - 1)))  # 把该点加到所属簇nk中
    #                 if distance_cl_2[index_2] > radmax[index_2]:
    #                     radmax[index_2] = distance_cl_2[index_2]
    #
    #             # cluster Ml-links
    #             # distance_sum_must = np.zeros(self.nmust,)
    #             # distance_each_must1 = np.zeros((self.nmust,))
    #             # distance_each_must2 = np.zeros((self.nmust,))
    #             for i in range(0, self.nmust):
    #                 sample1, sample2 = self.data[self.ML_index[i][0]-1, :], self.data[self.ML_index[i][1]-1, :]
    #                 distance_each_must1 = np.array([np.linalg.norm(sample1 - tmp) ** 2 for tmp in x])
    #                 distance_each_must2 = np.array([np.linalg.norm(sample2 - tmp) ** 2 for tmp in x])
    #                 distance_each_must = distance_each_must1 + distance_each_must2
    #                 index, min_dis = np.argmin(distance_each_must), min(distance_each_must)
    #                 radius_1, radius_2 = distance_each_must1[index], distance_each_must2[index]
    #                 if (self.ML_index[i][0] - 1) not in cluster_nk_OrgIndex[index]:  # 分配第一个点
    #                     min_dis_sample[self.ML_index[i][0] - 1] = radius_1  # 该点到该簇中心的距离
    #                     cluster_radius_sum[index] += radius_1     # 该点到簇中心点的距离和
    #                     list1[self.ML_index[i][0] - 1] = index     # 把该点所属的cluster加到list1中
    #                     cluster_nk_OrgIndex[index] = np.hstack((cluster_nk_OrgIndex[index],
    #                                                             (self.ML_index[i][0] - 1)))   # 把该点加到所属簇nk中
    #                     if radius_1 > radmax[index]:    # 该簇的最大距离
    #                         radmax[index] = radius_1
    #                 if (self.ML_index[i][1] - 1) not in cluster_nk_OrgIndex[index]:   # 分配第二个点
    #                     min_dis_sample[self.ML_index[i][1] - 1] = radius_2
    #                     cluster_radius_sum[index] += radius_2
    #                     list1[self.ML_index[i][1]-1] = index
    #                     cluster_nk_OrgIndex[index] = np.hstack((cluster_nk_OrgIndex[index],
    #                                                             (self.ML_index[i][1]-1)))
    #                     #cluster_nk_OrgIndex[index] = np.concatenate(cluster_nk_OrgIndex[index], self.ML_index[i][0]-1, self.ML_index[i][1]-1)
    #                     if radius_2 > radmax[index]:
    #                         radmax[index] = radius_2
    #
    #         f = min_dis_sample.sum()
    #
    #         for i in range(0, nc):
    #             nel[i] = len(cluster_nk_OrgIndex[i])
    #
    #         #cluster_radius = np.zeros((nc,))  # 计算每个簇的半径
    #         for i, cluster in enumerate(cluster_nk_OrgIndex):
    #             if cluster[0].size == 0:
    #                 continue
    #             else:
    #                 cluster_radius[i] = min_dis_sample[cluster[0]].mean()
    #
    #     if self.nrecord < 500:
    #         cluster_radius = 0.0
    #
    #     ratio = np.zeros((nc,), dtype=float)
    #     if nc > 5 and self.nrecord > 500:
    #         for i in range(0, nc):
    #             ratio[i] = radmax[i]/cluster_radius[i]
    #         ratmin = min(ratio)
    #         for j in range(0, nc):
    #             step1 = 5.0e-1 * ratmin / ratio[j]
    #             cluster_radius[j] = cluster_radius[j] + step1 * (radmax[j] - cluster_radius[j])
    #
    #     if nc < self.num_clusters_max:
    #         #lcand = np.zeros((0, 1), dtype=int)
    #         lcand = []
    #         ncand = 0
    #         for i in range(0, nc):
    #             ncand1 = 0
    #             if nel[i] > 2:
    #                 toler3 = 5.0e-1 * cluster_radius[i]
    #                 if nc > 1:
    #                     print('hello4')
    #                 for j in range(0, len(cluster_nk_OrgIndex[i])):
    #                     if min_dis_sample[cluster_nk_OrgIndex[i][j]] > cluster_radius[i]:
    #                         if ncand1 == 0:
    #                             ncand1 += 1
    #                             lcand.append(cluster_nk_OrgIndex[i][j])
    #                         else:
    #                             data_points = self.data[lcand]
    #                             sample_point = self.data[cluster_nk_OrgIndex[i][j]]
    #                             distance_points = np.array([np.linalg.norm(sample_point
    #                                                                        - tmp) ** 2 for tmp in data_points])
    #                             if (distance_points > toler3).all():
    #                                 ncand1 += 1
    #                                 lcand.append(cluster_nk_OrgIndex[i][j])
    #
    #             ncand += ncand1
    #     print('nc,ncand:', nc, ncand)
    #
    #     return f, list1, ncand, lcand, min_dis_sample

    def _assign_data(self, dataset, x, nc):
        cluster_plan = np.zeros((nc, dataset.shape[0]), dtype=int)
        min_dis_sample_1 = np.zeros((dataset.shape[0],))

        for i, sample in enumerate(dataset):
            distance = np.array([np.linalg.norm(sample - tmp) ** 2 for tmp in x])
            index = np.argmin(distance)
            min_dis_sample_1[i] = distance[index]
            cluster_plan[index, i] = 1  # 0,1 矩阵

        cluster_plan_index = np.argmax(cluster_plan, axis=0)  # 向量：每个数据点所属的cluster
        cluster_plan_nk = [np.where(tmp == 1) for tmp in cluster_plan]  # 列表：每个元素存储属于该cluster点的index

        return cluster_plan_index, cluster_plan_nk, min_dis_sample_1, cluster_plan

    def finresult(self, list1, min_dis_sample, x, nc):
        if nc == 1:
            distance = np.linalg.norm(self.data - x[0, :],  axis=1) ** 2
            f = distance.sum()
        else:
            f = min_dis_sample.sum()

        num_labels = len(np.unique(list1))
        viol, vcan, vmust, pur, NML = 0, 0, 0, 0, 0

        # Calculation of Violations
        mcval = 0
        mmval = 0
        for i in range(0, self.ncan):
            index1, index2 = self.CL_index[i][0]-1, self.CL_index[i][1]-1
            if list1[index1] != list1[index2]:
                mcval += 1

        for j in range(0, self.nmust):
            index1, index2 = self.ML_index[j][0]-1, self.ML_index[j][1]-1
            if list1[index1] == list1[index2]:
                mmval += 1

        mt = self.ncan + self.nmust
        mt1 = mcval + mmval
        if mt > 0:
            viol = 1.0e2 * float(mt1)/float(mt)
        if mt == 0:
            viol = 0.00

        if self.ncan > 0:
            vcan = 1.0e2 * float(mcval)/float(self.ncan)
        if self.ncan == 0:
            vcan = 0.00

        if self.nmust > 0:
            vmust = 1.0e2 * float(mmval)/float(self.nmust)
        if self.nmust == 0:
            vmust = 0.00

        # Purity
        if self.label:
            pur = 1.0e02 * self.purity(list1, self.Classlabel)

        # dbi
        if num_labels == 1:
            dbi = 0
        else:
            dbi = davies_bouldin_score(self.data, list1)

        #Dunn validity index (DVI)
        #dn =

        # silhouette
        if num_labels == 1:
            sil_avg = 0
            sil_pos = 0
        else:
            sil_avg = silhouette_score(self.data, list1, metric='euclidean') # 返回平均值
            sil_samples = silhouette_samples(self.data, list1, metric='euclidean') # 返回每个点的sil值
            s = [num_pos for num_pos in sil_samples if num_pos > 0]
            #sil_pos = len(s)/self.nrecord  # sil值为正的比例
            sil_pos = len(s)   # sil值为正的个数


        # Normalized mutual information
        if self.label:
            NML = normalized_mutual_info_score(self.Classlabel, list1)

        return f,viol, vcan, vmust,pur,dbi,sil_avg,sil_pos,NML

    def _main_alg(self, start_time, w):
        log_string = ''
        log_str = f'\t nc \t function\t viol\t vcan\t vmust\t purity\t sil_avg\t sil_pos\t DBI\t'
        log_string = '\n'.join([log_string, log_str])
        with open('./results/result_%s.txt' % self.dataset_name, 'a') as file:
            file.write(log_string)
            file.write('\n')

        for nc in range(1, self.num_clusters + 1):
            with open('./results/results_test.txt', 'a') as file:
                file.write(f'\t _main_alg_nc\t {nc}\t\n')
            if nc == 1:
                f, x = self.step1(w)
                print('f', f)
                toler = 1.0e-2 * f / float(self.nrecord)
                f, list1, ncand, lcand, min_dis_sample = self.cluster(x, nc, f)
                print('f', f)
                end_time = time.time()
                f, viol, vcan, vmust, pur, dbi, sil_avg, sil_pos, NML = self.finresult(list1, min_dis_sample, x, nc)
                time4 = end_time - start_time
                if time4 > self.tlimit:
                    break
            else:
                nstart, x2 = self._step2(toler, lcand, min_dis_sample, w)
                if nc == 6:
                    with open('./results/results_test.txt', 'a') as file:
                        file.write(f'\t _main_alg_x2\t {x2}\t\n')
                print('nstart:', nstart)
                fval = np.zeros((nstart,))
                for i, sample1 in enumerate(x2):
                    m = self.num_feature
                    # if nc == 6:
                    #     sample1 = x2[4, :]
                    ns = 1
                    z, barf = dgm(self.data, m, sample1, min_dis_sample, ns, nc)
                    if nc == 6:
                        with open('./results/results_test.txt', 'a') as file:
                            file.write(f'\t main_z\t fval\t {z}\t fval\t\n')
                    if nc == 6:
                        print('hello', z)
                    fval[i] = barf
                    x2[i, :] = z

                fbarmin, fbarmax = min(fval), max(fval)
# ================================================================
                fbarmin = fbarmin + self.gamma3 * (fbarmax - fbarmin)
                index_final = np.where(fval <= fbarmin)
                nstart = len(index_final[0])
                fval = fval[index_final[0]]
                x2 = x2[index_final[0], :]
# ================================================================
                x5 = np.empty_like(x2)
                x5[0, :] = x2[0, :]
                nstart2 = 1

                inner_break = False
                for i in range(1, nstart):
                    while True:
                        for j in range(0,nstart2):
                            distance = np.linalg.norm(x5[j, :]-x2[i, :])**2
                            if distance <= toler:
                                inner_break = True
                                break
                        if inner_break:
                            break
                        nstart2 += 1
                        x5[nstart2-1, :] = x2[i, :]
                x2 = x5[0:nstart2, :]
                nstart = nstart2
#                 with open('./results/results_test.txt', 'a') as file:
#                     file.write(f'\t main_x2\t {x2}\t\n')
# # ================================================================
                fbest = 1.0e+28
                for i in range(0, nstart):
                    x3 = np.vstack((x, x2[i, :]))
                    x3 = x3.reshape(np.size(x3), )
                    # if nc == 5:
                    #     with open('./results/results_test.txt', 'a') as file:
                    #         file.write(f'\t main_x3\t {x3}\t\n')
                    m = np.size(x3)
                    ns = 2
                    sample, fcurrent = dgm(self.data, m, x3, min_dis_sample, ns, nc)
                    # if nc == 5:
                    #     with open('./results/results_test.txt', 'a') as file:
                    #         file.write(f'\t main_sample\t {sample}\t\n')
                    if fcurrent < fbest:
                        xbest = sample
                # if nc == 5:
                #     print('hello')
                f = fbest
                x = np.zeros((nc, self.num_feature))
                for i in range(nc):
                    x[i, :] = xbest[i*self.num_feature:(i+1)*self.num_feature].T

                # with open('./results/result_example.txt', 'a') as file:
                #      file.write(f'\t nc\t {nc}\t\n'
                #                 f'\t x\t {x}\t\n')
                f, list1, ncand, lcand, min_dis_sample = self.cluster(x, nc, f)
                end_time = time.time()
                f, viol, vcan, vmust, pur, dbi, sil_avg, sil_pos, NML = self.finresult(list1, min_dis_sample, x, nc)
                time4 = end_time - start_time
                if time4 > self.tlimit:
                    break

            # log_str = f'\t {nc} \t {f}\t {viol}\t {vcan}\t {vmust}\t {pur}\t {sil_avg}\t {sil_pos}\t{dbi}\t'
            # log_string = '\n'.join([log_string, log_str])
            with open('./results/result_%s.txt' % self.dataset_name, 'a') as file:
                file.write(f'\t {nc} \t {round(f,4)}\t {round(viol,4)}\t {round(vcan,4)}\t'
                           f' {round(vmust,4)}\t {round(pur,4)}\t {round(sil_avg,4)}\t {round(sil_pos,4)}\t {round(dbi,4)}\t')
                file.write('\n')
            print('')


    def run(self):
        start_time = time.time()
        self.input()
        self.nrecord = self.data.shape[0]
        self.num_feature = self.data.shape[1]
        self.para_choose()
        w = np.ones((self.nrecord, 1))
        with open('./results/result_%s.txt' % self.dataset_name, 'a') as file:
            file.write(f'Current time: {time.asctime(time.localtime(time.time()))}''\r\n')
        self._main_alg(start_time, w)


if __name__ == '__main__':
    a = IncSemisupervisedClusteringAlgo()
    a.run()


