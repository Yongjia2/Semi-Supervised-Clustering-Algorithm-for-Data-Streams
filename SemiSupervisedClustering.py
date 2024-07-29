# This is semi-supervised cluster algorithm
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
Time: 2023-10-12
Author: Yongjia Yuan
"""


class IncSemisupervisedClusteringAlgo:

    def __init__(self, data, label, dataset_name, num_clusters, ML_index, CL_index, w, xsolution, nblocks):
        self.data = data
        self.label = label
        self.dataset_name = dataset_name
        self.nrecord = self.data.shape[0]
        self.num_feature = self.data.shape[1]
        self.num_clusters = num_clusters
        self.CL_index = CL_index
        self.ncan = self.CL_index.shape[0]
        self.ML_index = ML_index
        self.nmust = self.ML_index.shape[0]
        self.num_clusters_max = 100
        self.w = w
        self.nblocks = nblocks
        self.xsolution = xsolution

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

    def step1(self):     # 已加入权重w
        x = np.dot(self.data.T, self.w / np.sum(self.w))
        distance = np.linalg.norm(self.data - x.reshape((1, x.size)), axis=1) ** 2
        f = np.dot(distance, self.w)
        x = x.T
        return f, x

    def _step2(self, toler, lcand, min_dis_sample): # 已加入w
        fmin1 = np.zeros((np.size(lcand, 0), 1), dtype=float)
        x2 = np.zeros((0, self.num_feature))
        x4 = np.zeros((1, self.num_feature))

        if self.nrecord < 200:
            x2 = self.data[lcand, :]

        for i, sample in enumerate(lcand):
            distance_3 = np.array([np.linalg.norm(self.data[sample] - tmp) ** 2 for tmp in self.data])
            differ_1 = distance_3 - min_dis_sample
            index = np.where(differ_1 < 0)
            fmin1[i] = np.dot(differ_1[index], self.w[index])

        fmin1 = fmin1.flatten()
        i_min, fmin = fmin1.argmin(), fmin1.min()
        i_max, fmax = fmin1.argmax(), fmin1.max()
        #tnorm = tnorm + np.size(lcand, 0) * self.nrecord

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
                w1 = self.w[index2].sum()
                # for k in range(self.num_feature):
                #     d33 = 0.0
                #     for j in range(len(index2[0])):
                #         d33 = d33+self.w[index2[0][j]]*self.data[index2[0][j], k]
                #     if self.nblocks == 2 and i == 0:
                #         print('d33', d33)
                #     x4[0, k] = d33/w1
                x4 = np.dot(self.data[index2].T, self.w[index2] / w1).T

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
        decrease_d21 = np.dot(new_distance_start, self.w).flatten()
        max_dec = decrease_d21.min()
        d6 = decrease_d21.max()

        d2 = max_dec + self.gamma2 * (d6-max_dec)
        index_d2 = np.where(decrease_d21 <= d2)
        nstart = len(index_d2[0])
        x2 = x2[index_d2]


        return nstart, x2 #tnorm


    def cluster(self,x,nc): #已加入权重
        nel = np.zeros((nc, ), dtype=int)
        cluster_radius = np.zeros((nc, ), dtype=float)  # 计算每个簇的半径
        #cluster_nk_OrgIndex = {}
        cluster_nk_OrgIndex = [[] for _ in range(nc)]
        rad_sum = np.zeros((nc,))
        radmax = np.zeros((nc,))
        radmax = np.zeros((nc, 1), dtype=float)
        list1 = np.zeros((self.nrecord,), dtype=int)
        min_dis_sample = np.zeros((self.nrecord, ))

        with open('./results/test.txt', 'a') as file:
            file.write(f'nc,cluster_input_x:{nc}\t{x}\r\n')

        # cluster normal
        list_normalIndex = []
        for i in range(self.nrecord):
            if (i not in self.CL_index) and (i not in self.ML_index):
                list_normalIndex.append(i)
        list_normalIndex1 = np.array(list_normalIndex)

        normal_data = self.data[list_normalIndex1]
        cluster_plan_index, cluster_plan_nk, min_dis_sample_1, cluster_plan = self._assign_data(normal_data, x, nc)
        list1[list_normalIndex1] = cluster_plan_index  #向量：每个数据点所属的cluster
        min_dis_sample[list_normalIndex1] = min_dis_sample_1  # 将normal数据的中心距离输入到min_dis_sample中

        # for i in range(0, nc):  # Fortran算法
        #     rad_sum[i] = min_dis_sample_1[cluster_plan_nk[i]].sum()
        #     radmax[i] = max(min_dis_sample_1[cluster_plan_nk[i]])

        # cluster cl-links
        for i in range(0, self.ncan):
            f_sum = 1.0e+22
            sample1, sample2 = self.data[self.CL_index[i][0], :], self.data[self.CL_index[i][1], :]
            for j in range(0, nc):
                for jj in range(0, nc):
                    if j != jj:
                        distance_cl_1 = np.linalg.norm(sample1 - x[j, :]) ** 2
                        distance_cl_2 = np.linalg.norm(sample2 - x[jj, :]) ** 2
                        distance_sum = self.w[self.CL_index[i][0]]*distance_cl_1 + self.w[self.CL_index[i][1]]*distance_cl_2
                        if f_sum > distance_sum:
                            f_sum = distance_sum
                            list1[self.CL_index[i][0]] = j      # 把该点所属的cluster加到list1中
                            list1[self.CL_index[i][1]] = jj
                            min_dis_sample[self.CL_index[i][0]] = distance_cl_1  # 该点到该簇中心的距离
                            min_dis_sample[self.CL_index[i][1]] = distance_cl_2

        print('cl_sum:', np.dot(min_dis_sample, self.w))

            # rad_sum[list1[self.CL_index[i][0]]] += min_dis_sample[self.CL_index[i][0]] #Fortran算法
            # rad_sum[list1[self.CL_index[i][1]]] += min_dis_sample[self.CL_index[i][1]]
            # if min_dis_sample[self.CL_index[i][0]] > radmax[list1[self.CL_index[i][0]]]:  # 该簇的最大距离
            #     radmax[list1[self.CL_index[i][0]]] = min_dis_sample[self.CL_index[i][0]]
            # if min_dis_sample[self.CL_index[i][1]] > radmax[list1[self.CL_index[i][1]]:
            #     radmax[list1[self.CL_index[i][1]]] = min_dis_sample[self.CL_index[i][1]]

        # cluster Ml-links
        for i in range(0, self.nmust):
            f_sum = 1.0e+22
            sample1, sample2 = self.data[self.ML_index[i][0], :], self.data[self.ML_index[i][1], :]
            for j in range(0, nc):
                distance_cl_1 = np.linalg.norm(sample1 - x[j, :]) ** 2
                distance_cl_2 = np.linalg.norm(sample2 - x[j, :]) ** 2
                distance_sum = self.w[self.ML_index[i][0]]*distance_cl_1 + self.w[self.ML_index[i][1]]*distance_cl_2
                if f_sum > distance_sum:
                    f_sum = distance_sum
                    list1[self.ML_index[i][0]] = j  # 把该点所属的cluster加到list1中
                    list1[self.ML_index[i][1]] = j
                    min_dis_sample[self.ML_index[i][0]] = distance_cl_1  # 该点到该簇中心的距离
                    min_dis_sample[self.ML_index[i][1]] = distance_cl_2
            # rad_sum[list1[self.ML_index[i][0]]] += min_dis_sample[self.ML_index[i][0]]  #Fortran算法
            # rad_sum[list1[self.ML_index[i][1]]] += min_dis_sample[self.ML_index[i][1]]
            # if min_dis_sample[self.ML_index[i][0]] > radmax[list1[self.ML_index[i][0]]]:  # 该簇的最大距离
            #     radmax[list1[self.ML_index[i][0]]] = min_dis_sample[self.ML_index[i][0]]
            # if min_dis_sample[self.ML_index[i][1]] > radmax[list1[self.ML_index[i][1]]]:
            #     radmax[list1[self.ML_index[i][1]]] = min_dis_sample[self.ML_index[i][1]]

        print('ml_sum:', np.dot(min_dis_sample, self.w))
        # for i in range(0, nc):
        #     cluster_nk_OrgIndex[i] = []

        for k, sample in enumerate(list1):
            cluster_nk_OrgIndex[sample].append(k)

        for i in range(0, nc):
            nel[i] = len(cluster_nk_OrgIndex[i])

        cluster_radius_sum = np.zeros((nc,))  # 计算每个簇到中心点的距离和及离簇中心最远的点的距离

        for i in range(0, nc):
            if len(cluster_nk_OrgIndex[i]) == 0:
                continue
            else:
                #cluster_radius[i] = rad_sum[i] / nel[i]  # fortran 算法
                cluster_radius_sum[i] = min_dis_sample[cluster_nk_OrgIndex[i]].sum() # 我的写法
                cluster_radius[i] = np.dot(min_dis_sample[cluster_nk_OrgIndex[i]], self.w[cluster_nk_OrgIndex[i]])/np.size(cluster_nk_OrgIndex[i])
                radmax[i] = max(min_dis_sample[cluster_nk_OrgIndex[i]])

        Org_clu_rad = cluster_radius


        f = np.dot(min_dis_sample, self.w)  # 加入了权重w

        if nc == 1:
            list1[:] = 0

        if self.nrecord < 500:
            for i in range(0, nc):
                cluster_radius[i] = 0.0

        # if nc > 5 and self.nrecord > 500:
        #     ratio = np.zeros((nc,), dtype=float)
        #     for i in range(0, nc):
        #         ratio[i] = radmax[i]/cluster_radius[i]
        #     ratmin = min(ratio)
        #     for j in range(0, nc):
        #         step1 = 5.0e-1 * ratmin / ratio[j]
        #         cluster_radius[j] = cluster_radius[j] + step1 * (radmax[j] - cluster_radius[j])

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

        if self.nblocks == 3:
            with open('./results/test.txt', 'a') as file:
                file.write(f'clusters:f,nc,nel, clusters-rad,cluster_radius_sum:{f}\t{nc}\t{nel} \t'
                           f'{cluster_radius_sum} \r\n'
                           f'ncand\t {ncand} \r\n')
        print('nc,ncand',nc,ncand)
        return f, list1, ncand, lcand, min_dis_sample, cluster_nk_OrgIndex, cluster_radius


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

    def _main_alg(self):
        for nc in range(1, self.num_clusters + 1):
            if nc == 1:
                f, x = self.step1()
                print('f', f)
                toler = 1.0e-2 * f / float(self.nrecord)
                f, list1, ncand, lcand, min_dis_sample, cluster_nk_OrgIndex, Org_clu_rad = self.cluster(x, nc)
                print('f', f)
                # f, viol, vcan, vmust, pur, dbi, sil_avg, sil_pos, NML = self.finresult(list1, min_dis_sample, x, nc)
            else:
                nstart, x2 = self._step2(toler, lcand, min_dis_sample)
                # with open('./results/test.txt', 'a') as file:
                #   file.write(f' nc,step2-x2: {nc}\t {x2} \t \r\n')
                print('nblocks,nc,nstart:', self.nblocks, nc, nstart)
                fval = np.zeros((nstart,))
                for i, sample1 in enumerate(x2):
                    m = self.num_feature
                    # if self.nblocks == 2:
                    #     with open('./results/test.txt', 'a') as file:
                    #         file.write(f'main_x2_before: {i}\t {sample1} \t   \r\n')
                    ns = 1
                    z, barf = dgm(self.data, m, sample1, min_dis_sample, ns, nc, self.w, self.nblocks)
                    # if self.nblocks == 2:
                    #     with open('./results/test.txt', 'a') as file:
                    #         file.write(f'main_x2_after: {i}\t {z} \t   \r\n')
                    #     print('stop')
                    fval[i] = barf
                    x2[i, :] = z

                # if self.nblocks == 2:
                #     with open('./results/test.txt', 'a') as file:
                #         file.write(f'main_x2: {nc}\t {x2} \t   \r\n')
                #     print('stop')

                fbarmin, fbarmax = min(fval), max(fval)
                print(fbarmin, fbarmax)
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

                for i in range(1, nstart):
                    inner_break = False
                    while True:
                        for j in range(0,nstart2):
                            distance = np.linalg.norm(x5[j, :]-x2[i, :])**2
                            # if nc==10:
                            #     print("nc_start,toler",distance,toler)
                            if distance <= toler:
                                inner_break = True
                                break
                        if inner_break:
                            break
                        nstart2 += 1
                        # if nc==6:
                        #     print('nc,nstart2',nc,nstart2)
                        x5[nstart2-1, :] = x2[i, :]
                x2 = x5[0:nstart2, :]
                nstart = nstart2
                print('nstart2',nstart)
                if self.nblocks > 1:
                    x2 = np.vstack((x2, self.xsolution[0, :]))
                    nstart = nstart+1
# # ================================================================
                fbest = 1.0e+28
                for i in range(0, nstart):
                    x3 = np.vstack((x, x2[i, :]))
                    x3 = x3.reshape(np.size(x3), )
                    m = np.size(x3)
                    ns = 2
                    sample, fcurrent = dgm(self.data, m, x3, min_dis_sample, ns, nc, self.w,self.nblocks)
                    if fcurrent < fbest:
                        fbest = fcurrent
                        xbest = sample
                f = fbest
                x = np.zeros((nc, self.num_feature))
                for i in range(nc):
                    x[i, :] = xbest[i*self.num_feature:(i+1)*self.num_feature].T

                f, list1, ncand, lcand, min_dis_sample, cluster_nk_OrgIndex, Org_clu_rad = self.cluster(x, nc)
                print('cluster_finish')
                with open('./results/test.txt', 'a') as file:
                    file.write(f' f,nc: {f} \t {nc} \t \r\n')

                print('f,nc:', f, nc)
                #f, viol, vcan, vmust, pur, dbi, sil_avg, sil_pos, NML = self.finresult(list1, min_dis_sample, x, nc)

        return x, min_dis_sample, cluster_nk_OrgIndex, list1, Org_clu_rad

    def run(self):
        self.para_choose()
        x, min_dis_sample, cluster_nk_OrgIndex, list1, Org_clu_rad = self._main_alg()

        return x, min_dis_sample, cluster_nk_OrgIndex, list1, Org_clu_rad



