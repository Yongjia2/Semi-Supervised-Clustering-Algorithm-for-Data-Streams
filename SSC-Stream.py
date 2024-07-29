import pandas as pd
import os
import numpy as np
import time
import math
from SemiSupervisedClustering import IncSemisupervisedClusteringAlgo
from deter_points import Clu_Der_Points

from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from collections import Counter
from sklearn import datasets

"""
time: 2023-10-10
author: Yongjia Yuan
note: this is semi-supervised cluster algorithm on real-world data set
"""

datasets = [
    'example'
]

class SemisupervisedClusteringBigDataAlgo():

    def __init__(self, data_path='./data/', num_clusters=3):
        self.data_path = data_path
        self.data = []
        self.data_path_list = []
        self.num_feature = None
        self.num_blocks = None
        self.num_clusters = num_clusters
        self.nrecord = None

    def initlization(self):
        data_list = os.listdir(self.data_path)
        data_list.sort(key=lambda ss: int(ss.split('.')[0]))
        self.data_path_list = data_list
        self.num_blocks = len(data_list)
        tmp = pd.read_csv(os.path.join(self.data_path, self.data_path_list[0]), header=None).iloc[:, :]
        self.num_feature = tmp.shape[1] - 1

    def input(self, dataset_name):
        self.datasetName = dataset_name
        self.data_path = os.path.join(self.data_path + self.datasetName)
        self.initlization()

    def _update_data(self, kblocks, b=None, w2=None):
        Org_data = pd.read_csv(os.path.join(self.data_path, self.data_path_list[kblocks - 1]), header=None, ).iloc[:,:]
        data = np.array(Org_data.iloc[:, :-1])
        self.Org_data = data
        self.Org_label = np.array(Org_data.iloc[:, -1])
        n2 = b.shape[0]
        self.nrecord1 = data.shape[0]

        if kblocks == 1:
            w = np.ones((self.nrecord1, 1))
        else:
            print('The number of points from the previous block:', n2)
            w = np.ones((np.size(data, 0), 1))
            data = np.vstack((data, b))
            w = np.vstack((w, w2))

        self.data = data
        self.w = w
        self.nrecord = data.shape[0]

    def _thresh_choose_new(self):
        if self.nrecord1 <= 10000:
            self.thresh = 3.0e-1
        elif (self.nrecord1 > 10000) and (self.nrecord1 <= 20000):
            self.thresh = 2.0e-1
        elif (self.nrecord1 > 20000) and (self.nrecord1 <= 50000):
            self.thresh = 1.0e-1
        elif self.nrecord1 > 50000:
            self.thresh = 0.5e-2

    def WholeData(self, Centers,nc):
        wholedata = pd.read_csv("./data/example_whole.txt", sep=',', index_col=None, header=None).iloc[:, :]
        self.whole_data = np.array(wholedata.iloc[:, :-1])
        self.whole_label = np.array(wholedata.iloc[:, -1])
        whole_record = np.size(self.whole_data, 0)
        whole_w = np.ones((whole_record,))

        ML_whole = np.zeros((0, 2), dtype=int)
        CL_whole = np.zeros((0, 2), dtype=int)
        # upload the constraints
        for i in range(1, self.num_blocks + 1):
            cl_block = pd.read_csv('./data/cons/CL/%d.txt' % i, sep=',', index_col=None, header=None)
            ml_block = pd.read_csv('./data/cons/ML/%d.txt' % i, sep=',', index_col=None, header=None)
            cl_block, ml_block = np.array(cl_block), np.array(ml_block)
            if i == 1:
                CL_whole = np.vstack((CL_whole, cl_block))
                ML_whole = np.vstack((ML_whole, ml_block))
            else:
                CL_whole = np.vstack((CL_whole, cl_block + self.num_DataBlock[i - 2]))
                ML_whole = np.vstack((ML_whole, ml_block + self.num_DataBlock[i - 2]))

        self.can_whole = CL_whole
        self.must_whole = ML_whole
        self.ncan_whole = np.size(CL_whole, 0)
        self.nmust_whole = np.size(ML_whole, 0)

        list1 = np.zeros((whole_record,), dtype=int)
        min_dis_sample = np.zeros((whole_record,))

        # cluster normal
        list_normalIndex = []
        for i in range(whole_record):
            if (i not in self.can_whole) and (i not in self.must_whole):
                list_normalIndex.append(i)
        list_normalIndex1 = np.array(list_normalIndex)

        normal_data = self.whole_data[list_normalIndex1]
        cluster_plan_index, cluster_plan_nk, min_dis_sample_1, cluster_plan = self._assign_data(normal_data, Centers, nc)
        list1[list_normalIndex1] = cluster_plan_index
        min_dis_sample[list_normalIndex1] = min_dis_sample_1

        # cluster cl-links
        for i in range(0, self.ncan):
            f_sum = 1.0e+22
            sample1, sample2 = self.whole_data[self.can_whole[i][0], :], self.whole_data[self.can_whole[i][1], :]
            for j in range(0, nc):
                for jj in range(0, nc):
                    if j != jj:
                        distance_cl_1 = np.linalg.norm(sample1 - Centers[j, :]) ** 2
                        distance_cl_2 = np.linalg.norm(sample2 - Centers[jj, :]) ** 2
                        distance_sum = whole_w[self.can_whole[i][0]] * distance_cl_1 + whole_w[
                            self.can_whole[i][1]] * distance_cl_2
                        if f_sum > distance_sum:
                            f_sum = distance_sum
                            list1[self.can_whole[i][0]] = j
                            list1[self.can_whole[i][1]] = jj
                            min_dis_sample[self.can_whole[i][0]] = distance_cl_1
                            min_dis_sample[self.can_whole[i][1]] = distance_cl_2

        # cluster Ml-links
        for i in range(0, self.nmust):
            f_sum = 1.0e+22
            sample1, sample2 = self.whole_data[self.must_whole[i][0], :], self.whole_data[self.must_whole[i][1], :]
            for j in range(0, nc):
                distance_cl_1 = np.linalg.norm(sample1 - Centers[j, :]) ** 2
                distance_cl_2 = np.linalg.norm(sample2 - Centers[j, :]) ** 2
                distance_sum = whole_w[self.must_whole[i][0]] * distance_cl_1 + whole_w[
                    self.must_whole[i][1]] * distance_cl_2
                if f_sum > distance_sum:
                    f_sum = distance_sum
                    list1[self.must_whole[i][0]] = j  # 把该点所属的cluster加到list1中
                    list1[self.must_whole[i][1]] = j
                    min_dis_sample[self.must_whole[i][0]] = distance_cl_1  # 该点到该簇中心的距离
                    min_dis_sample[self.must_whole[i][1]] = distance_cl_2

        return list1, min_dis_sample

    def _assign_data(self, dataset, x, nc):
        cluster_plan = np.zeros((nc, dataset.shape[0]), dtype=int)
        min_dis_sample_1 = np.zeros((dataset.shape[0],))

        for i, sample in enumerate(dataset):
            distance = np.array([np.linalg.norm(sample - tmp) ** 2 for tmp in x])
            index = np.argmin(distance)
            min_dis_sample_1[i] = distance[index]
            cluster_plan[index, i] = 1  # 0,1 矩阵

        cluster_plan_index = np.argmax(cluster_plan, axis=0)
        cluster_plan_nk = [np.where(tmp == 1) for tmp in cluster_plan]

        return cluster_plan_index, cluster_plan_nk, min_dis_sample_1, cluster_plan


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
                    if result[i] == k and label[i] == j:
                        count += 1
                p_k.append(count)
            temp_t = max(p_k)
            t.append(temp_t)

        pur = sum(t) / total_num

        return pur

    def finalresult(self, list1, min_dis_sample, x, nc):
        if nc == 1:
            distance = np.linalg.norm(self.whole_data - x[0, :],  axis=1) ** 2
            f = distance.sum()
        else:
            f = min_dis_sample.sum()

        num_labels = len(np.unique(list1))
        viol, vcan, vmust, pur, NML = 0, 0, 0, 0, 0

        # Calculation of Violations
        mcval = 0
        mmval = 0
        for i in range(0, self.ncan_whole):
            index1, index2 = self.can_whole[i][0], self.can_whole[i][1]
            if list1[index1] != list1[index2]:
                mcval += 1

        for j in range(0, self.nmust_whole):
            index1, index2 = self.must_whole[j][0], self.must_whole[j][1]
            if list1[index1] == list1[index2]:
                mmval += 1

        mt = self.ncan_whole + self.ncan_whole
        mt1 = mcval + mmval
        if mt > 0:
            viol = 1.0e2 * float(mt1)/float(mt)
        if mt == 0:
            viol = 0.00

        if self.ncan_whole > 0:
            vcan = 1.0e2 * float(mcval)/float(self.ncan)
        if self.ncan_whole == 0:
            vcan = 0.00

        if self.nmust_whole > 0:
            vmust = 1.0e2 * float(mmval)/float(self.nmust)
        if self.nmust_whole == 0:
            vmust = 0.00

        # Purity
        pur = 1.0e02 * self.purity(list1, self.whole_label)

        # dbi
        if num_labels == 1:
            dbi = 0
        else:
            dbi = davies_bouldin_score(self.whole_data, list1)

        # silhouette
        if num_labels == 1:
            sil_avg = 0
            sil_pos = 0
        else:
            sil_avg = silhouette_score(self.whole_data, list1, metric='euclidean')
            sil_samples = silhouette_samples(self.whole_data, list1, metric='euclidean')
            s = [num_pos for num_pos in sil_samples if num_pos > 0]
            sil_pos = len(s)/self.nrecord
            #sil_pos = len(s)

        return f, viol, vcan, vmust, pur, dbi, sil_avg, sil_pos, NML,

    def Select_constrints(self):
        Hard_can, Easy_can = np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int)
        Hard_must, Easy_must = np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int)
        for i in range(0, self.ncan):
            Avg_dist1 = np.array(
                [np.linalg.norm(self.data[self.CL_index[i, 0]] - tmp) ** 2 for tmp in self.data]).mean()
            Avg_dist2 = np.array(
                [np.linalg.norm(self.data[self.CL_index[i, 1]] - tmp) ** 2 for tmp in self.data]).mean()
            dist_can = np.array([np.linalg.norm(self.data[self.CL_index[i, 0]] - self.data[self.CL_index[i, 1]]) ** 2])
            if dist_can >= Avg_dist1 and dist_can >= Avg_dist2:
                Easy_can = np.vstack((Easy_can, self.CL_index[i, :]))
            else:
                Hard_can = np.vstack([Hard_can, self.CL_index[i, :]])

        for i in range(0, self.nmust):
            Avg_dist1 = np.array(
                [np.linalg.norm(self.data[self.ML_index[i, 0]] - tmp) ** 2 for tmp in self.data]).mean()
            Avg_dist2 = np.array(
                [np.linalg.norm(self.data[self.ML_index[i, 1]] - tmp) ** 2 for tmp in self.data]).mean()
            dist_must = np.array([np.linalg.norm(self.data[self.ML_index[i, 0]] - self.data[self.ML_index[i, 1]]) ** 2])
            if dist_must < Avg_dist1 and dist_must < Avg_dist2:
                Easy_must = np.vstack((Easy_must, self.ML_index[i, :]))
            else:
                Hard_must = np.vstack((Hard_must, self.ML_index[i, :]))

        return Hard_can, Easy_can, Hard_must, Easy_must

    def make_constraints(self, label):
        rng1 = np.random.default_rng(38)
        rng2 = np.random.default_rng(40)
        num_const = math.floor(len(label) * 0.1)
        index = np.array([sample for sample in range(0, len(label))])
        const_list = np.zeros((num_const, 3), dtype=int)
        label_const_list = np.zeros((num_const, 2), dtype=int)
        num = 0

        for i in range(0, 1000*num_const):
            if num >= num_const:
                break
            index_1 = rng1.choice(index)
            index_2 = rng2.choice(index)
            if index_1 != index_2:
                if label[index_1] == label[index_2]:
                    if num == 0:
                        const_list[num, 0], const_list[num, 1], const_list[num, 2] = index_1, index_2, 1
                        label_const_list[num, 0], label_const_list[num, 1] = label[index_1], label[index_2]
                        num += 1
                    if num > 0:
                        flag = True
                        while flag:
                            for j in range(0, num):
                                if index_1 in const_list[j, :] and index_2 in const_list[j, :]:
                                    flag = False
                                    break
                            if flag:
                                const_list[num, 0], const_list[num, 1], const_list[num, 2] = index_1, index_2, 1
                                label_const_list[num, 0], label_const_list[num, 1] = label[index_1], label[index_2]
                                num += 1
                else:
                    if num == 0:
                        const_list[num, 0], const_list[num, 1], const_list[num, 2] = index_1, index_2, -1
                        label_const_list[num, 0], label_const_list[num, 1] = label[index_1], label[index_2]
                        num += 1
                    if num > 0:
                        flag = True
                        while flag:
                            for j in range(0, num):
                                if index_1 in const_list[j, :] and index_2 in const_list[j, :]:
                                    flag = False
                                    break
                            if flag:
                                const_list[num, 0], const_list[num, 1], const_list[num, 2] = index_1, index_2, -1
                                label_const_list[num, 0], label_const_list[num, 1] = label[index_1], label[index_2]
                                num += 1
        ml_index, cl_index = np.where(const_list[:,-1] == 1), np.where(const_list[:,-1] == -1)
        ml, cl = const_list[ml_index[0], 0:2], const_list[cl_index[0], 0:2]

        return ml, cl

    def update_const(self, cannot_link, must_link, eps, Clu_Der_Points_data):
        Pre_can = np.empty_like(cannot_link)
        Pre_must = np.empty_like(must_link)
        n1, n2 = 0, 0
        for i in range(np.size(cannot_link, 0)):
            dist1 = np.array(
                [np.linalg.norm(self.data[cannot_link[i, 0]] - sample1) ** 2 for sample1 in Clu_Der_Points_data])
            #p1 = np.where(dist1 <= eps)
            p1 = dist1.argmin()

            dist2 = np.array(
                [np.linalg.norm(self.data[cannot_link[i, 1]] - sample2) ** 2 for sample2 in Clu_Der_Points_data])
            #p2 = np.where(dist2 <= eps)
            p2 = dist2.argmin()

            if p1 != p2:
                Pre_can[n1, 0] = p1
                Pre_can[n1, 1] = p2
                n1 = n1 + 1

        for i in range(np.size(must_link, 0)):
            dist3 = np.array(
                [np.linalg.norm(self.data[must_link[i, 0]] - sample3) ** 2 for sample3 in Clu_Der_Points_data])
            # p3 = np.where(dist3 <= eps)
            p3 = dist3.argmin()
            Pre_must[i, 0] = p3

            dist4 = np.array(
                [np.linalg.norm(self.data[must_link[i, 1]] - sample4) ** 2 for sample4 in Clu_Der_Points_data])
            # p4 = np.where(dist4 <= eps)
            p4 = dist4.argmin()
            Pre_must[i, 1] = p4

        return Pre_can, Pre_must

    def main(self):
        for dataset_name in datasets:
            self.input(dataset_name)
            self.num_DataBlock = np.zeros((self.num_blocks), dtype=int)  # number of points of each blocks
            pre_data = np.random.rand(0, self.num_feature)
            w2 = np.random.rand(pre_data.shape[0])
            pre_can = np.zeros((0, 2))
            pre_must = np.zeros((0, 2))
            Centers = []
            whole_centers = []

            for kblocks in range(1, self.num_blocks + 1):
                print(f'start computing {kblocks} block, {self.num_blocks} blocks in total')
                self._update_data(kblocks, b=pre_data, w2=w2)
                self.num_DataBlock[kblocks-1] = np.size(self.Org_data, 0)
                self._thresh_choose_new()
                self.ML_index, self.CL_index = self.make_constraints(self.Org_label)

                if kblocks > 1:
                    self.ML_index = np.vstack((self.ML_index,  pre_must + self.nrecord1))
                    self.CL_index = np.vstack((self.CL_index, pre_can + self.nrecord1))

                self.nmust, self.ncan = self.ML_index.shape[0], self.CL_index.shape[0]

                Hard_can, Easy_can, Hard_must, Easy_must = self.Select_constrints()

                xsolution = Centers
                SemisupervisedAlgo = IncSemisupervisedClusteringAlgo(self.data, self.Org_label, dataset_name,
                                                self.num_clusters, self.ML_index, self.CL_index, self.w, xsolution, kblocks)

                Centers, min_dis_sample, cluster_plan_nk, cluster_plan_index, cluster_radius = SemisupervisedAlgo.run()

                whole_centers.append(Centers)
                if kblocks < self.num_blocks:
                    eps, epslion_netPoints, contourPoints, corePoints, w_eps, w_contour, w_core = Clu_Der_Points(Centers,
                                                                    self.data, min_dis_sample, cluster_plan_nk,
                                                                cluster_plan_index, cluster_radius, self.w, self.thresh,
                                                                                                                 kblocks)

                    Clu_Der_Points_data = np.concatenate((epslion_netPoints, contourPoints, corePoints), axis=0)
                    w_Clu_Der_Points = np.concatenate((w_eps, w_contour, w_core), axis=0)
                    pre_can, pre_must = self.update_const(Easy_can, Easy_must, eps, Clu_Der_Points_data)

                    pre_data = Clu_Der_Points_data
                    w2 = w_Clu_Der_Points

                else:
                    cluster_plan_index_whole, min_dis_sample_whole = self.WholeData(Centers,self.num_clusters)
                    f, viol, vcan, vmust, pur, dbi, sil_avg, sil_pos, NML = self.finalresult(cluster_plan_index_whole,
                                                                    min_dis_sample_whole, Centers, self.num_clusters)

if __name__ == '__main__':
    a = SemisupervisedClusteringBigDataAlgo()
    a.main()


