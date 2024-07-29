import pandas as pd
import numpy as np
import math
import copy
from Functions import calculate_distances
from sklearn import metrics
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, normalized_mutual_info_score

def Clu_Der_Points(x, dataset, min_dis_sample, cluster_plan_nk, cluster_plan_index, cluster_radius, w, thresh,nblocks):
    nc = np.size(x, 0)
    nrecord = np.size(dataset, 0)
    num_feature = np.size(dataset, 1)

    with open('./results/test.txt', 'a') as file:
        file.write(f'deter_input_x:{x}\r\n')

    # Contour points
    thresh1 = 2.0e-1 * thresh
    l2 = np.zeros((nc, 1))
    l1 = []  # contour points of all data set
    l11 = []  # list: contour points of each cluster
    list_boundary = []
    for i, cluster in enumerate(cluster_plan_nk):
        cluster = np.array(cluster)
        if len(cluster) == 0:
            continue
        else:
            l1_1 = []  # boundary points in each cluster
            l = []
            n2 = math.ceil(thresh1 * len(cluster))
            if n2 == 0:
                n2 = 1
            l2[i] = n2
            outside_points_index = np.where(min_dis_sample[cluster] > cluster_radius[i])
            outside_points = cluster[outside_points_index[0]]
            max_index = np.argmax(min_dis_sample[outside_points])
            j2 = outside_points[max_index]
            l1.append(j2)
            list_boundary.append(j2)
            l1_1.append(j2)
            l.append(dataset[j2, :])
            while len(l1_1) <= n2:
                dis_min = np.zeros((len(outside_points), 1))
                for j, sample in enumerate(outside_points):
                    if sample in l1_1:
                        dis_min[j] = 0
                        continue
                    else:
                        distance_given_points = np.array(
                            [np.linalg.norm(dataset[sample, :] - tmp) ** 2 for tmp in l])
                        dis_min[j] = min(distance_given_points)
                kmax_index, d3 = dis_min.argmax(), dis_min.max()
                kmax = outside_points[kmax_index]
                point_New = dataset[kmax, :]
                d = d3  # the minimum distance between the new point and the given points
                if d <= cluster_radius[i]:  # the stop criteria
                    break
                else:
                    l1_1.append(kmax)
                    l1.append(kmax)
                    l.append(dataset[kmax, :])
                    list_boundary.append(kmax)
            l11.append(l1_1)  # list: 每一个元素为每一个簇的boundary points

    contour_points_index = list_boundary
    w_contour_points = w[contour_points_index]
    contour_points = dataset[contour_points_index]

    print('num_contour:',np.size(contour_points_index))
    # contour_points_index = list_boundary
    # contour_points_index = [i for i in list_boundary if
    #                         i not in set(epslion_netPoints_INDEX)]  # remove epslionnet-points
    # w_contour_points_index = np.ones((len(contour_points_index), 1))

    # core points
    #distances = calculate_distances(dataset)
    ##SI_samples = silhouette_samples(distances, cluster_plan_index, metric="precomputed")

    #SI_samples = silhouette_samples(dataset, cluster_plan_index)

    thresh2 = 2.0e-1 * thresh
    SI_samples = np.zeros((np.size(dataset, 0),))
    sa = np.zeros((nc,))
    for i in range(0, np.size(dataset, 0)):
        k1 = cluster_plan_index[i]
        for k in range(0, nc):
            sa[k] = 0.0e+00
        for j in range(np.size(dataset, 0)):
            if j != i:
                d1 = np.linalg.norm((dataset[i, :] - dataset[j, :]))
                #d1 = np.sqrt(d1)
                k2 = cluster_plan_index[j]
                sa[k2] = sa[k2] + d1
        da1 = sa[k1]/float(len(cluster_plan_nk[k1]))
        da2 = 1.0e+26
        for k in range(nc):
            if k != k1:
                da3 = sa[k]/float(len(cluster_plan_nk[k]))
                da2 = min(da2, da3)
        SI_samples[i] = (da2-da1)/max(da1, da2)

    SI_avg, SI_max = SI_samples.mean(), SI_samples.max()
    bd1 = SI_avg
    bd2 = SI_max
    simid = (bd1 + bd2) / 2
    #print('SI_max,SI_avg', SI_avg, SI_max)
    while True:
        core_id = np.where(SI_samples > simid)[0]
        list_corepoints_index = [x for x in core_id if x not in contour_points_index]  # remove boundary-points
        #print('num_corepoints', len(list_corepoints_index))
        ratio_core = len(list_corepoints_index) / np.size(dataset, 0)
        #('ratio,simid', ratio_core, simid)
        if ratio_core > thresh2:
            bd1 = simid
            simid = (bd1 + bd2) / 2
        elif ratio_core < 0.5 * thresh2:
            bd2 = simid
            simid = (bd1 + bd2) / 2
        else:
            break

    w_corepoints = w[list_corepoints_index]
    core_points = dataset[list_corepoints_index]

    print('num_corepoints:', np.size(list_corepoints_index))

    # epsilon_net points
    c1 = 1.0e-9
    f = np.dot(min_dis_sample, w)

    union = set.union(set(list_corepoints_index), set(contour_points_index))  # remove contour and core points
    union = list(union)
    w2 = np.zeros((nrecord,))
    b = np.zeros((nrecord, num_feature))
    n2, b[0:nc, :], iter = nc, x, 0

    while True:
        # epsilon-net points
        w2[0:nc] = np.ones((nc,))
        eps = c1 * f
        l1 = np.zeros((nrecord,), dtype=int)
        l1[union] = 1
        l3, w1 = [], []   # epsilon-net points 的下标和权重
        for i in range(0, nrecord):
            if min_dis_sample[i] <= eps and i not in union:
                w2[cluster_plan_index[i]] += w[i]  # update weights of the center point
                l1[i] = 1

        for i, cluster_id in enumerate(cluster_plan_nk):
            cluster_id = np.array(cluster_id)
            while True:
                l1_index = np.where(l1 == 0)
                common_index = np.array(list(set(l1_index[0]) & set(cluster_id)))
                if common_index.size == 0:
                    break
                else:
                    dataset_remain = dataset[list(common_index)]
                    max_index = np.argmax(min_dis_sample[common_index])  # 这个簇中距离中心最远的点
                    id_sample = common_index[max_index]
                    sample_point = dataset[id_sample]
                    distance = [np.linalg.norm(sample_point - tmp) ** 2 for tmp in dataset_remain]
                    id_of_samples = np.argwhere(np.array(distance) <= eps)
                    id_of_dataset_samples = common_index[id_of_samples.flatten()]
                    l1[id_of_dataset_samples] = 1
                    id_sample = copy.copy(id_sample)
                    l3.append(id_sample)
                    w1.append(w[id_of_dataset_samples].sum())

        n4 = len(l3)
        ratio = float(n4+nc) / float(np.size(dataset, 0))
        if ratio > 0.6*thresh:
            iter = iter + 1
            if iter > 30:
                break
            else:
                c1 = 2.0e0 * c1
        else:
            break

    epslion_netPoints_INDEX = l3[0:n4]
    b[nc:n4+nc] = dataset[np.array(epslion_netPoints_INDEX)]
    w2[nc:n4+nc] = w1[0:n4]
    epslion_netPoints = b[0:n4+nc]
    w2 = w2[0:n4+nc]
    w_eps_netPoints = np.array([w2]).T

    # cluster determine points
    #Clu_Der_Points_index = epslion_netPoints_INDEX + contour_points_index + list_corepoints_index
    #w_Clu_Der_Points_index = np.concatenate((w_eps_netPoints, w_contour_points_index, w_corepoints_index), axis=1)

    if nblocks == 2:
        print('stop')

    print('deterpoints',np.size(contour_points_index),np.size(list_corepoints_index),n4+nc)
    return eps, epslion_netPoints, contour_points, core_points, w_eps_netPoints, w_contour_points, w_corepoints
