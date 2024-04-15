import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from scipy import spatial
from numpy import *
import torch.nn as nn
import torchvision
from scipy.stats import entropy

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


# sign表示是否被选出标记，tmp表示提取的特征，label表示真实标签, dis表示样本到中心的距离
# P表示概率矩阵,
class LR(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LR, self).__init__()
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm1d(in_dim)
        # self.D = nn.Dropout(0.4)
        self.linear = nn.Linear(in_dim,out_dim)
    def forward(self,x):
        x = self.relu(x)
        x = self.BN(x)
        # x = self.D(x)
        x = self.linear(x)
        return x

def aug_data(image,transform):
    aug_data = []
    for i in range(len(image)):
        a_image = image[i]
        a_image = np.transpose(a_image, (1,2,0))
        img = Image.fromarray(np.uint8(a_image))
        aug_img = transform(img)
        aug_img = np.array(aug_img)
        aug_data.append(aug_img)
    aug_data = np.array(aug_data)
    aug_data = torch.from_numpy(aug_data)
    return aug_data
def loss_supervised(tmp, one_hot_label, centers, prelabel, gama, unique):
    # 类内距离
    m, n = one_hot_label.shape
    loss1 = torch.empty((len(prelabel))).to(device)
    for i in range(len(prelabel)):
        # loss1表示kmeans损失
        loss1[i] = torch.norm(tmp[i]-centers[unique == prelabel[i]]) * torch.norm(tmp[i]-centers[unique == prelabel[i]])
        # loss1[i] = torch.norm(tmp1[i] - centers[prelabel[i]]) * torch.norm(
        #     tmp1[i] - centers[prelabel[i]])
    loss1 = loss1 / len(prelabel)
    # loss2 = torch.nn.CrossEntropyLoss(tmp,prelabel)
    # one_hot_label = 0.9 * np.array(one_hot_label.cpu()) + 0.1 * np.ones((m, n))
    # one_hot_label = torch.from_numpy(one_hot_label).to(device)
    # 类间距离
    log_prob = torch.nn.functional.log_softmax(tmp, dim=1)
    loss2 = log_prob * one_hot_label
    loss2 = torch.sum(loss2, 1) / len(prelabel)

    loss =  gama * loss1 - loss2

    return torch.sum(loss)

def loss_supervised1(tmp, one_hot_label, P, centers, prelabel, gama):
    # 类内距离
    loss1 = torch.empty((len(prelabel))).to(device)
    for i in range(len(prelabel)):
        # loss1表示kmeans损失
        loss1[i] = torch.norm(tmp[i]-centers[prelabel[i]]) * torch.norm(tmp[i]-centers[prelabel[i]])
    loss1 = loss1 / len(prelabel)

    # 类间距离
    log_prob = torch.nn.functional.log_softmax(P, dim=1)
    loss2 = log_prob * one_hot_label
    # loss3 = loss3.detach().numpy()
    loss2 = torch.sum(loss2, 1) / len(prelabel)

    loss = gama * loss1 - loss2

    return torch.sum(loss)

def Weight(prelabel_dis, n, ratio):
    x=0
    weight = np.empty(n)
    Ind = np.arange(n)  # 索引，也就是序号
    Ind = Ind.reshape(-1, 1)
    prelabel_dis_ind = np.hstack((prelabel_dis, Ind))
    index1 = np.lexsort((prelabel_dis_ind[:, 1], prelabel_dis_ind[:, 0]))
    prelabel_dis_ind = prelabel_dis_ind[index1]

    dis = prelabel_dis_ind[:, 1]
    prelabel = prelabel_dis_ind[:, 0]
    prelabel = prelabel.astype(int)
    ind = prelabel_dis_ind[:, 2]
    ind = ind.astype(int)
    U = np.unique(prelabel)
    num_cluster = len(U)
    index = np.empty(num_cluster+1)
    index_of_lam = np.empty(num_cluster)
    index[0] = 0
    for i in range(num_cluster):
        index_of_lam[i] = np.sum(prelabel == i) * ratio
        index[i + 1] = index[i] + np.sum(prelabel == i)
    index_of_lam = index_of_lam.astype(int)
    index = index.astype(int)
    for i in range(num_cluster):
        dis0 = dis[index[i]:index[i+1]]
        lam = dis0[index_of_lam[i]]
        weight0 = np.where(dis0 <= lam, -(dis0 / lam) + 1, 0)
        cnt_array = np.where(weight0, 0, 1)
        x = x+np.sum(cnt_array)
        weight[ind[index[i]:index[i+1]]] = weight0
    return weight

# 按照比例选择
def select0(label_and_dis, n, number_select, lim):          # weight:自步权重； label_and_dis:真实标签*预测标签*以及每个样本到中心点的距离；n:样本数
    sign = []
    prelabel = label_and_dis[:, 1]  # 取出预测序号列
    prelabel = prelabel.astype(int)
    Ind = np.arange(n)  # 索引，也就是序号
    Ind = Ind.reshape(-1, 1)
    new_l_and_d = np.hstack((label_and_dis, Ind))
    # new_l_and_d = new_l_and_d.transpose()                  # label_and_dis:真实标签**以及每个样本到中预测标签心点的距离*序号
    index1 = np.lexsort((new_l_and_d[:, 2], new_l_and_d[:, 1]))
    new_l_and_d = new_l_and_d[index1]                        # 得到有序排列的数组

    # 对有序数组分割
    new_l_and_d2 = new_l_and_d[:, 1]  # 取出预测序号列
    new_l_and_d2 = new_l_and_d2.astype(int)
    index = new_l_and_d[:, 3]  # 取出索引号
    index = index.astype(int)
    U = np.unique(new_l_and_d2)
    U1 = np.delete(U, [len(U)-1])  # 去掉最后一个元素
    num = len(U)
    index_of_index = np.empty(num)  # 创建不同元素的起始序列号
    num_kind = np.empty(num)  # 每个种类的元素个数
    num_kind[0] = 1
    index_of_index[0] = 0
    for i in U1:
        # num_kind[i] = number_select / num
        num_kind[i+1] = 1
        index_of_index[i+1] = index_of_index[i] + np.sum(new_l_and_d2 == i)

    index_of_index = index_of_index.astype(int)
    index_of_index = np.append(index_of_index, len(new_l_and_d2))
    index_of_index1 = np.delete(index_of_index, [0, num])  # 去掉第一个元素
    # num_kind = number_select / num
    num_kind = num_kind.astype(int)

    # for i in range(len(index_of_index1)):
    #     ind = index[index_of_index[i]:index_of_index[i + 1]]
    #     r = 0 + lim
    #     j = 1
    #     while j <= num_kind[i] and r < len(ind):
    #         s_label = knn(tmp[ind[r]], tmp, prelabel, k+1)
    #         if len(np.unique(s_label)) != 1:
    #             sign.append(ind[r])
    #             j = j + 1
    #         r = r + 1

    ind = index[0:index_of_index[1]]
    # c, ind_index, b_ind = np.intersect1d(ind, sign_all, return_indices=True)
    # ind = np.delete(ind, ind_index, 0)
    sign = ind[0:num_kind[0]]
    j = 2
    for i in index_of_index1:
        ind = index[i:index_of_index[j]]
        # c, ind_index, b_ind = np.intersect1d(ind, sign_all, return_indices=True)
        # ind = np.delete(ind, ind_index, 0)
        sign1 = ind[0:num_kind[j - 1]]
        sign = np.concatenate((sign, sign1), axis=0)
        j += 1
    return sign

def select(label_and_dis, n, number_select, sign_all, lim, tmp, k):
    prelabel = label_and_dis[:, 1]  # 取出预测序号列
    prelabel = prelabel.astype(int)
    sign = []
    Ind = np.arange(n)  # 索引，也就是序号
    Ind = Ind.reshape(-1, 1)
    new_l_and_d = np.hstack((label_and_dis, Ind))  # label_and_dis:真实标签*预测标签*以及每个样本到中预测标签心点的距离*序号
    index1 = np.lexsort((new_l_and_d[:, 2], new_l_and_d[:, 1]))
    new_l_and_d = new_l_and_d[index1]  # 得到有序排列的数组

    # 对有序数组分割
    new_l_and_d2 = new_l_and_d[:, 1]  # 取出预测序号列
    new_l_and_d2 = new_l_and_d2.astype(int)

    index = new_l_and_d[:, 3]  # 取出索引号
    index = index.astype(int)
    if len(sign_all) != 0:
        c, ind_index, b_ind = np.intersect1d(index, sign_all, return_indices=True)
        index = np.delete(index, ind_index, 0)
        c, ind_index, b_ind = np.intersect1d(new_l_and_d2, sign_all, return_indices=True)
        new_l_and_d2 = np.delete(new_l_and_d2, ind_index, 0)
    U = np.unique(new_l_and_d2)
    U1 = np.delete(U, [0, 0])  # 去掉第一个元素
    num = len(U)

    index_of_index = np.empty(num)  # 创建不同元素的起始序列号
    number = int(number_select / num)

    index_of_index[0] = 0
    for i in U1:
        index_of_index[i] = index_of_index[i - 1] + np.sum(new_l_and_d2 == i)

    index_of_index = index_of_index.astype(int)
    index_of_index = np.append(index_of_index, len(new_l_and_d2))
    index_of_index1 = np.delete(index_of_index, [num])  # 去掉最后一个元素

    for i in range(len(index_of_index1)):
        ind = index[index_of_index[i]:index_of_index[i+1]]
        r = 0 + lim
        j = 1
        while j <= number:
            if knn(tmp[ind[r]], tmp, prelabel, k+1) == False:
                sign.append(ind[r])
                j = j+1
            r = r+1
    sign = np.array(sign)
    return sign

# 按照比例选择
def select1(label_and_dis, n, number_select, sign_all, lim):
    sign = []
    prelabel = label_and_dis[:, 1]  # 取出预测序号列
    prelabel = prelabel.astype(int)
    Ind = np.arange(n)  # 索引，也就是序号
    Ind = Ind.reshape(-1, 1)
    new_l_and_d = np.hstack((label_and_dis, Ind))
    # new_l_and_d = new_l_and_d.transpose()                  # label_and_dis:真实标签**以及每个样本到中预测标签心点的距离*序号
    index1 = np.lexsort((-new_l_and_d[:, 2], new_l_and_d[:, 1]))
    new_l_and_d = new_l_and_d[index1]  # 得到有序排列的数组

    # 对有序数组分割
    new_l_and_d2 = new_l_and_d[:, 1]  # 取出预测序号列
    new_l_and_d2 = new_l_and_d2.astype(int)
    index = new_l_and_d[:, 3]  # 取出索引号
    index = index.astype(int)
    U = np.unique(new_l_and_d2)
    U1 = np.delete(U, [0, 0])  # 去掉第一个元素
    num = len(U)

    index_of_index = np.empty(num)  # 创建不同元素的起始序列号
    num_kind = np.empty(num)  # 每个种类的元素个数
    num_kind[0] = number_select / num
    index_of_index[0] = 0
    for i in U1:
        # num_kind[i] = number_select / num
        num_kind[i] = 1
        index_of_index[i] = index_of_index[i - 1] + np.sum(new_l_and_d2 == i)

    index_of_index = index_of_index.astype(int)
    index_of_index = np.append(index_of_index, len(new_l_and_d2))
    index_of_index1 = np.delete(index_of_index, [0, num])  # 去掉第一个以及最后一个元素
    # num_kind = number_select / num
    num_kind = num_kind.astype(int)

    # for i in range(len(index_of_index1)):
    #     ind = index[index_of_index[i]:index_of_index[i + 1]]
    #     r = 0 + lim
    #     j = 1
    #     while j <= num_kind[i] and r < len(ind):
    #         s_label = knn(tmp[ind[r]], tmp, prelabel, k+1)
    #         if len(np.unique(s_label)) != 1:
    #             sign.append(ind[r])
    #             j = j + 1
    #         r = r + 1

    ind = index[0:index_of_index[1]]
    c, ind_index, b_ind = np.intersect1d(ind, sign_all, return_indices=True)
    ind = np.delete(ind, ind_index, 0)
    sign = ind[lim:lim+num_kind[0]]
    j = 2
    for i in index_of_index1:
        ind = index[i:index_of_index[j]]
        c, ind_index, b_ind = np.intersect1d(ind, sign_all, return_indices=True)
        ind = np.delete(ind, ind_index, 0)
        sign1 = ind[lim:lim+num_kind[j-1]]
        sign = np.concatenate((sign, sign1), axis=0)
        j += 1
    # sign = np.array(sign)
    # sign = sign.astype(int)
    sign_all = np.hstack((sign_all, sign))
    if len(sign) != number_select:
        m = number_select - len(sign)
        c, ind_index, b_ind = np.intersect1d(index, sign_all, return_indices=True)
        ind = np.delete(index, ind_index, 0)
        n = np.random.randint(0, len(ind), size=m)
        sign = np.concatenate((sign, ind[n]), axis=0)

    return sign



def distance_euclidean_scipy(vec1, vec2, distance="euclidean"):
    return spatial.distance.cdist(vec1, vec2, distance)


def fast_kmeans(data, k):
    """k-means聚类算法
    k       - 指定分簇数量
    data      - ndarray(m, n)，m个样本的数据集，每个样本n个属性值
    """

    m, n = data.shape  # m：样本数量，n：每个样本的属性值个数
    result = np.empty(m, dtype=np.int)  # m个样本的聚类结果
    cores = data[np.random.choice(np.arange(m), k, replace=False)]  # 从m个数据样本中不重复地随机选择k个样本作为质心

    while True:  # 迭代计算

        distance = distance_euclidean_scipy(data, cores)
        index_min = np.argmin(distance, axis=1)  # 每个样本距离最近的质心索引序号

        if (index_min == result).all():  # 如果样本聚类没有改变
            return result, cores  # 则返回聚类结果和质心数据

        result[:] = index_min  # 重新分类
        for i in range(k):  # 遍历质心集
            items = data[result == i]  # 找出对应当前质心的子样本集
            cores[i] = np.mean(items, axis=0)  # 以子样本集的均值作为当前质心的位置

# 已标记数据分配不变
def fast_semi_kmeans(label, data, k, sign):
    """k-means聚类算法
    k       - 指定分簇数量
    data     - ndarray(m+p, n)，m个样本的数据集，每个样本n个属性值
    lable        - ndarray(p, n)，m个样本的数据集，每个样本n个属性值
    sign     - ndarray(p,) 所选样本序号
    """

    L_data = data[sign, :]
    U_data = np.delete(data, sign, 0)
    p, n = L_data.shape
    m, n = U_data.shape  # m：样本数量，n：每个样本的属性值个数
    result = np.empty(m+p, dtype=np.int)  # m个样本的聚类结果
    x = np.empty(m + p, dtype=np.int)
    A = np.arange(0, m+p)
    usign = np.delete(A, sign, 0)

    # 初始化聚类中心，从有标签数据中初始化每一类的聚类中心
    centers = np.empty((k, n)) # k个聚类中心
    # cores = unlable[np.random.choice(np.arange(m), k, replace=False)]  # 从m个数据样本中不重复地随机选择k个样本作为质心
    # 获取所有类别
    labels1 = np.unique(label)
    labels = np.arange(k)
    for lab in labels1:
        df = np.where(label == lab)
        df = np.array(df, dtype=int)
        df = df.flatten()
        df = L_data[df, :]
        centers[lab] = np.mean(df, axis=0)
    if len(labels1)!=len(labels):
        d, labels1_ind, labels_ind = np.intersect1d(labels1, labels, return_indices=True)
        label2 = np.delete(labels, labels_ind, 0)
        c = U_data[np.random.choice(np.arange(m), len(label2), replace=False)]
        for i in range(len(label2)):
            centers[label2[i]] = c[i]

    for i in range(len(sign)):
        x[sign[i]] = label[i]
    while True:  # 迭代计算
        distance = distance_euclidean_scipy(U_data, centers)
        index_min = np.argmin(distance, axis=1)  # 每个样本距离最近的质心索引序号
        for i in range(len(usign)):
            x[usign[i]] = index_min[i]

        if (x == result).all():  # 如果样本聚类没有改变
            return result, centers  # 则返回聚类结果和质心数据

        result = x  # 重新分类
        for i in range(k):  # 遍历质心集
            items = data[result == i]  # 找出对应当前质心的子样本集
            centers[i] = np.mean(items, axis=0)  # 以子样本集的均值作为当前质心的位置

# 对于已标记数据也是随机分配
def fast_semi_kmeans1(label, data, k, sign):
    """k-means聚类算法
    k       - 指定分簇数量
    data     - ndarray(m+p, n)，m个样本的数据集，每个样本n个属性值
    lable        - ndarray(p, n)，m个样本的数据集，每个样本n个属性值
    sign     - ndarray(p,) 所选样本序号
    """

    L_data = data[sign, :]
    U_data = np.delete(data, sign, 0)
    p, n = L_data.shape
    m, n = U_data.shape  # m：样本数量，n：每个样本的属性值个数
    result = np.empty(m+p, dtype=np.int)  # m个样本的聚类结果
    x = np.empty(m + p, dtype=np.int)
    A = np.arange(0, m+p)
    usign = np.delete(A, sign, 0)

    # 初始化聚类中心，从有标签数据中初始化每一类的聚类中心
    centers = np.empty((k, n)) # k个聚类中心
    # cores = unlable[np.random.choice(np.arange(m), k, replace=False)]  # 从m个数据样本中不重复地随机选择k个样本作为质心
    # 获取所有类别
    labels1 = np.unique(label)
    labels = np.arange(k)
    for lab in labels1:
        df = np.where(label == lab)
        df = np.array(df, dtype=int)
        df = df.flatten()
        df = L_data[df, :]
        centers[lab] = np.mean(df, axis=0)
    if len(labels1)!=len(labels):
        d, labels1_ind, labels_ind = np.intersect1d(labels1, labels, return_indices=True)
        label2 = np.delete(labels, labels_ind, 0)
        c = U_data[np.random.choice(np.arange(m), len(label2), replace=False)]
        for i in range(len(label2)):
            centers[label2[i]] = c[i]

    while True:  # 迭代计算
        distance = distance_euclidean_scipy(data, centers)
        index_min = np.argmin(distance, axis=1)  # 每个样本距离最近的质心索引序号

        if (index_min == result).all():  # 如果样本聚类没有改变
            return result, centers  # 则返回聚类结果和质心数据

        result[:] = index_min  # 重新分类
        for i in range(k):  # 遍历质心集
            items = data[result == i]  # 找出对应当前质心的子样本集
            centers[i] = np.mean(items, axis=0)  # 以子样本集的均值作为当前质心的位置

def acc(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]



def option1(label_and_dis, n, number_select, sign_all):
    Ind = np.arange(n)  # 索引，也就是序号
    Ind = Ind.reshape(-1, 1)
    new_l_and_d = np.hstack((label_and_dis, Ind))
    if len(sign_all) != 0:
        new_l_and_d = np.delete(new_l_and_d, sign_all, 0)

    index1 = np.lexsort((new_l_and_d[:, 2], new_l_and_d[:, 1]))
    new_l_and_d = new_l_and_d[index1]  # 得到有序排列的数组

    # 对有序数组分割
    new_l_and_d2 = new_l_and_d[:, 1]  # 取出预测序号列
    new_l_and_d2 = new_l_and_d2.astype(int)
    index = new_l_and_d[:, 3]  # 取出索引号
    index = index.astype(int)
    U = np.unique(new_l_and_d2)
    U1 = np.delete(U, [len(U)-1])  # 去掉第一个元素
    num = len(U)

    index_of_index = np.empty(num)  # 创建不同元素的起始序列号
    num_kind = np.empty(num)  # 每个种类的元素个数
    num_kind[0] = number_select
    index_of_index[0] = 0
    for i in U1:
        num_kind[i+1] = np.sum(new_l_and_d2 == i)
        num_kind[i+1] = number_select
        index_of_index[i+1] = index_of_index[i] + np.sum(new_l_and_d2 == i)

    index_of_index = index_of_index.astype(int)
    index_of_index = np.append(index_of_index, len(new_l_and_d2))
    index_of_index1 = np.delete(index_of_index, [0, num])  # 去掉第一个以及最后一个元素
    # num_kind = number_select
    num_kind = num_kind.astype(int)

    ind = index[0:index_of_index[1]]
    sign = ind[0:num_kind[0]]
    j = 2
    for i in index_of_index1:
        ind = index[i:index_of_index[j]]
        # c, ind_index, b_ind = np.intersect1d(ind, sign_all, return_indices=True)
        # ind = np.delete(ind, ind_index, 0)
        sign1 = ind[0:num_kind[j-1]]
        sign = np.concatenate((sign, sign1), axis=0)
        j += 1

    return sign

def knn(input,dataSet,label,k):
    dataSize = dataSet.shape[0]
    ####计算欧式距离
    diff = tile(input,(dataSize,1)) - dataSet
    sqdiff = diff ** 2
    squareDist = sum(sqdiff,axis = 1) # 行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5

    sortedDistIndex = argsort(dist) # argsort()根据元素的值从小到大对元素进行排序，返回下标
    k_label = label[sortedDistIndex[0:k]]
    return k_label
    # if len(np.unique(k_label)) == 1:
    #     return True
    # else:
    #     return False

def knn_dis(input,dataSet):
    dataSize = dataSet.shape[0]
    ####计算欧式距离
    diff = tile(input,(dataSize,1)) - dataSet
    sqdiff = diff ** 2
    squareDist = sum(sqdiff,axis = 1) # 行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5

    return min(dist)

def cluster_desity(tmp):
    s = 0
    for i in range(len(tmp)):
        x = tmp[i]
        y = np.delete(tmp, i, axis=0)
        s = s + knn_dis(x,y)
    return s


# 每类选一个, 距离从小到大排序
# def select0(label_and_dis, n, number_select, lim):          # weight:自步权重； label_and_dis:真实标签*预测标签*以及每个样本到中心点的距离；n:样本数
#     Ind = np.arange(n)  # 索引，也就是序号
#     Ind = Ind.reshape(-1, 1)
#     new_l_and_d = np.hstack((label_and_dis, Ind))
#     # new_l_and_d = new_l_and_d.transpose()                  # label_and_dis:真实标签**以及每个样本到中预测标签心点的距离*序号
#     index1 = np.lexsort((new_l_and_d[:, 2], new_l_and_d[:, 1]))
#     new_l_and_d = new_l_and_d[index1]                        # 得到有序排列的数组
#
#     # 对有序数组分割
#     new_l_and_d2 = new_l_and_d[:, 1]  # 取出预测序号列
#     new_l_and_d2 = new_l_and_d2.astype(int)
#     index = new_l_and_d[:, 3]  # 取出索引号
#     index = index.astype(int)
#     U = np.unique(new_l_and_d2)
#     U1 = np.delete(U, [0, 0])  # 去掉第一个元素
#     num = len(U)
#     index_of_index = np.empty(num)  # 创建不同元素的起始序列号
#     num_kind = np.empty(num)  # 每个种类的元素个数
#     num_kind[0] = 1
#     index_of_index[0] = 0
#     for i in U1:
#         num_kind[i] = 1
#         index_of_index[i] = index_of_index[i - 1] + np.sum(new_l_and_d2 == i)
#
#     index_of_index = index_of_index.astype(int)
#     index_of_index1 = np.delete(index_of_index, [0, 0])  # 去掉第一个元素
#     # num_kind = num_kind / n * number_select
#     num_kind = num_kind.astype(int)
#
#     sign = index[lim:lim+num_kind[0]]
#     a = 1
#     for i in index_of_index1:
#         i = i+lim
#         sign = np.concatenate((sign, index[i:i + num_kind[a]]), axis=0)
#         a += 1
#
#     if len(sign) != number_select:
#         m = number_select - len(sign)
#         c, ind_index, b_ind = np.intersect1d(index, sign, return_indices=True)
#         ind = np.delete(index, ind_index, 0)
#         n = np.random.randint(0, len(ind), size=m)
#         sign = np.concatenate((sign, ind[n]), axis=0)
#     return sign

# 每类选一个 距离从大到小排序
# def select(label_and_dis, n, number_select, sign_all, lim):
#     Ind = np.arange(n)  # 索引，也就是序号
#     Ind = Ind.reshape(-1, 1)
#     new_l_and_d = np.hstack((label_and_dis, Ind))
#     # new_l_and_d = new_l_and_d.transpose()                  # label_and_dis:真实标签**以及每个样本到中预测标签心点的距离*序号
#     index1 = np.lexsort((-new_l_and_d[:, 2], new_l_and_d[:, 1]))
#     new_l_and_d = new_l_and_d[index1]  # 得到有序排列的数组
#
#     # 对有序数组分割
#     new_l_and_d2 = new_l_and_d[:, 1]  # 取出预测序号列
#     new_l_and_d2 = new_l_and_d2.astype(int)
#     index = new_l_and_d[:, 3]  # 取出索引号
#     index = index.astype(int)
#     U = np.unique(new_l_and_d2)
#     U1 = np.delete(U, [0, 0])  # 去掉第一个元素
#     num = len(U)
#
#     index_of_index = np.empty(num)  # 创建不同元素的起始序列号
#     num_kind = np.empty(num)  # 每个种类的元素个数
#     num_kind[0] = 1
#     index_of_index[0] = 0
#     for i in U1:
#         num_kind[i] = 1
#         index_of_index[i] = index_of_index[i - 1] + np.sum(new_l_and_d2 == i)
#
#     index_of_index = index_of_index.astype(int)
#     index_of_index = np.append(index_of_index, len(new_l_and_d2))
#     index_of_index1 = np.delete(index_of_index, [0, num])  # 去掉第一个以及最后一个元素
#     # num_kind = num_kind / n * number_select
#     num_kind = num_kind.astype(int)
#
#     ind = index[0:index_of_index[1]]
#     c, ind_index, b_ind = np.intersect1d(ind, sign_all, return_indices=True)
#     ind = np.delete(ind, ind_index, 0)
#     sign = ind[lim:lim+num_kind[0]]
#     j = 2
#     for i in index_of_index1:
#         ind = index[i:index_of_index[j]]
#         c, ind_index, b_ind = np.intersect1d(ind, sign_all, return_indices=True)
#         ind = np.delete(ind, ind_index, 0)
#         sign1 = ind[lim:lim+num_kind[j-1]]
#         sign = np.concatenate((sign, sign1), axis=0)
#         j += 1
#
#     sign_all = np.hstack((sign_all, sign))
#     if len(sign) != number_select:
#         m = number_select - len(sign)
#         c, ind_index, b_ind = np.intersect1d(index, sign_all, return_indices=True)
#         ind = np.delete(index, ind_index, 0)
#         n = np.random.randint(0, len(ind), size=m)
#         sign = np.concatenate((sign, ind[n]), axis=0)
#
#     return sign