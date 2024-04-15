import argparse
import torch
import torchvision
import numpy as np
import timm
from torch.nn import functional as F
from util import select, loss_supervised, select0, fast_kmeans, fast_semi_kmeans, distance_euclidean_scipy, \
    acc, option1, select1, LR, aug_data, cluster_desity, Weight
import torch.optim as optim
from sklearn import metrics
from time import time
from torch.optim.swa_utils import AveragedModel, SWALR
from datasets import load_STL_10
from sklearn.preprocessing import MinMaxScaler
import itertools
from timm.data.mixup import Mixup
from train_contrastive import pretext_model, aug_transform
from sklearn.cluster import KMeans
from numpy import *


batch_size = 700  # 一次训练的样本数目
batch_size1 = 128

def main(args):
    start1 = time()
    k = 8
    repeat = 1
    beta = 1e-5
    gama = 0
    bound = 200
    pseudo_number = 200
    ratio = 0.1
    # Round = int(input("please input the number of training round："))
    Round = 40
    roundx = 3
    # L_number = int(input("please input the number of labeled instance："))
    EPOCH = 7
    EPOCH1 = 300
    out_dim = 200
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # model = pretext_model()
    model = timm.create_model('convnext_tiny', pretrained=True, num_classes=out_dim).to(device)
    model.load_state_dict(torch.load('./STL10_result/simclr.pt'))
    model.eval()
    # num_f = model.get_classifier().in_features
    # torch.save(model.state_dict(), './Cifar10_result/simclr.pt')
    tmp, label, image = load_STL_10(model)
    # np.save('./STL10_result/STL10_image.npy', image)
    np.save('./STL10_result/label.npy', label)
    np.save('./STL10_result/tmp.npy', tmp)
    n = len(label)  # 样本数
    L_number = 400   # int(n * 0.02)
    U = np.unique(label)
    num_cluster = len(U)  # 样本类数目
    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(tmp)
    pre_label = kmeans.labels_
    centers = kmeans.cluster_centers_
    # pre_label, centers = fast_kmeans(tmp, num_cluster)
    print("fast_Kmeans准确率：", acc(label, pre_label))
    model1 = LR(out_dim, num_cluster).to(device)
    # 定义网络优化器
    optimizer = optim.Adam(itertools.chain(model.parameters(), model1.parameters()),
                           lr=2e-5,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=0,
                           amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 128)

    # 每一类标签都改为onehot
    one_hotlabel1 = torch.LongTensor(U)  # 把所有label变成long型
    one_hotlabel1 = F.one_hot(one_hotlabel1, num_classes=num_cluster)  # 把label变成one-hot
    one_hot_label1 = one_hotlabel1.numpy()

    # 所有标签都改为onehot
    one_hotlabel = torch.LongTensor(label)  # 把所有label变成long型
    one_hotlabel = F.one_hot(one_hotlabel, num_classes=num_cluster)  # 把label变成one-hot
    one_hot_label = one_hotlabel.numpy()

    U_loss = []
    S_loss = []
    accuary = []
    ave_U_loss = []
    ave_S_loss = []
    NMI = []
    ARI = []
    sign_all = []

    for round in range(Round):

        print(round)
        start = time()

        dis1 = distance_euclidean_scipy(tmp, centers)  # 每个样本到所有中心点的距离
        # dis1 = dis1 * dis1
        dis = dis1.min(1)  # 每个样本到所属中心点的距离
        dis = dis.reshape(-1, 1)

        # 得到真实标签*预测标签*选择权重
        label_and_dis = np.vstack((label, pre_label))
        label_and_dis = label_and_dis.transpose()
        label_and_dis = np.hstack((label_and_dis, dis))

        # 选择的标签
        # 第一次选样本用select0，之后用select（去掉了以前选过的标签）
        if round == 1:
            sign = select1(label_and_dis, n, int(L_number / Round), sign_all, bound)
            sign_all = sign
        elif round > 1:
            sign = select1(label_and_dis, n, int(L_number / Round), sign_all, bound)
            sign_all = np.hstack((sign_all, sign))
        else:
            center_sign1 = select0(label_and_dis, n, num_cluster, bound)

        # 标签对齐
        if round > 0:
            pseudo_number = int(pseudo_number+10)
            l = label[center_sign1]
            p_signall_label = np.empty(len(sign_all))
            for i in range(len(sign_all)):
                for x in center_sign1:
                    if label[x] == label[sign_all[i]]:
                        p_signall_label[i] = pre_label[x]
            p_signall_label = p_signall_label.astype(int)
            # p_signall_label = label[sign_all]
        p_label1 = np.empty(num_cluster)
        print('number of sign: ', len(sign_all))
        for g in range(roundx):
            start0 = time()
            pseudo_label = []
            dis1 = distance_euclidean_scipy(tmp, centers)  # 每个样本到所有中心点的距离
            # dis1 = dis1 * dis1
            dis = dis1.min(1)  # 每个样本到所属中心点的距离
            dis = dis.reshape(-1, 1)

            # 得到真实标签*预测标签*选择权重
            label_and_dis = np.vstack((label, pre_label))
            label_and_dis = label_and_dis.transpose()
            label_and_dis = np.hstack((label_and_dis, dis))
            p_number = pseudo_number * (g + 1)
            p_sign = option1(label_and_dis, n, p_number, sign_all)
            ACC = []
            density = []
            y2 = label[p_sign]
            p2 = pre_label[p_sign]
            index = range(0, len(p_sign), p_number)
            index = list(index)
            index.append(len(p_sign))
            index = np.array(index)
            for i in range(len(index) - 1):
                y = y2[index[i]:index[i + 1]]
                x = p2[index[i]:index[i + 1]]
                sign0 = p_sign[index[i]:index[i + 1]]
                ACC.append(acc(y, x))
                tmp0 = tmp[sign0, :]
                density.append(cluster_desity(tmp0))
            print('choose acc:', ACC)
            sign_label0 = label[center_sign1]
            for i in range(num_cluster):
            # pos = np.where(p_sign == sign_all[i])
                p_label1[i] = i
            p_label1 = p_label1.astype(int)
            # p_label1 = label[center_sign1]
            sign_label1 = label[center_sign1]
            for i in range(num_cluster):
                p_label = tile(p_label1[i],p_number).tolist()
                pseudo_label= pseudo_label+p_label
            pseudo_label = np.array(pseudo_label)
            if len(sign_all) != 0:
                pseudo_label = np.hstack((pseudo_label, p_signall_label))
                p_sign = np.hstack((p_sign, sign_all))
            pseudo_hotlabel = torch.LongTensor(pseudo_label)  # 把所有label变成long型
            pseudo_hotlabel = F.one_hot(pseudo_hotlabel, num_classes=num_cluster)  # 把label变成one-hot
            pseudo_hotlabel = pseudo_hotlabel.numpy()

            one_hot_label_in_sign = one_hot_label[sign_all, :]
            print("number_labeled:",len(p_sign))
            # 打乱被标记数据
            shuffle_ix = np.random.permutation(np.arange(len(p_sign)))
            p_sign = p_sign[shuffle_ix]
            pseudo_label = pseudo_label[shuffle_ix]
            pseudo_hotlabel = pseudo_hotlabel[shuffle_ix]

            image_sign = image[p_sign, :]

            # 训练有监督部分
            index_sign = range(0, len(p_sign), batch_size1)
            index_sign = list(index_sign)
            if index_sign[-1] != len(p_sign):
                index_sign.append(len(p_sign))
            index_sign = np.array(index_sign)
            index_sign1 = np.delete(index_sign, len(index_sign) - 1)  # 去掉index_sign里最后一个数
            # aug_image = aug_data(image_sign, aug_transform)
            model.eval()
            model1.eval()
            b = 1
            for i in index_sign1:
                img0 = image_sign[i: index_sign[b]]
                img0 = torch.from_numpy(img0).to(device)
                if b == 1:
                    tmp_sign = model(img0)
                    tmp_sign = model1(tmp_sign)
                    tmp_sign = tmp_sign.cpu().detach().numpy()
                else:
                    tmp_sign0 = model(img0)
                    tmp_sign0 = model1(tmp_sign0)
                    tmp_sign0 = tmp_sign0.cpu().detach().numpy()
                    tmp_sign = np.vstack((tmp_sign, tmp_sign0))
                b = b + 1
            signlable_unique = np.unique(pseudo_label)
            center_sign = np.empty((len(signlable_unique), num_cluster))
            for i in range(len(signlable_unique)):
                items = tmp_sign[pseudo_label == signlable_unique[i]]
                if len(items) == 1:
                    center_sign[i] = items
                else:
                    center_sign[i] = np.mean(items, axis=0)
            center_sign = torch.from_numpy(center_sign).to(device)
            p_centers = np.empty(len(p_sign))
            for i in range(len(p_sign)):
                p_centers = center_sign[pseudo_label[i]]
            tmp_sign_reshape = tmp_sign.reshape(-1,1)
            p_centers_reshape = p_centers.reshape(-1,1)

            model.train()
            model1.train()
            print("supervised:")
            for u in range(repeat):
                for epoch in range(EPOCH1):
                    loss_sum = 0
                    b = 1
                    for i in index_sign1:
                        # 得到每一批次的序号、标签
                        sign = p_sign[i: index_sign[b]]
                        one_hot_label_in_sign1 = pseudo_hotlabel[i: index_sign[b]]
                        m,n1 = one_hot_label_in_sign1.shape
                        one_hot_label_in_sign1 = 0.8 * one_hot_label_in_sign1 + 0.02 * np.ones((m, n1))
                        one_hot_label_in_sign1 = torch.from_numpy(one_hot_label_in_sign1).to(device)
                        prelable = pseudo_label[i: index_sign[b]]
                        b += 1

                        # 提取每个批次中原始image
                        image1 = image[sign, :]
                        # image1 = aug_data(image1, aug_transform).to(device)
                        # 提取特征
                        image1 = torch.from_numpy(image1).to(device)
                        tmp = model(image1)
                        tmp = model1(tmp)

                        loss = loss_supervised(tmp, one_hot_label_in_sign1, center_sign, prelable, gama, signlable_unique)
                        # if b % 10 == 0:
                        #     print("loss: ", loss.item())
                        loss.requires_grad_(True)

                        # 更新网络参数
                        optimizer.zero_grad()  # 将网络中所有的参数的导数都清0
                        loss.backward()  # 计算梯度
                        optimizer.step()  # 更新参数
                        scheduler.step()

                        loss_sum = loss_sum + loss.item()
                        S_loss.append(loss.item())
                    ave_S_loss.append(loss_sum / (b - 1))

            # 用更新后的网络提取特征

            index_image3 = range(0, n, batch_size)
            index_image3 = list(index_image3)
            if index_image3[-1] != n:
                index_image3.append(n)
            index_image3 = np.array(index_image3)
            index_image4 = np.delete(index_image3, len(index_image3) - 1)  # 去掉index_image3里最后一个数
            b = 1
            for i in index_image4:
                if i == 0:
                    image0 = image[i: index_image3[b]]
                    image0 = torch.from_numpy(image0).to(device)
                    tmp = model(image0).cpu().detach().numpy()
                    classfication_tmp = model1(torch.from_numpy(tmp).to(device)).cpu().detach().numpy()
                    b += 1
                else:
                    image0 = image[i: index_image3[b]]
                    image0 = torch.from_numpy(image0).to(device)
                    tmp1 = model(image0).cpu().detach().numpy()
                    tmp2 = model1(torch.from_numpy(tmp1).to(device)).cpu().detach().numpy()
                    tmp = np.vstack((tmp, tmp1))
                    classfication_tmp = np.vstack((classfication_tmp, tmp2))
                    b += 1
            print("     classfication")
            predict = torch.softmax(torch.from_numpy(classfication_tmp), dim=0).cpu().detach().numpy()
            predict_cla = np.argmax(predict, axis=1)
            print("     ACC：", acc(label, predict_cla))



            dis1 = distance_euclidean_scipy(tmp, centers)  # 每个样本到所有中心点的距离
            # dis1 = dis1 * dis1
            dis = dis1.min(1)  # 每个样本到所属中心点的距离
            dis = dis.reshape(-1, 1)
            pre_label1 = pre_label.reshape(-1, 1)
            prelabel_and_dis = np.hstack((pre_label1, dis))
            ratio = 0.35*(g+1)/roundx
            weight = Weight(prelabel_and_dis,n,ratio)
            cnt_array = np.where(weight, 0, 1)
            print("number_unweighted:",np.sum(cnt_array))

            # 更新自步权重
            # dis_t = np.sort(dis)
            # dis_t = dis_t.reshape(-1, 1)  # 得到有序的dis

            # 更新自步权重
            # lam = dis_t[int(n * (round + 1) / (Round+1) - 1), 0]  # 表示自步中的lambda
            # weight = np.where(dis <= lam, -(dis / lam) + 1, 0)
            # lam = 2 * dis_t[int(n * (round+1)/(Round+1)-1), 0]  # 表示自步中的lambda
            # weight = lam * dis_t_inv
            # weight = np.where(weight > 1, 1, weight)
            centers = torch.from_numpy(centers).to(device)
            # 除去已标签数据的weight, image, pre_label
            weight1 = weight
            image1 = image
            Pre_label1 = pre_label
            # if round == 0:
            #     weight1 = weight
            #     image1 = image
            #     Pre_label1 = pre_label
            # else:
            #     weight1 = np.delete(weight, sign_all, 0)
            #     image1 = np.delete(image, sign_all, 0)
            #     Pre_label1 = np.delete(pre_label, sign_all, 0)
            index_image1 = range(0, n - len(sign_all), batch_size1)
            index_image1 = list(index_image1)
            if index_image1[-1] != len(image1):
                index_image1.append(len(image1))
            index_image1 = np.array(index_image1)
            index_image2 = np.delete(index_image1, len(index_image1) - 1)  # 去掉index_image1里最后一个数
            print("unsupervised:")
            # 训练无监督部分
            model.train()
            model1.train()
            for u in range(repeat):
                for epoch in range(EPOCH):
                    loss_sum = 0
                    b = 1
                    for i in index_image2:
                        # 得到每一批次的image, weight
                        image0 = image1[i: index_image1[b]]
                        image0 = torch.from_numpy(image0).to(device)
                        weight0 = weight1[i: index_image1[b]]
                        Pre_label0 = Pre_label1[i: index_image1[b]]
                        b += 1

                        weight0 = torch.from_numpy(weight0).to(device)
                        Pre_label0 = torch.from_numpy(Pre_label0).to(device)
                        tmp = model(image0)

                        # 计算距离
                        dis = torch.empty((len(image0))).to(device)
                        for x in range(len(image0)):
                            a = centers[Pre_label0[x]]
                            dis[x] = torch.norm(tmp[x] - a) * torch.norm(tmp[x] - a)

                        loss = beta * (weight0 * dis) / len(image0)
                        loss = torch.sum(loss)
                        # if b % 100 == 0:
                        #     print("loss: ", loss.item())
                        loss.requires_grad_(True)

                        # 更新网络参数
                        optimizer.zero_grad()  # 将网络中所有的参数的导数都清0
                        loss.backward()  # 计算梯度
                        optimizer.step()  # 更新参数
                        scheduler.step()

                        loss_sum = loss_sum + loss.item()
                        U_loss.append(loss.item())
                    ave_U_loss.append(loss_sum / (b-1))

            model.eval()
            model1.eval()
            # 用更新后的网络提取特征
            index_image3 = range(0, n, batch_size)
            index_image3 = list(index_image3)
            if index_image3[-1] != n:
                index_image3.append(n)
            index_image3 = np.array(index_image3)
            index_image4 = np.delete(index_image3, len(index_image3) - 1)  # 去掉index_image3里最后一个数
            b = 1
            for i in index_image4:
                if i == 0:
                    image0 = image[i: index_image3[b]]
                    image0 = torch.from_numpy(image0).to(device)
                    tmp = model(image0).cpu().detach().numpy()
                    classfication_tmp = model1(torch.from_numpy(tmp).to(device)).cpu().detach().numpy()
                    b += 1
                else:
                    image0 = image[i: index_image3[b]]
                    image0 = torch.from_numpy(image0).to(device)
                    tmp1 = model(image0).cpu().detach().numpy()
                    tmp2 = model1(torch.from_numpy(tmp1).to(device)).cpu().detach().numpy()
                    tmp = np.vstack((tmp, tmp1))
                    classfication_tmp = np.vstack((classfication_tmp, tmp2))
                    b += 1
            print("     classfication")
            predict = torch.softmax(torch.from_numpy(classfication_tmp), dim=0).cpu().detach().numpy()
            predict_cla = np.argmax(predict, axis=1)
            print("     ACC：", acc(label, predict_cla))
            p_number = 200
            p_sign = option1(label_and_dis, n, p_number, sign_all)
            ACC = []
            density = []
            y2 = label[p_sign]
            p2 = pre_label[p_sign]
            index = range(0, len(p_sign), p_number)
            index = list(index)
            index.append(len(p_sign))
            index = np.array(index)
            for i in range(len(index) - 1):
                y = y2[index[i]:index[i + 1]]
                x = p2[index[i]:index[i + 1]]
                sign0 = p_sign[index[i]:index[i + 1]]
                ACC.append(acc(y, x))
                tmp0 = tmp[sign0, :]
                density.append(cluster_desity(tmp0))
            print('200 choose acc:', ACC)
            print(" `   time:", time() - start0)
            tmp_name = "./STL10_result/" + "tmp" + str(round) +"_"+str(g)+ ".npy"
            np.save(tmp_name, tmp)

        model.eval()
        model1.eval()
        # 用更新后的网络提取特征
        index_image3 = range(0, n, batch_size)
        index_image3 = list(index_image3)
        if index_image3[-1] != n:
            index_image3.append(n)
        index_image3 = np.array(index_image3)
        index_image4 = np.delete(index_image3, len(index_image3) - 1)  # 去掉index_image3里最后一个数
        b = 1
        for i in index_image4:
            if i == 0:
                image0 = image[i: index_image3[b]]
                image0 = torch.from_numpy(image0).to(device)
                tmp = model(image0).cpu().detach().numpy()
                classfication_tmp = model1(torch.from_numpy(tmp).to(device)).cpu().detach().numpy()
                b += 1
            else:
                image0 = image[i: index_image3[b]]
                image0 = torch.from_numpy(image0).to(device)
                tmp1 = model(image0).cpu().detach().numpy()
                tmp2 = model1(torch.from_numpy(tmp1).to(device)).cpu().detach().numpy()
                tmp = np.vstack((tmp, tmp1))
                classfication_tmp = np.vstack((classfication_tmp, tmp2))
                b += 1
        print("     classfication")
        predict = torch.softmax(torch.from_numpy(classfication_tmp), dim=0).cpu().detach().numpy()
        predict_cla = np.argmax(predict, axis=1)
        print("     ACC：", acc(label, predict_cla))
        # ARI
        ari = metrics.adjusted_rand_score(label, pre_label)
        ARI.append(ari)
        print('ARI:', ari)

        # NMI
        nmi = metrics.normalized_mutual_info_score(label, pre_label)
        NMI.append(nmi)
        print('NMI:', nmi)


    print("time:", time() - start)


    U_loss = np.array(U_loss)
    S_loss = np.array(S_loss)
    accuary = np.array(accuary)
    ave_U_loss = np.array(ave_U_loss)
    ave_S_loss = np.array(ave_S_loss)
    ARI = np.array(ARI)
    NMI = np.array(NMI)
    image_save = image[sign_all, :]
    np.save('./STL10_result/image_save.npy', image_save)
    np.save('./STL10_result/STL10_tmp.npy', tmp)
    np.save('./STL10_result/U_loss.npy', U_loss)
    np.save('./STL10_result/S_loss.npy', S_loss)
    np.save('./STL10_result/accuary.npy', accuary)
    np.save('./STL10_result/ave_U_loss.npy', ave_U_loss)
    np.save('./STL10_result/ave_S_loss.npy', ave_S_loss)
    np.save('./STL10_result/sign.npy', sign_all)

    np.save('./STL10_result/NMI.npy', NMI)
    np.save('./STL10_result/ARI.npy', ARI)
    torch.save(model.state_dict(), './STL10_result/STL_10.pt')
    print("All_time:", time() - start1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. cuda:0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)