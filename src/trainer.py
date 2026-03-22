# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
from Modules import KMeans
import pickle
import torch.nn.functional as F
import os
import time as Time

def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['HR@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics  


def LSHT_inference(model, args, data_loader):
    device = args.device
    model = model.to(device)
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in data_loader:
            test_batch = [x.to(device) for x in test_batch]
            
            rep_diffu= model(test_batch[0], test_batch[1], train_flag=False)
            scores_rec_diffu = model.diffu_rep_pre(rep_diffu)
            metrics = hrs_and_ndcgs_k(scores_rec_diffu, test_batch[1], [5, 10, 20])
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print(test_metrics_dict_mean)




# def contra_loss(diffu,labels,num_cluster,criterion):
#     centers=diffu[torch.randperm(len(diffu))[:num_cluster]]
#     for i in range(num_cluster):
#         if torch.sum(labels == i) == 0:
#             centers[i] = diffu[torch.randint(0, diffu.size(0), (1,))]
#         else:
#             centers[i] = torch.mean(diffu[labels == i], dim=0)
    
#     distance=None    
#     # centers=torch.stack(centers)
#     for i in range(num_cluster):
#         centers = torch.cat((centers[i:i + 1], centers[0:i],
#         centers[i + 1:]), 0)
#         distance_= torch.einsum('nc,kc->nk', [
#     nn.functional.normalize(diffu[labels==i], dim=1),
#     nn.functional.normalize(centers, dim=1)
# ])

#         if distance is None:
#             distance = F.softmax(distance_, dim=1)
#         else:
#             distance = torch.cat((distance, F.softmax(distance_, dim=1)),
#                                             0)
#             idx = torch.zeros(distance.shape[0], dtype=torch.long).cuda()

#     loss = criterion(distance, idx)
#     return loss

def model_train(tra_data_loader, val_data_loader, test_data_loader, model, args, logger):
    """
    模型训练函数
    :param tra_data_loader: 训练数据加载器
    :param val_data_loader: 验证数据加载器
    :param test_data_loader: 测试数据加载器
    :param model: 模型
    :param args: 参数
    :param logger: 日志记录器
    """
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks  # 指定计算指标的k值列表，如[5, 10, 20]
    model = model.to(device)
    is_parallel = args.num_gpu > 1
    if is_parallel:
        model = nn.DataParallel(model)  # 如果有多GPU，则使用数据并行
    optimizer = optimizers(model, args)  # 根据参数创建优化器
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)  # 学习率调度器
    # 初始化最佳指标字典
    best_metrics_dict = {'Best_HR@5': 0, 'Best_NDCG@5': 0, 'Best_HR@10': 0, 'Best_NDCG@10': 0, 'Best_HR@20': 0, 'Best_NDCG@20': 0}
    # 记录最佳指标对应的轮次
    best_epoch = {'Best_epoch_HR@5': 0, 'Best_epoch_NDCG@5': 0, 'Best_epoch_HR@10': 0, 'Best_epoch_NDCG@10': 0, 'Best_epoch_HR@20': 0, 'Best_epoch_NDCG@20': 0}
    bad_count = 0  # 用于早停的计数器
    
    # criterion = nn.CrossEntropyLoss().to(args.device)  # 注释掉的交叉熵损失函数
    
    for epoch_temp in range(epochs):        
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model.train()  # 设置模型为训练模式
        metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        flag_update = 0  # 标记是否更新了最佳模型
        train_start = Time.time()  # 记录训练开始时间
        
        # 遍历训练数据批次
        for index_temp, train_batch in enumerate(tra_data_loader):
            train_batch = [x.to(device) for x in train_batch]  # 将批次数据移到设备上
            optimizer.zero_grad()  # 清空梯度
            # 前向传播，获取扩散表示和编码
            diffu_rep, code = model(train_batch[0], train_batch[1], train_flag=True)  
            # 计算扩散损失和对比损失
            loss_diffu_value, loss_contra = model.loss_diffu_ce(diffu_rep, train_batch[1])  # 使用这个而不是上面的注释行
            # loss_contra = contra_loss(diffu_rep)  # 注释掉的对比损失计算
            # loss_contra = 0  # 注释掉的固定对比损失
            loss_all = loss_diffu_value + args.lambda_contra * loss_contra  # 总损失
            loss_all.backward()  # 反向传播
            optimizer.step()  # 更新参数
            
            # 每隔一定批次打印一次损失
            if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
                print('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
                logger.info('[%d/%d] Loss: %.4f Loss_contra: %.4f' % (index_temp, len(tra_data_loader), loss_all.item(), loss_contra))
        
        print("loss in epoch {}: {}".format(epoch_temp, loss_all.item()))  # 打印当前轮次的损失
        print("Train cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time()-train_start)))  # 打印训练耗时

        lr_scheduler.step()  # 更新学习率

        # 在指定轮次进行验证
        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            model.eval()  # 设置模型为评估模式
            with torch.no_grad():  # 不计算梯度
                metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
                # metrics_dict_mean = {}  # 注释掉的平均指标字典
                eval_start = Time.time()  # 记录评估开始时间

                # 遍历验证数据批次
                for val_batch in val_data_loader:
                    val_batch = [x.to(device) for x in val_batch]  # 将验证批次数据移到设备上
                    # 获取扩散表示和编码
                    rep_diffu, code = model(val_batch[0], val_batch[1], train_flag=False)
                    # 通过扩散表示预测分数
                    scores_rec_diffu = model.diffu_rep_pre(rep_diffu)    # 内积计算
    
                    # 计算各种指标
                    metrics = hrs_and_ndcgs_k(scores_rec_diffu, val_batch[1], metric_ks)
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)  # 将指标添加到字典中
            # if epoch_temp % 10 == 0:
            #   torch.save(rep_diffu, str(epoch_temp)+'generation_nv.pt')  # 注释掉的保存扩散表示
            
            # 更新最佳指标
            for key_temp, values_temp in metrics_dict.items():
                values_mean = round(np.mean(values_temp) * 100, 4)  # 计算平均值并保留4位小数
                if values_mean > best_metrics_dict['Best_' + key_temp]:  # 如果当前指标优于最佳指标
                    flag_update = 1  # 标记模型已更新
                    bad_count = 0  # 重置早停计数器
                    best_metrics_dict['Best_' + key_temp] = values_mean  # 更新最佳指标
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp  # 记录最佳指标对应的轮次
            # if epoch_temp % 10 == 0:
            #    torch.save(code, str(epoch_temp)+'code_nc.pt')  # 注释掉的保存编码
            print("Evalution cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time()-eval_start)))  # 打印评估耗时)
        
            if flag_update == 0:  # 如果模型没有更新
                bad_count += 1  # 增加早停计数器
            else:  # 如果模型更新了
                print(best_metrics_dict)  # 打印最佳指标
                print(best_epoch)  # 打印最佳轮次
                logger.info(best_metrics_dict)  # 记录最佳指标
                logger.info(best_epoch)  # 记录最佳轮次
                best_model = copy.deepcopy(model)  # 深拷贝当前模型作为最佳模型
            if bad_count >= args.patience:  # 如果连续未更新次数超过耐心值
                break  # 早停
    
    logger.info(best_metrics_dict)  # 记录最佳指标
    logger.info(best_epoch)  # 记录最佳轮次
        
    if args.eval_interval > epochs:  # 如果评估间隔大于总轮次
        best_model = copy.deepcopy(model)  # 直接将当前模型作为最佳模型
    
    top_100_item = []  # 用于存储top100项目的列表
    
    # 在测试集上评估最佳模型
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in test_data_loader:
            test_batch = [x.to(device) for x in test_batch]  # 将测试批次数据移到设备上
            # 使用最佳模型进行预测
            rep_diffu, code = best_model(test_batch[0], test_batch[1], train_flag=False)
            # 通过扩散表示预测分数
            scores_rec_diffu = best_model.diffu_rep_pre(rep_diffu)   # 内积计算
            # scores_rec_diffu = best_model.routing_rep_pre(rep_diffu)   # 路由 # 注释掉的路由方法
            
            # 获取top100项目索引
            _, indices = torch.topk(scores_rec_diffu, k=100)
            top_100_item.append(indices)

            # 计算测试指标
            metrics = hrs_and_ndcgs_k(scores_rec_diffu, test_batch[1], metric_ks)
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    
    # 计算测试指标的平均值
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    
    print('Test------------------------------------------------------')
    logger.info('Test------------------------------------------------------')
    print(test_metrics_dict_mean)  # 打印测试指标
    logger.info(test_metrics_dict_mean)  # 记录测试指标
    print('Best Eval---------------------------------------------------------')
    logger.info('Best Eval---------------------------------------------------------')
    print(best_metrics_dict)  # 打印最佳验证指标
    print(best_epoch)  # 打印最佳轮次
    logger.info(best_metrics_dict)  # 记录最佳验证指标
    logger.info(best_epoch)  # 记录最佳轮次

    print(args)  # 打印参数

    # 如果需要计算多样性指标
    if args.diversity_measure:
        path_data = '../datasets/data/category/' + args.dataset +'/id_category_dict.pkl'
        with open(path_data, 'rb') as f:
            id_category_dict = pickle.load(f)  # 加载ID到类别的映射字典
        id_top_100 = torch.cat(top_100_item, dim=0).tolist()  # 连接所有top100项目ID
        category_list_100 = []
        for id_top_100_temp in id_top_100:
            category_temp_list = [] 
            for id_temp in id_top_100_temp:
                category_temp_list.append(id_category_dict[id_temp])  # 将项目ID转换为类别
            category_list_100.append(category_temp_list)
        category_list_100.append(category_list_100)  # 这行可能有问题，应该是将每个批次的类别列表添加到总列表中
        path_data_category = '../datasets/data/category/' + args.dataset +'/DiffuRec_top100_category.pkl'
        with open(path_data_category, 'wb') as f:
            pickle.dump(category_list_100, f)  # 保存top100项目的类别信息
            
    return best_model, test_metrics_dict_mean  # 返回最佳模型和测试指标平均值
    
