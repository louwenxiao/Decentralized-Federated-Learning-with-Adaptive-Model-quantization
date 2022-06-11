import os
import sys
import time
import argparse
import socket
import pickle
import asyncio
import concurrent.futures
import json
import random
import pandas as pd
import datasets 

import numpy as np
from numpy import linalg
import torch

from config import *
from comm_utils import *
import datasets, models


parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--model_type', type=str, default='ale')
# CIFAR10 lr=0.01 epoch=400 decay_rate=0.993
# FMNIST lr=0.002 epoch=150 decay_rate=1
# CIFAR100 lr=0.01 epoch=400 decay_rate=0.996
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_pattern', type=int, default=1)
parser.add_argument('--alpha', type=float, default=200)
parser.add_argument('--algorithm', type=str, default='proposed')
# parser.add_argument('--prob', type=float, default=1.0)      # 压缩率
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--decay_rate', type=float, default=0.99)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--local_updates', type=int, default=-1)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
SERVER_IP = "127.0.0.1"
RESULT = [[0],[0],[0],[4]]      # 分别用来保存：带宽MB，时间s，精度，损失
model_size = 0

def main():
    result = []
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    # init config
    common_config = CommonConfig()
    common_config.master_listen_port_base += random.randint(0, 20) * 21
    common_config.dataset_type = args.dataset_type
    common_config.batch_size = args.batch_size
    common_config.epoch = args.epoch
    common_config.lr = args.lr
    common_config.decay_rate = args.decay_rate


    with open("worker_config.json") as json_file:
        workers_config = json.load(json_file)

    # init p2p topology
    workers_config['worker_config_list'] = workers_config['worker_config_list'][:8]         # 初始化用户个数
    worker_num = len(workers_config['worker_config_list'])
    adjacency_matrix = np.ones((worker_num, worker_num), dtype=np.int)
    
    topology = get_topology(worker_num=worker_num)
    ratios = np.ones(worker_num)
    
    for worker_idx in range(worker_num):
        adjacency_matrix[worker_idx][worker_idx] = 0

    p2p_port = np.zeros_like(adjacency_matrix)
    curr_port = common_config.p2p_listen_port_base + random.randint(0, 20) * 200
    for idx_row in range(len(adjacency_matrix)):
        for idx_col in range(len(adjacency_matrix[0])):
            if adjacency_matrix[idx_row][idx_col] != 0:
                curr_port += 1
                p2p_port[idx_row][idx_col] = curr_port

    # create workers
    for worker_idx, worker_config in enumerate(workers_config['worker_config_list']):
        custom = dict()
        custom["neighbor_info"] = dict()

        custom["bandwidth"] = worker_config["bandwidth"]

        common_config.worker_list.append(
            Worker(config=ClientConfig(idx=worker_idx,
                                       client_ip=worker_config['ip_address'],
                                       master_ip=SERVER_IP,
                                       master_port=common_config.master_listen_port_base+worker_idx,
                                       custom=custom),
                   common_config=common_config, 
                   user_name=worker_config['user_name'],
                   pass_wd=worker_config['pass_wd'],
                   local_scripts_path=workers_config['scripts_path']['local'],
                   remote_scripts_path=workers_config['scripts_path']['remote'],
                   location='local'
                   )
        )
    
    # init workers' config
    for worker_idx, worker_config in enumerate(workers_config['worker_config_list']):
        for neighbor_idx, link in enumerate(adjacency_matrix[worker_idx]):
            if link == 1:
                neighbor_config = common_config.worker_list[neighbor_idx].config
                neighbor_ip = neighbor_config.client_ip
                neighbor_bandwidth = neighbor_config.custom["bandwidth"]

                # neighbor ip, send_port, listen_port
                common_config.worker_list[worker_idx].config.custom["neighbor_info"][neighbor_idx] = \
                        (neighbor_ip, p2p_port[worker_idx][neighbor_idx], p2p_port[neighbor_idx][worker_idx], neighbor_bandwidth)

    # Create model instance
    global_model = models.create_model_instance(common_config.dataset_type)
    init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
    model_size = init_para.nelement() * 4 / 1024 / 1024
    print("Model Size: {} MB".format(model_size))

    # partition dataset
    train_data_partition, test_data_partition = partition_data(common_config.dataset_type, args.data_pattern,worker_num=worker_num)
    for worker_idx, worker in enumerate(common_config.worker_list):
        worker.config.para = init_para
        worker.config.custom["train_data_idxes"] = train_data_partition.use(worker_idx)
        worker.config.custom["test_data_idxes"] = test_data_partition.use(worker_idx)

    # connect socket and send init config
    communication_parallel(common_config.worker_list, action="init")
    
    training_recorder = TrainingRecorder(common_config.worker_list, common_config.recoder)
    if args.local_updates > 0:
        local_steps = args.local_updates
    else:
        local_steps = int(np.ceil(50000 / worker_num / args.batch_size))
    print("local steps: {}".format(local_steps))
    
    
    # bandwidth = np.zeros((2, worker_num))
    # bandwidth[0] = np.random.rand(worker_num) * 1.5 + 1.0 # 1MB/s ~ 2.5MB/s
    # bandwidth[1] = bandwidth[0].copy() / 2.0 # 假设上传速度为下载速度的一半
    # print("bandwidth:", bandwidth)

    total_round = 0
    for epoch_num in range(1, 1+common_config.epoch):           # 需要添加一个代码：每一次通信，需要记录精度等信息
        print("\n--**--\nEpoch: {}".format(epoch_num))

        total_transfer_size = 0
        for worker in common_config.worker_list:
            worker_data_size = np.sum(topology[worker.idx]) * model_size
            common_config.recoder.add_scalar('Data Size-'+str(worker.idx), worker_data_size, epoch_num)
            total_transfer_size += worker_data_size
            neighbors_list = list()
            for neighbor_idx, link in enumerate(topology[worker.idx]):
                if link == 1:
                    neighbors_list.append((neighbor_idx, 1.0 / (np.max([np.sum(topology[worker.idx]), np.sum(topology[neighbor_idx])])+1)))
            # send_data_socket((local_steps, neighbors_list, ratios[worker.idx],bandwidth), worker.socket)
            send_data_socket((local_steps, neighbors_list), worker.socket)
            # 对应110行，client文件

        common_config.recoder.add_scalar('Data Size', total_transfer_size, epoch_num)
        common_config.recoder.add_scalar('Num of Links', np.sum(topology), epoch_num)
        common_config.recoder.add_scalar('Avg Rate', np.average(ratios), epoch_num)
        comp_time, acc, loss = training_recorder.get_train_info()
        
        
        RESULT[0].append(RESULT[0][epoch_num-1]+total_transfer_size)
        RESULT[1].append(RESULT[1][epoch_num-1]+comp_time)
        RESULT[2].append(acc)
        RESULT[3].append(loss)

        pd.DataFrame(RESULT).to_csv('./result/DPSGD_{}_{}_{}_{}_{}.csv'.format(args.dataset_type,
                                args.data_pattern,args.lr,args.batch_size,args.local_updates))       # 数据集，数据分布，学习率，batch，τ

    # close socket
    for worker in common_config.worker_list:
        worker.socket.shutdown(2)
        worker.socket.close()
        
        

class TrainingRecorder(object):
    def __init__(self, worker_list, recorder, beta=0.95):
        self.worker_list = worker_list
        self.worker_num = len(worker_list)
        self.beta = beta
        self.moving_consensus_distance = np.ones((self.worker_num, self.worker_num)) * 1e-6
        self.avg_update_norm = 0
        self.round = 0
        self.epoch = 0
        self.recorder = recorder
        self.total_time = 0

        for i in range(self.worker_num):
            self.moving_consensus_distance[i][i] = 0

    def get_train_info(self):
        self.round += 1
        communication_parallel(self.worker_list, action="get")
        avg_train_loss = 0.0
        round_update_norm = np.zeros(self.worker_num)
        max_comp_time, avg_acc, avg_loss = 0,0,0
        
        for worker in self.worker_list:
            comp_time, acc, test_loss = worker.train_info[-1]
            max_comp_time = max(max_comp_time,comp_time)
            avg_acc = avg_acc + acc
            avg_loss = avg_loss + test_loss
        
        avg_acc = avg_acc/len(self.worker_list)
        avg_loss = avg_loss/len(self.worker_list)

        print("max_comp_time: {}, avg_acc: {}, avg_loss: {}".format(max_comp_time, avg_acc, avg_loss))

        return max_comp_time, avg_acc, avg_loss

    def get_test_info(self):
        self.epoch += 1
        communication_parallel(self.worker_list, action="get")
        avg_acc = 0.0
        avg_test_loss = 0.0
        epoch_time = 0.0
        for worker in self.worker_list:
            _, worker_time, acc, loss, train_loss = worker.train_info[-1]
            self.recorder.add_scalar('Accuracy/worker_' + str(worker.idx), acc, self.round)
            self.recorder.add_scalar('Test_loss/worker_' + str(worker.idx), loss, self.round)
            self.recorder.add_scalar('Time/worker_' + str(worker.idx), worker_time, self.epoch)

            avg_acc += acc
            avg_test_loss += loss
            epoch_time = max(epoch_time, worker_time)
        
        avg_acc /= self.worker_num
        avg_test_loss /= self.worker_num
        self.total_time += epoch_time
        self.recorder.add_scalar('Time/total', epoch_time, self.epoch)
        self.recorder.add_scalar('Accuracy/average', avg_acc, self.epoch)
        self.recorder.add_scalar('Test_loss/average', avg_test_loss, self.epoch)
        self.recorder.add_scalar('Accuracy/round_average', avg_acc, self.round)
        self.recorder.add_scalar('Test_loss/round_average', avg_test_loss, self.round)
        print("Epoch: {}, time: {}, average accuracy: {}, average test loss: {}, average train loss: {}".format(self.epoch, self.total_time, avg_acc, avg_test_loss, train_loss))

def communication_parallel(worker_list, action, data=None):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_list),)
        tasks = []
        for worker in worker_list:
            if action == "init":
                tasks.append(loop.run_in_executor(executor, worker.send_init_config))
            elif action == "get":
                tasks.append(loop.run_in_executor(executor, worker.get_config))
            elif action == "send":
                tasks.append(loop.run_in_executor(executor, worker.send_data, data))
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
    except:
        sys.exit(0)

def partition_data(dataset_type, data_pattern, worker_num=10):      # 使用6个用户训练，需要添加一个IID
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)

    partition_sizes = np.ones((10, worker_num)) * (1.0 / worker_num)  

    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=partition_sizes)            # 这种方式与训练集数据保持一致
    
    return train_data_partition, test_data_partition

def get_topology(worker_num=10):
    topology = np.zeros((worker_num, worker_num), dtype=np.int)
    for worker_idx in range(worker_num):
        topology[worker_idx][worker_idx-1] = 1
        topology[worker_idx-1][worker_idx] = 1
    return topology

if __name__ == "__main__":
    main()
