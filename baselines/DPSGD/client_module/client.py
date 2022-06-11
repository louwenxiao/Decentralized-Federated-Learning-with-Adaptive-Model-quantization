import os
import time
import argparse
import asyncio
import copy
import concurrent.futures

import numpy as np
from numpy import linalg
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import ClientConfig
from client_comm_utils import *
from training_utils import train, test
import utils
import datasets, models

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--idx', type=str, default="0",
                    help='index of worker')
parser.add_argument('--master_ip', type=str, default="127.0.0.1",
                    help='IP address for controller or ps')
parser.add_argument('--master_port', type=int, default=58000, metavar='N',
                    help='')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--min_lr', type=float, default=0.002)
parser.add_argument('--ratio', type=float, default=0.2)
parser.add_argument('--decay_rate', type=float, default=0.97)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--use_cuda', action="store_false", default=True)
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--algorithm', type=str, default='proposed')


args = parser.parse_args()

if args.visible_cuda == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = str((int(args.idx))%4)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_cuda
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
# device2 = torch.device("cpu")

def main():
    client_config = ClientConfig(
        idx=args.idx,
        master_ip=args.master_ip,
        master_port=args.master_port
    )
    utils.create_dir("logs")
    recorder = SummaryWriter("logs/log_"+str(args.idx))
    # receive config
    master_socket = connect_send_socket(args.master_ip, args.master_port)
    config_received = get_data_socket(master_socket)
    for k, v in config_received.__dict__.items():
        setattr(client_config, k, v)

    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    
    # create model
    local_model = models.create_model_instance(args.dataset_type)
    torch.nn.utils.vector_to_parameters(client_config.para, local_model.parameters())
    local_model.to(device)
    para_nums = torch.nn.utils.parameters_to_vector(local_model.parameters()).nelement()
    model_size = para_nums * 4 / 1024 / 1024
    print("Model Size: {} MB".format(model_size))

    # create dataset
    print("train data len : {}\n".format(len(client_config.custom["train_data_idxes"])))
    train_dataset, test_dataset = datasets.load_datasets(args.dataset_type)
    train_loader = datasets.create_dataloaders(train_dataset, batch_size=args.batch_size, selected_idxs=client_config.custom["train_data_idxes"])
    print("train dataset:")
    utils.count_dataset(train_loader)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, selected_idxs=client_config.custom["test_data_idxes"], shuffle=False)
    print("test dataset:")
    utils.count_dataset(test_loader)
    
    # create p2p communication socket
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=20,)
    tasks = []
    for _, (neighbor_ip, send_port, listen_port, _) in client_config.custom["neighbor_info"].items():
        tasks.append(loop.run_in_executor(executor, connect_send_socket, neighbor_ip, send_port))
        tasks.append(loop.run_in_executor(executor, connect_get_socket, client_config.client_ip, listen_port))
    loop.run_until_complete(asyncio.wait(tasks))

    # save socket for later communication
    for task_idx, neighbor_idx in enumerate(client_config.custom["neighbor_info"].keys()):
        client_config.send_socket_dict[neighbor_idx] = tasks[task_idx*2].result()
        client_config.get_socket_dict[neighbor_idx] = tasks[task_idx*2+1].result()
    loop.close()

    epoch_lr = args.lr
    neighbor_para = dict()
    for epoch in range(1, 1+args.epoch):
        print("--**--")
        epoch_start_time = time.time()
        if epoch > 1 and epoch % 1 == 0:
            epoch_lr = max((args.decay_rate * epoch_lr, args.min_lr))
        print("epoch-{} lr: {}".format(epoch, epoch_lr))

        local_steps, comm_neighbors =  get_data_socket(master_socket)
        # comm_neighbors包含两个，第一个是用户名，第二个是聚合权重

        # 计算本地更新。获得新模型的中间结果
        start_time = time.time()
        optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, momentum=0.9, weight_decay=5e-4)
        train_loss = train(local_model, train_loader, optimizer, local_iters=local_steps, device=device)
        new_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).clone().detach().to(device)
        
        # 发送本地量化模型到邻居，并更新本地副本
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=20,)
        tasks = []
        for neighbor_idx, _ in comm_neighbors:
            print("neighbor : {}".format(neighbor_idx))
            tasks.append(loop.run_in_executor(executor, send_data_socket, new_para,
                                                client_config.send_socket_dict[neighbor_idx]))
            tasks.append(loop.run_in_executor(executor, get_compressed_model, client_config, 
                                                neighbor_idx, para_nums))
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
        print("compress, send and get time: ", time.time() - start_time)
        
        # 更新
        mid_para = aggregate_model(new_para,client_config,comm_neighbors)  
        torch.nn.utils.vector_to_parameters(mid_para, local_model.parameters())
        
        test_loss, acc = test(local_model, test_loader, device)
        # 测试，下面是发送（计算时间，精度，测试损失到协调器
        
        send_data_socket((5, acc, test_loss), master_socket)

        recorder.add_scalar('acc_worker-' + str(args.idx), acc, epoch)
        recorder.add_scalar('test_loss_worker-' + str(args.idx), test_loss, epoch)
        recorder.add_scalar('train_loss_worker-' + str(args.idx), train_loss, epoch)
        print("epoch: {}, train loss: {}, test loss: {}, test accuracy: {}".format(epoch, train_loss, test_loss, acc))
        print("\n\n")
        
        
    torch.save(local_model.state_dict(), './logs/model_'+str(args.idx)+'.pkl')
    # close socket
    for _, conn in client_config.send_socket_dict.items():
        conn.shutdown(2)
        conn.close()
    for _, conn in client_config.get_socket_dict.items():
        conn.shutdown(2)
        conn.close()
    master_socket.shutdown(2)
    master_socket.close()


def aggregate_model(new_para,client_config,comm_neighbors):
    with torch.no_grad():
        para_delta = torch.zeros_like(new_para)
        for neighbor_idx, _ in comm_neighbors:
            para_delta += client_config.neighbor_paras[neighbor_idx]
        local_para =  (para_delta + new_para) / (len(comm_neighbors)+1)
    return torch.cuda.FloatTensor(local_para.to(torch.float32))


def get_compressed_model(config, name, nelement):
    received_para = get_data_socket(config.get_socket_dict[name])
    config.neighbor_paras[name] = received_para
    

if __name__ == '__main__':
    main()
