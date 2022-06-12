import os
from typing import List
import paramiko
from scp import SCPClient

from torch.utils.tensorboard import SummaryWriter
from comm_utils import *


class ClientAction:
    LOCAL_TRAINING = "local_training"


class Worker:
    def __init__(self,
                 config,
                 common_config,
                 user_name,
                 pass_wd,
                 local_scripts_path,
                 remote_scripts_path,
                 location="remote"
                 ):
        self.config = config
        self.common_config = common_config
        self.user_name = user_name
        self.pass_wd = pass_wd
        self.idx = config.idx
        self.local_scripts_path = local_scripts_path
        self.remote_scripts_path = remote_scripts_path

        self.socket = None
        self.train_info = list()

        # Start remote process
        # while not self.__check_worker_script_exist():
        #     self.__send_scripts()
        #     break

        if location=="remote":
            self.__send_scripts()
            self.__start_remote_worker_process()
        else:
            self.__start_local_worker_process()

    def __check_worker_script_exist(self):
        if not len(self.local_scripts_path) is 0:
            return True
        else:
            return False

    def __send_scripts(self):
        s = paramiko.SSHClient()
        s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        s.connect(self.config.client_ip, username=self.user_name, password=self.pass_wd)

        scp_client = SCPClient(s.get_transport(), socket_timeout=15.0)
        try:
            scp_client.put(self.local_scripts_path, self.remote_scripts_path, True)
        except FileNotFoundError as e:
            print(e)
            print("file not found " + self.local_scripts_path)
        else:
            print("file was uploaded to", self.user_name, ": ", self.config.client_ip)
        scp_client.close()
        s.close()

    def __start_remote_worker_process(self):
        s = paramiko.SSHClient()
        s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        s.connect(self.config.client_ip, username=self.user_name, password=self.pass_wd)
        stdin, stdout, stderr = s.exec_command('cd ' + self.remote_scripts_path + ';ls')
        print(stdout.read().decode('utf-8'))
        s.exec_command('cd ' + self.remote_scripts_path + '/client_module' + ';nohup python.exe client.py --listen_port ' + str(self.listen_port) + ' --master_listen_port ' + str(
            self.config.master_port) + ' --idx ' + str(self.idx) + '&')

        print("start process at ", self.user_name, ": ", self.config.client_ip)
    
    def __start_local_worker_process(self):
        python_path = '/data/yxu/software/Anaconda/envs/torch1.6/bin/python'
        os.system('cd ' + os.getcwd() + '/client_module' + ';nohup  ' + python_path + ' -u client.py --master_ip ' 
                     + self.config.master_ip + ' --master_port ' + str(self.config.master_port)  + ' --idx ' + str(self.idx) 
                     + ' --dataset_type ' + str(self.common_config.dataset_type) 
                     + ' --epoch ' + str(self.common_config.epoch) + ' --batch_size ' + str(self.common_config.batch_size) 
                     + ' --ratio ' + str(self.common_config.ratio) + ' --lr ' + str(self.common_config.lr) 
                     + ' --decay_rate ' + str(self.common_config.decay_rate) + ' > client_' + str(self.idx) + '_log.txt 2>&1 &')

        print("start process at ", self.user_name, ": ", self.config.client_ip)

    def send_data(self, data):
        send_data_socket(data, self.socket)

    def send_init_config(self):
        self.socket = connect_get_socket(self.config.master_ip, self.config.master_port)
        send_data_socket(self.config, self.socket)

    def get_config(self):
        self.train_info = list()
        self.train_info.append(get_data_socket(self.socket))


class CommonConfig:
    def __init__(self):
        self.recoder: SummaryWriter = SummaryWriter()

        self.dataset_type = 'CIFAR10'
        self.model_type = 'AlexNet'
        self.use_cuda = True
        self.training_mode = 'local'

        self.epoch_start = 0
        self.epoch = 200

        self.batch_size = 64
        self.test_batch_size = 64

        self.lr = 0.1
        self.decay_rate = 0.97
        # self.step_size = 1.0
        self.ratio = 1.0
        self.algorithm = "proposed"


        self.master_listen_port_base = 30999
        self.p2p_listen_port_base = 24000

        self.worker_list: List[Worker] = list()


class ClientConfig:
    def __init__(self,
                 idx: int,
                 client_ip: str,
                 master_ip: str,
                 master_port: int,
                 custom: dict = dict()
                 ):
        self.idx = idx
        self.client_ip = client_ip
        self.master_ip = master_ip
        self.master_port = master_port
        self.custom = custom
        self.epoch_num: int = 1
        self.model = list()
        self.para = dict()
        self.resource = {"CPU": "1"}
        self.acc: float = 0
        self.loss: float = 1
        self.running_time: int = 0
