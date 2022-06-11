from typing import List

class ClientConfig:
    def __init__(self,
                 idx: int,
                 master_ip: str,
                 master_port: int,
                 custom: dict = dict()
                 ):
        self.idx = idx
        self.master_ip = master_ip
        self.master_port = master_port
        self.custom = custom
        self.epoch_num: int = 1
        self.model = list()
        self.para = dict()
        self.estimated_local_paras = dict()
        self.neighbor_paras = dict()
        self.neighbor_indices = dict()
        self.estimated_consensus_distance = dict()
        self.send_socket_dict = dict()
        self.get_socket_dict = dict()
        self.bandwidth = dict()
        self.resource = {"CPU": "1"}
        self.acc: float = 6
        self.loss: float = 1
        self.running_time: int = 0
