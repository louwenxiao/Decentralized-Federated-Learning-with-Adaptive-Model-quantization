import os
import time
import math
import numpy as np

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def killport(port):
    if is_port_in_use(port):
        print("Warning: port " + str(port) + "is in use")
        command = '''kill -9 $(netstat -nlp | grep :''' + str(
            port) + ''' | awk '{print $7}' | awk -F"/" '{ print $1 }')'''
        os.system(command)

def count_dataset(loader):
    counts = np.zeros(len(loader.loader.dataset.classes))
    for _, target in loader.loader:
        labels = target.view(-1).numpy()
        for label in labels:
            counts[label] += 1
    print("class counts: ", counts)
    print("total data count: ", np.sum(counts))

def printer(content, fid):
    print(content)
    content = content.rstrip('\n') + '\n'
    fid.write(content)
    fid.flush()


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{:>3}m {:2.0f}s'.format(m, s)

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
