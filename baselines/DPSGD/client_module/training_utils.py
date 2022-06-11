import sys
import time
import math
import re
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def train(model, data_loader, optimizer, local_iters=None, device=torch.device("cpu")):
    model.train()
    print("local_iters: ", local_iters)

    train_loss = 0.0
    samples_num = 0
    for iter_idx in range(local_iters):
        data, target = next(data_loader)
            
        data, target = data.to(device), target.to(device)
        
        output = model(data)

        optimizer.zero_grad()
        
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        train_loss += (loss.item() * data.size(0))
        samples_num += data.size(0)

    if samples_num != 0:
        train_loss /= samples_num
    
    return train_loss

def test(model, data_loader, device=torch.device("cpu")):
    model.eval()
    data_loader = data_loader.loader
    test_loss = 0.0
    test_accuracy = 0.0

    correct = 0

    with torch.no_grad():
        for data, target in data_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            # sum up batch loss
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()

            correct += batch_correct
            

    test_loss /= len(data_loader.dataset)
    test_accuracy = np.float(1.0 * correct / len(data_loader.dataset))

    return test_loss, test_accuracy
