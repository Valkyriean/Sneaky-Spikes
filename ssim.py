import torch
import torch.nn.functional as F
from poisoned_dataset import PoisonedDataset
from datasets import get_dataset
import numpy as np
from spikingjelly.datasets.n_mnist import NMNIST
import copy
import os
import csv

def ssim(img1, img2, K1=0.01, K2=0.03, L=4):
    """
    Compute SSIM (Structural Similarity Index) between two images.
    :param img1: First image (torch tensor)
    :param img2: Second image (torch tensor)
    :param K1: Constant for SSIM (default 0.01)
    :param K2: Constant for SSIM (default 0.03)
    :param L: Dynamic range of pixel values (default 1 for normalized images)
    :return: SSIM value
    """
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    mu1 = F.avg_pool2d(img1, 3, 1, 0)
    mu2 = F.avg_pool2d(img2, 3, 1, 0)
    
    sigma1 = F.avg_pool2d(img1 * img1, 3, 1, 0) - mu1 * mu1
    sigma2 = F.avg_pool2d(img2 * img2, 3, 1, 0) - mu2 * mu2
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 0) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1 + sigma2 + C2))
    return ssim_map.mean()

def calculate_average_ssim(data_clean, data_backdoored):
    """
    Calculate the average SSIM over all samples, frames, and polarities.
    :param data_clean: Clean data (numpy array of shape [N, T, C, H, W])
    :param data_backdoored: Backdoored data (numpy array of shape [N, T, C, H, W])
    :return: Average SSIM value
    """
    if not torch.is_tensor(data_clean):
        data_clean = torch.tensor(data_clean, dtype=torch.float32)
    if not torch.is_tensor(data_backdoored):
        data_backdoored = torch.tensor(data_backdoored, dtype=torch.float32)

    N, T, C, H, W = data_clean.shape
    total_ssim = 0.0
    count = 0
    
    for i in range(N):
        for t in range(T):
            for c in range(C):
                img_clean = data_clean[i, t, c].unsqueeze(0).unsqueeze(0)
                img_backdoored = data_backdoored[i, t, c].unsqueeze(0).unsqueeze(0)
                total_ssim += ssim(img_clean, img_backdoored)
                count += 1
                
    average_ssim = total_ssim / count
    return average_ssim

def save_result(dataset, epsilon,pos, polarity, trigger_size , trigger_type,
                            frame_gap, ssim):
    save_path = "experiments"
    # Create a folder for the experiments, by default named 'experiments'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create if not exists a csv file, appending the new info
    path = '{}/ssim_results.csv'.format(save_path)
    header = ['dataset', 'epsilon', 'pos','polarity', 'trigger_size','type', 'frame_gaps', "ssim"]

    if not os.path.exists(path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Append the new info to the csv file
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([dataset, epsilon,pos, polarity, trigger_size , trigger_type,
                            frame_gap, ssim])
        
       
datasets = ["gesture", "cifar10", "caltech"] 

for dataset in datasets:

    train_set, test_set = get_dataset(dataset, 16, 'data')

    clean_data = np.array([np.array(x[0])for x in train_set])
    
    trigger_type = "blink"
    epsilon=1
    pos="top-left"
    polarity=1
    trigger_size=0.1
    frame_gap = 2
    poisoned_train_data = PoisonedDataset(train_set,0, mode='train', epsilon=1,
                                pos=pos, attack_type=trigger_type, time_step=16,
                                trigger_size=trigger_size, dataname=dataset,
                                polarity=polarity, n_masks=2, least=True, most_polarity=True, frame_gap = frame_gap)
    average_ssim_value = calculate_average_ssim(clean_data, poisoned_train_data.data)
    average_ssim_value = float(average_ssim_value)
    print("Average SSIM:", average_ssim_value) 
    save_result(dataset, epsilon,pos, polarity, trigger_size , trigger_type,
                        frame_gap, average_ssim_value)

    trigger_type = "static"
    poisoned_train_data = PoisonedDataset(train_set,0, mode='train', epsilon=1,
                                pos=pos, attack_type=trigger_type, time_step=16,
                                trigger_size=trigger_size, dataname=dataset,
                                polarity=polarity, n_masks=2, least=True, most_polarity=True, frame_gap = frame_gap)
    average_ssim_value = calculate_average_ssim(clean_data, poisoned_train_data.data)
    average_ssim_value = float(average_ssim_value)
    print("Average SSIM:", average_ssim_value) 
    save_result(dataset, epsilon,pos, polarity, trigger_size , trigger_type,
                        frame_gap, average_ssim_value)