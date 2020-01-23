#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Misc helper functions """

import os
import copy
import cv2
import numpy as np
import subprocess
import copy
from matplotlib.pyplot import imshow

import torch
import torchvision.transforms as transforms
from torchvision import models

from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad


class NormalizeInverse(transforms.Normalize):
    # Undo normalization on images

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super(NormalizeInverse, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(NormalizeInverse, self).__call__(tensor.clone())


def create_folder(folder_name):
    try:
        subprocess.call(['mkdir','-p',folder_name])
    except OSError:
        None

def save_saliency_map(image, saliency_map, filename):
    """ 
    Save saliency map on image.
    
    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W) 
        filename: string with complete path and file extension

    """

    image = image.data.cpu().numpy()
    saliency_map = saliency_map.data.cpu().numpy()

    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0,1)

    saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    saliency_map = cv2.resize(saliency_map, (224,224))

    image = np.uint8(image * 255).transpose(1,2,0)
    image = cv2.resize(image, (224, 224))

    # Apply JET colormap
    color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    
    # Combine image with heatmap
    img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)

    cv2.imwrite(filename, np.uint8(255 * img_with_heatmap))


def compute_and_store_saliency_maps(sample_loader, model, device, max_batch_num, saliency_method, saliency_path):
    create_folder(saliency_path)

    for batch_idx, (data, target) in enumerate(sample_loader):
        data, target = data.to(device).requires_grad_(), target.to(device)

        _ = model.forward(data)
        saliency_map = saliency_method.saliency(data)
    
        filename = "saliency_map_" + str(batch_idx)
        torch.save(saliency_map, os.path.join(saliency_path, filename))
        
        if batch_idx == max_batch_num:
            break


def compute_saliency_maps(sample_loader, fullgrad_model, gradcam_model, device):

    for batch_idx, (data, target) in enumerate(sample_loader):
        data, target = data.to(device).requires_grad_(), target.to(device)

        _ = model.forward(data)

        cam = fullgrad.saliency(data)
        cam_simple = simple_fullgrad.saliency(data)


def remove_salient_pixels(image_batch, saliency_maps, num_pixels=100, most_salient=True, replacement=[1.0]):
    # Check that the data and the saliency map have the same batch size and the
    # same image dimention.
    assert image_batch.size()[0] == saliency_maps.size()[0], \
            "Images and saliency maps do not have the same batch size."
    assert image_batch.size()[2:3] == saliency_maps.size()[2:3], \
            "Images and saliency maps do not have the same image size."
    
    [batch_size, channel_size, column_size, row_size] = image_batch.size()

    for i in range(batch_size):
        #print("num_pixels:{}".format(num_pixels))
        indexes = torch.topk(saliency_maps[i].view((-1)), k=num_pixels, largest=most_salient)[1]
        #print("indexes:{}".format(indexes))
        rows = indexes / row_size
        columns = indexes % row_size
        if len(replacement) == 1:
            image_batch[i, :, rows, columns] = replacement[0]
        else:
            for j in range(len(replacement)):
                image_batch[i, j, rows, columns] = replacement[i]
    torch.cuda.empty_cache()
    return image_batch

def remove_random_salient_pixels(image_batch, seed, k_percentage, replacement):

    output = copy.deepcopy(image_batch)
    output.requires_grad = False
    torch.manual_seed(seed)
    [batch_size, channel_size, column_size, row_size] = image_batch.size()

   # create binary mask for all batched
    bin_mask = torch.FloatTensor(batch_size, 3, 224, 224).uniform_() < k_percentage

    for i in range(batch_size):
        if len(replacement) == 1:
            # replace True values with replacement value
            output[i,:,:,:][bin_mask[i, :, :, :]] = replacement[0]
        else:
            for j in range(len(replacement)):
                output[i, j, :, :][bin_mask[i, j, :, :]] = replacement[i]
    torch.cuda.empty_cache()

    return output
