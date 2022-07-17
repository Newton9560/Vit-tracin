import os
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json
import torch
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

classifications = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


'''
description: 对列表进行分割
param {*} full_list 待分割的列表
param {*} shuffle 是否打乱
param {*} ratio1 第一个分割出的列表的比例
param {*} ratio2 第二个分割出的列表的比例
return {*} 分割出的第一个列表，分割出的第二个列表，分割出的第三个列表
'''
def ran_split(full_list,shuffle=False,ratio1=0.8,ratio2=0.1):
	n_total = len(full_list)
	offset1 = int(n_total * ratio1)
	offset2 = int(n_total * ratio2) + offset1
	if n_total == 0 or offset1 < 1:
		return [], full_list
	if shuffle:
		random.shuffle(full_list)   # 打乱排序
	return full_list[:offset1], full_list[offset1:offset2], full_list[offset2:]

'''
description: 用于读取xml文件
param {*} root 文件根目录
param {*} paths 文件路径
return {*} 路径列表，标注列表
'''
def read_xml(root_path, paths, image_root_paths):
    path_list = []
    annotation_list = []
    bar = tqdm(paths)
    for p in bar:
        tree = ET.parse(root_path + "\\" + p)
        root = tree.getroot()
        for child in root:
            if(child.tag == 'filename'):
                if child.text in image_root_paths:
                    path_list.append(child.text)
                else:
                    break
            elif(child.tag == 'object'):
                for c in child:
                    if(c.tag == 'name'):
                        annotation_list.append(c.text)
                        break
                break
    return path_list, annotation_list


def write_data_path(filename, data_paths, data_paths_labels):
    with open("classification.json", "r") as f:
        classification = json.load(f)
    data_paths = tqdm(data_paths)
    data_paths_labels = tqdm(data_paths_labels)
    with open(filename + "-path.txt", 'w') as file1:
        file1.truncate(0)
        for path in data_paths:
            
                file1.write(path + '\n')
        print("[Success] Write paths of data in file {0}".format(filename + "-path.txt"))
    with open(filename + "-anno.txt", 'w') as file2:
        file2.truncate(0)
        for annotation in data_paths_labels:
            label = classification[annotation]
            file2.write(label + '\n')
        print("[Success] Write annotations of data in file {0}".format(filename + "-anno.txt"))
        
        

def read_file(filename, form = None):
    data = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            if(form == "int"):
                data.append(int(line.strip()))
            else:
                data.append(line.strip())
    return data


def write_data(filename, list):
    with open(filename, 'w') as file:
        file.truncate(0)
        for data in list:
            file.write(data + '\n')
        print("[Success] Write {0} lines of data in file {1}".format(list.length, filename + "-path.txt"))
        
@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def get_weight_path(root, num):
    return root + str(num) + ".pth"


def get_class_from_path(path):
    """根据文件路径返回类别

    Args:
        path (_string_): 文件路径

    Returns:
        _string_: 类别
    """
    with open("../cifar10/classification.json", "r") as f:
        classification = json.load(f)
    path = Path(path)
    return classification[str(path.parts[3])]


def get_class_from_path2(path):
    """根据文件路径返回类别

    Args:
        path (_string_): 文件路径

    Returns:
        _string_: 类别
    """
    path = Path(path)
    return str(path.parts[3])


def tracin_get(a, b):
    """ dot product between two lists"""
    return sum([torch.dot(at.flatten(), bt.flatten()) for at, bt in zip(a, b)])
    # breakpoint()


def get_gradient(grads, model):
    """
    pick the gradients by name.
    """
    return [grad for grad, (n, p) in zip(grads, model.named_parameters())]

def get_sorted_index(list, ration = 1.0):
    """对list进行排序，返回下表

    Args:
        list (_list_): 待排序的数组
        ration (float, optional): 返回前ration%个. Defaults to 1.0.

    Returns:
        _list_: 从大到小的下表组成的list，取前ration%个
    """
    index  = sorted(range(len(list)), key=lambda k: list[k], reverse=True)
    return index[:int(len(list)*ration)]


def statistic_histogram(data, name):
    return 0


def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for index, row in enumerate(plots):
        if index == 0:
            y.append((row[2])) 
            x.append((row[1]))
        else:
            y.append(float(row[2])) 
            x.append(float(row[1]))
    return x ,y


def statistic_histogram2(num1, num2):
    data1 = read_file("../tracin_file/checkpoint" + str(num1) + "_0.3.txt")
    data2 = read_file("../tracin_file/checkpoint" + str(num2) + "_0.3.txt")
    nums1 = [0,0,0,0,0,0,0,0,0,0]
    for a in data1:
        nums1[classifications.index(get_class_from_path2(a))] = nums1[classifications.index(get_class_from_path2(a))] + 1
        
    nums2 = [0,0,0,0,0,0,0,0,0,0]
    for a in data2:
        nums2[classifications.index(get_class_from_path2(a))] = nums2[classifications.index(get_class_from_path2(a))] + 1

    plt.figure(figsize=(20,10), dpi=80)
    plt.style.use('ggplot')
    plt.xlabel('Class')
    plt.ylabel('Number')

    ax1 = plt.subplot(221)
    ax1.set_title('The statistics of epoch {}'.format(str(num1)))
    ax1.bar(
        x = classifications, ## 设置x轴内容
        height = nums1,  ## 设置y轴内容
    )

    ax2 = plt.subplot(222)
    ax2.set_title('The statistics of epoch {}'.format(str(num2)))
    ax2.bar(
        x = classifications, ## 设置x轴内容
        height = nums2,  ## 设置y轴内容
    )
    
    
def statistic_histogram(txt):
    data1 = read_file(txt)
    nums1 = [0,0,0,0,0,0,0,0,0,0]
    for a in data1:
        nums1[classifications.index(get_class_from_path2(a))] = nums1[classifications.index(get_class_from_path2(a))] + 1
        
    
    plt.figure(figsize=(10,5), dpi=80)
    plt.style.use('ggplot')
    plt.xlabel('Class')
    plt.ylabel('Number')
    
    plt.title('The statistics of {}'.format(str(txt)))
    plt.bar(
        x = classifications, ## 设置x轴内容
        height = nums1,  ## 设置y轴内容
    )

def show_lines(x, y):
    colors = ['darkred','burlywood','cyan','darkgreen','darkviolet','red','yellow']
    plt.figure(figsize=(15,10), dpi=80)
    for index, data in enumerate(x):
        plt.plot(data[1:], y[index][1:], color=colors[index], label='Default')
        plt.xlabel('Steps',fontsize=20)
        plt.ylabel('Acc',fontsize=20)
    plt.show()