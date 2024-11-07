import shutil

from dataset import Dataset_Loader,DatasetLoaderTest
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 
from torchvision import models
import torch.nn as nn
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast,GradScaler
import timm
#%matplotlib inline
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import network as myNetwork

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, 0)
    return softmax_x
    

    


def infer_label(model : nn.Module,file_path:str,model_path:str):
    test_data = Dataset_Loader(file_path, train_flag=False)
    test_loader = DataLoader(dataset=test_data, num_workers=1, pin_memory=True, batch_size=1)
    """
    model = models.resnet50(pretrained=False)
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, 214)
    """
    model = model.cuda()
    # 加载训练好的模型
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    right_num = 0

    for i, (image, label) in enumerate(test_loader):
        """
        src = image.numpy()
        src = src.reshape(3, 512, 512)
        src = np.transpose(src, (1, 2, 0))
        """

        image = image.cuda() 
        #label = label.cuda()
        pred = model(image)
        pred = pred.data.cpu().numpy()[0]
        score = softmax(pred)
        pred_id = np.argmax(score)
        fake_possibility = score[1]
        if int(label) == int(pred_id):
            right_num += 1
        print("{},{:.3f},{:.3f},{}".format(i,right_num/(i+1),fake_possibility,label))



def infer_label2(model : nn.Module,file_path:str,model_path:str,batch_size =16):
    test_data = Dataset_Loader(file_path, train_flag=False,preprocess_mode=1)
    test_loader = DataLoader(dataset=test_data, num_workers=4, pin_memory=True, batch_size=batch_size)


    """
    model = models.resnet50(pretrained=False)
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, 214)
    """
    model = model.cuda()
    # 加载训练好的模型
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    right_num = 0
    total_num = 0
    for i, (image, label) in enumerate(test_loader):
        total_num += len(label)
        """
        src = image.numpy()
        src = src.reshape(3, 512, 512)
        src = np.transpose(src, (1, 2, 0))
        """

        image = image.cuda()


        #label = label.cuda()
        with autocast():
            pred = model(image)

            #print(pred)
            pred = torch.nn.functional.softmax(pred, dim=1)

        score = pred.data.cpu().numpy()





        #pred_id = np.argmax(score,axis=1)
        for idx in range(len(label)):
            fake_possibility = score[idx][1]

            if fake_possibility > 0.5:
                pred_id = 1
            else:
                pred_id = 0
            if int(label[idx]) == int(pred_id):
                right_num += 1
            print("{},{:.3f},{:.3f},{}".format(i,right_num/(total_num),fake_possibility,label[idx]))


def infer_label2_same(model : nn.Module,file_path:str,model_path:str,batch_size =16):
    test_data = DatasetLoaderTest(file_path, train_flag=False,preprocess_mode=1)
    test_loader = DataLoader(dataset=test_data, num_workers=4, pin_memory=True, batch_size=batch_size)



    model = model.cuda()

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    right_num = 0
    total_num = 0
    for i, (image, label,path) in enumerate(test_loader):
        total_num += len(label)
        """
        src = image.numpy()
        src = src.reshape(3, 512, 512)
        src = np.transpose(src, (1, 2, 0))
        """

        image = image.cuda()


        #label = label.cuda()
        with autocast():
            pred = model(image)

            #print(pred)
            pred = torch.nn.functional.softmax(pred, dim=1)

        score = pred.data.cpu().numpy()





        #pred_id = np.argmax(score,axis=1)
        for idx in range(len(label)):
            fake_possibility = score[idx][1]

            if fake_possibility > 0.5:
                pred_id = 1
            else:
                pred_id = 0
            if int(label[idx]) == int(pred_id):
                right_num += 1
            else:
                shutil.copy(path[idx],r"./mylogger/wrong")
            print("{},{:.3f},{:.3f},{}".format(i,right_num/total_num,fake_possibility,label[idx]))



    my_mode = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=2)

    model_path = r"./step_output/12000_step_convnextv2_tiny-base_ema.pth.tar"
    infer_label2(my_mode, r"./wangtu1.txt", model_path, 8)
