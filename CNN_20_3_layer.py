# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding=utf-8
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
# import skimage.io as io
import matplotlib.pyplot as plt
from torch.autograd import Variable 
import torch.utils.data as Data
import torch
from sklearn.model_selection import train_test_split
from sklearn import model_selection as cv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  

image_width = 20
image_height = 20
class_number = 5 
captcha_len = 3

def load_sample(sample_dir):
    print("Load picture data")
    file_name_list = os.listdir(sample_dir)
    # file_name_list.sort(key=lambda x: int(x[2:-4])) 
    file_name_list = [os.path.join(sample_dir, file) for file in file_name_list]

    return (np.asarray(file_name_list))

def read_picture(path,fault_code=1):
    data_dir = path 
    images = load_sample(data_dir) 
    print(len(images), images[:2]) 

  
    image_list = []
    lable_img = images[0:-1]
    target_img = images[1:len(images)]
    for file in lable_img:
        image_value = tf.compat.v1.read_file(file)
        img = tf.image.decode_jpeg(image_value, channels=3)

        x = tf.cast(img, dtype=tf.float32) / 255.
        image_list.append(x)
    image_list = np.asarray(image_list)

    image_list1 = []
    target_name = []
    for file in target_img:
        image_value = tf.compat.v1.read_file(file)
        target_name.append(file[-14:-4])
        img = tf.image.decode_jpeg(image_value, channels=1)

        x = tf.cast(img, dtype=tf.float32) / 255.
        image_list1.append(x)
    image_list1 = np.asarray(image_list1)


    target_name = np.array(target_name)
    temp_images = image_list.reshape([len(image_list), 3, image_width, image_height])

    tar_images = image_list1.reshape([len(image_list1),image_width, image_height])

    return temp_images, tar_images, target_name



class MyNet(nn.Module):
    def __init__(self, class_number = class_number, image_width = image_width, image_height = image_height):
        super(MyNet, self).__init__()
        self.class_number = class_number     
        # self.char_len=char_len;             
        self.image_width=image_width       
        self.image_height=image_height    
        self.layer1 = nn.Sequential( 
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),  # drop 50% of the neuron
            # nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.Dropout(0.2),  # drop 50% of the neuron
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=1, ceil_mode=False)
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=1, ceil_mode=False)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=4*4*128, out_features=64*64, bias=True), # 像素64 in_features=8*8*256
            nn.Dropout(0.2),  # drop 50% of the neuron
            nn.LeakyReLU(),
            nn.Linear(in_features=64*64, out_features=image_width*image_height, bias=True),
            nn.LeakyReLU(),

        )
        self.rfc = nn.Sequential(
            nn.Linear(1000, self.class_number))

        self.softmax = nn.Sequential(
            nn.Softmax(dim=0)
        )
    def forward(self, img):
        out = self.layer1(img)
        out = self.layer2(out)
        out = self.layer3(out)


        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = out.view(out.size(0),image_width , image_height)
        return out


def calculate_indicater(TRUE_lable, PRE_lable):
    Len_s = len(TRUE_lable)
    Sum_rmse = 0
    Sum_mae = 0
    Sum_mape = 0
    Sum_R2_1 = 0
    Sum_R2_2 = 0

    Sum_mean = TRUE_lable.cpu().detach().numpy()
    Sum_mean = Sum_mean.reshape(len(Sum_mean), -1)
    Sum_mean = np.mean(Sum_mean, 0)  
    for i in range(0, Len_s):
        data_p = PRE_lable[i].cpu().detach().numpy()
        data_t = TRUE_lable[i].cpu().detach().numpy()

        # min_num = np.min(data_p)
        # where_0 = np.where(data_p <= min_num)
        # data_p[where_0] = min_num

        data_p = data_p.reshape(-1)
        data_t = data_t.reshape(-1)
        Sum_rmse = Sum_rmse + np.mean(np.power((data_t - data_p), 2))
        Sum_mae = Sum_mae + np.mean(abs(data_t - data_p))
        Sum_mape = Sum_mape + np.mean(abs(data_t - data_p) / abs(data_t))

        Sum_R2_1 = Sum_R2_1 + np.power(np.mean((data_t - data_p)), 2)
        Sum_R2_2 = Sum_R2_2 + np.power(np.mean((Sum_mean - data_p)), 2)

    Sum_rmse = Sum_rmse / (i + 1)
    RMSE = np.sqrt(Sum_rmse)
    MAE = Sum_mae / (i + 1)
    MAPE = Sum_mape / (i + 1)
    R2 = 1 - (Sum_R2_1 / Sum_R2_2)

    return RMSE, MAE, MAPE, R2


def calculate_indicater_1(TRUE_lable, PRE_lable):
    Len_s = len(TRUE_lable)
    Sum_rmse = 0
    Sum_mae = 0
    Sum_mape = 0
    Sum_R2_1 = 0
    Sum_R2_2 = 0

    Sum_mean = TRUE_lable
    Sum_mean = Sum_mean.reshape(len(Sum_mean), -1)
    Sum_mean = np.mean(Sum_mean, 0)  
    for i in range(0, Len_s):
        data_p = PRE_lable[i]
        data_t = TRUE_lable[i]

        # min_num = np.min(data_t)
        # where_0 = np.where(data_p <= 0)
        # data_p[where_0] = min_num

        data_p = data_p.reshape(-1)
        data_t = data_t.reshape(-1)
        Sum_rmse = Sum_rmse + np.mean(np.power((data_t - data_p), 2))
        Sum_mae = Sum_mae + np.mean(abs(data_t - data_p))
        Sum_mape = Sum_mape + np.mean(abs(data_t - data_p) / abs(data_t))
        # Sum_R2_1 = Sum_R2_1 + np.mean(np.power((data_t - data_p), 2))
        # Sum_R2_2 = Sum_R2_2 + np.mean(np.power((Sum_mean - data_p), 2))

        Sum_R2_1 = Sum_R2_1 + np.power(np.mean((data_t - data_p)), 2)
        Sum_R2_2 = Sum_R2_2 + np.power(np.mean((Sum_mean - data_p)), 2)

    Sum_rmse = Sum_rmse / (i + 1)
    RMSE = np.sqrt(Sum_rmse)
    MAE = Sum_mae / (i + 1)
    MAPE = Sum_mape / (i + 1)
    R2 = 1 - (Sum_R2_1 / Sum_R2_2)

    return RMSE, MAE, MAPE, R2

# Hyper Parameters
def train_val(train_dataloader,  num_epochs=30, Val_data=0, ):


    net.train()
    net.to(device)
    LOSS = []
    save_data = []
    val_data = []
    for epoch in range(num_epochs):
        all_loss = 0
        RMSE = 0
        MAE = 0
        MAPE = 0
        R2  = 0
        myloss = all_loss 
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images).to(device)
            labels = Variable(labels.float()).to(device)
            # labels = torch.unsqueeze(labels, 1)
            predict_labels = net(images).to(device)

            rmse, mae, mape, r2 = calculate_indicater(labels, predict_labels)

            RMSE = RMSE + rmse
            MAE = MAE + mae
            MAPE = MAPE + mape
            R2 = R2 + r2
            loss = criterion(predict_labels, labels).to(device)
            optimizer.zero_grad()
            # scheduler.step(loss)
            loss.backward()
            optimizer.step()
            all_loss += loss
            # print('train_batch========>is', i+1)
            lr = optimizer.param_groups[0]['lr']
        RMSE = RMSE / (i+1)
        MAE = MAE / (i+1)
        MAPE = MAPE / (i+1)
        R2 = R2 / (i+1)
        print('RMSE----->', RMSE)
        print('MAE----->', MAE)
        print('MAPE----->', MAPE)
        print('R2----->', R2)
        save_data.append([RMSE, MAE, MAPE, R2])
        all_loss = all_loss/(i+1)
        # print("epoch:", epoch, "step:", i, "loss:", loss.item())
        print("epoch:", epoch+1,  "loss:", all_loss.item())
        LOSS.append(all_loss.item())

  

        net.eval()
        net.to(device)
        Pre = []
        Tr = []
        for i, (T_images, T_labels) in enumerate(Val_data):
            T_images = Variable(T_images).to(device)
            T_labels = Variable(T_labels.float()).to(device)
            predict_labels_v = net(T_images).to(device)
            pp = predict_labels_v.cpu().detach().numpy()
            tt = T_labels.cpu().detach().numpy()

            Pre.append(pp.reshape(1, -1))
            Tr.append(tt.reshape(1, -1))

        Pre = np.array(Pre)
        Tr = np.array(Tr)
        rmse, mae, mape, r2 = calculate_indicater_1(Tr, Pre)
        print('RMSE----->', rmse)
        print('MAE----->', mae)
        print('MAPE----->', mape)
        print('R2----->', r2)
        val_data.append([rmse, mae, mape, r2])

    LOSS = np.array(LOSS)
    LOSS = pd.DataFrame(LOSS)
    LOSS.to_csv(loss_path)
    save_data = np.array(save_data)
    save_data = pd.DataFrame(save_data, columns=['RMSE', 'MAE', 'MAPE', 'R2'])
    save_data.to_csv(Save_path)
    val_data = np.array(val_data)
    val_data = pd.DataFrame(val_data, columns=['RMSE', 'MAE', 'MAPE', 'R2'])
    val_data.to_csv(Val_path)
    torch.save(net, modelpath)  # current is model.pkl

    print("save last model")
    return net

def Test(testdata, model, Test_name, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device is %s" % device)
    model.eval()
    model.to(device)
    correct = 0
    Pre = []
    Tr = []
    for i, (T_images, T_labels) in enumerate(testdata):
        T_images = Variable(T_images).to(device)
        T_labels = Variable(T_labels.float()).to(device)
        # labels = torch.unsqueeze(labels, 1)
        predict_labels_v = model(T_images).to(device)
        pp = predict_labels_v.cpu().detach().numpy()

        tt = T_labels.cpu().detach().numpy()
        # min_pp = np.min(pp)
        # min_tt = np.min(tt)
        # where_0 = np.where(pp <= min_tt)
        # pp[where_0] = min_tt

        Pre.append(pp.reshape(1, -1))
        Tr.append(tt.reshape(1, -1))

        plt.subplot(2, 1, 1)
        plt.imshow(pp[0], cmap='bwr', aspect='equal')  # coolwarm
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0.01, wspace=0)
        plt.margins(0, 0)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=16)

        plt.subplot(2, 1, 2)
        plt.imshow(tt[0], cmap='bwr', aspect='equal')  # coolwarm
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0.01, wspace=0)
        plt.margins(0, 0)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=16)
        plt.savefig(predict_path + Test_name[i] + '.png')
        # plt.show()

    Pre = np.array(Pre)
    Tr = np.array(Tr)
    rmse, mae, mape, r2 = calculate_indicater_1(Tr, Pre)
    print('RMSE----->', rmse)
    print(MAE----->', mae)
    print('MAPE----->', mape)
    print('R2----->', r2)


def Get_name(train_y, y_test, target_name):
    Test_name = []
    i = 0
    for matrix_all in train_y:

        for matrix_test in y_test:
            if (matrix_all == matrix_test).all():
                Test_name.append(target_name[i])
        i = i+1

    return np.array(Test_name)



if __name__ == "__main__":
 
    path1 = './feature(20)/'  # picture-48
    train_x, train_y, target_name = read_picture(path1)
    modelpath = './model/model_20_3_layer.ckpt'
    loss_path = './loss/loss_20_3_layer.csv'
    Save_path = './save_data/save_data_20_3_layer.csv'
    Val_path = './save_data/val_data_20_3_layer.csv'
    predict_path = './CNN_20_3/'  

   
    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3)

    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.66)


    Test_name = Get_name(train_y, y_test, target_name)

    batch_size = 30

    train_X = Variable(torch.Tensor(x_train))
    train_Y = Variable(torch.Tensor(y_train))

    Torch_Traindata = Data.TensorDataset(train_X, train_Y)
    Torch_Traindata = Data.DataLoader(dataset=Torch_Traindata, batch_size=batch_size, shuffle=False,
                                      num_workers=2, drop_last=False)  # num_workers使用两个进程提取数据

 
    val_X = Variable(torch.Tensor(x_val))
    val_Y = Variable(torch.Tensor(y_val))
    Torch_Valdata = Data.TensorDataset(val_X, val_Y)
    Torch_Valdata = Data.DataLoader(dataset=Torch_Valdata, batch_size=1, shuffle=False,
                                     num_workers=2, drop_last=False)  
 
    test_X = Variable(torch.Tensor(x_test))
    test_Y = Variable(torch.Tensor(y_test))
    Torch_Testdata = Data.TensorDataset(test_X, test_Y)
    Torch_Testdata = Data.DataLoader(dataset=Torch_Testdata, batch_size= 1, shuffle=False,
                                      num_workers=2, drop_last=False ) 

    # ------------------end------------------#


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is %s" % device)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device is %s" % device)

    net = MyNet()

    learning_rate = 0.001
    num_epochs = 1000
    criterion = nn.HuberLoss(reduction='mean').to(device)  # MultiLabelSoftMarginLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

    # scheduler = ExponentialLR(optimizer, gamma=0.95)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=30)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9, last_epoch=-1)


    # model = train_val(Torch_Traindata, num_epochs=num_epochs, Val_data = Torch_Valdata)
    model = torch.load(modelpath)
    model.eval()
    Test(Torch_Testdata, model, Test_name)