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

image_width = 50
image_height = 50
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
            nn.Linear(in_features=11*11*128, out_features=64*64, bias=True), # 64 in_features=8*8*256
            nn.Dropout(0.2),  # drop 50% of the neuron
            nn.LeakyReLU(),
            nn.Linear(in_features=64*64, out_features=50*50, bias=True),
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
        # out = self.layer4(out)
        # out = self.layer5(out)
        # out = self.layer6(out)
        # ff = out.view(out.size(0), -1)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = out.view(out.size(0), 50, 50)
        gg = 0


        # bb = out.detach().numpy()
        # out = self.rfc(out)
        return out


# Hyper Parameters
def train_val(train_dataloader,  num_epochs=30, ):

    net.train()
    net.to(device)

    for epoch in range(num_epochs):
        all_loss = 0

        myloss = all_loss
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images).to(device)
            labels = Variable(labels.float()).to(device)
            # labels = torch.unsqueeze(labels, 1)
            predict_labels = net(images).to(device)
            loss = criterion(predict_labels, labels).to(device)
            optimizer.zero_grad()
            # scheduler.step(loss)
            loss.backward()
            optimizer.step()
            all_loss += loss
            # print('train_batch========>is', i+1)
            lr = optimizer.param_groups[0]['lr']
            # print('batch-is', i+1, '=============>lr', lr)
            # if ((i + 1) % 10 == 0):
            #     print("epoch:%d\tstep:%d\tloss:%f" % (epoch + 1, i + 1, loss.item()))
            # if ((i + 1) % 100 == 0):
            #     torch.save(net.state_dict(), "model.pkl")  # current is model.pkl
            #     print("save model")
        # if all_loss < myloss:
        #     break

        all_loss = all_loss/(i+1)
        # print("epoch:", epoch, "step:", i, "loss:", loss.item())
        print("epoch:", epoch+1,  "loss:", all_loss.item())

    torch.save(net, modelpath)  # current is model.pkl

    print("save last model")
    return net

def Test(testdata, model, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device is %s" % device)
    model.eval()
    model.to(device)
    correct = 0
    for i, (T_images, T_labels) in enumerate(testdata):
        T_images = Variable(T_images).to(device)
        T_labels = Variable(T_labels.float()).to(device)
        # labels = torch.unsqueeze(labels, 1)
        predict_labels_v = model(T_images).to(device)
        pp = predict_labels_v.cpu().detach().numpy()

        tt = T_labels.cpu().detach().numpy()

        for j in range(0,len(pp)):

            ppp = pp[j]
            ttt = tt[j]
            # Picture1 = MinMaxScaler().fit_transform(Picture)
            plt.subplot(2,1,1)
            plt.imshow(ppp, cmap='bwr', aspect='equal')  # coolwarm
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.subplot(2, 1, 2)
            plt.imshow(ttt, cmap='bwr', aspect='equal')  # coolwarm
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.show()

            ll=1


    #     T_pre = predict_labels_v.data.max(1)[1]
    #     correct += T_pre.eq(T_labels.data).sum()
    #     hh = len(testdata.dataset)
    # accuracy = float(correct) / len(testdata.dataset)
    # print('Test =============>test-ture{}%'.format(accuracy * 100))

if __name__ == "__main__":
   
    path1 = './feature/'  # picture-48
    train_x, train_y, target_name = read_picture(path1)
    modelpath = './model/model_3layer.ckpt'

   
    # Len_x = len(train_x)
    # rate_x = 0.7
    # x_train = train_x[0:int(np.round(Len_x*rate_x, 0))]
    # y_train = train_y[0:int(np.round(Len_x*rate_x, 0))]
    # x_test = train_x[int(np.round(Len_x*rate_x, 0)): Len_x]
    # y_test = train_y[int(np.round(Len_x*rate_x, 0)): Len_x]

    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)
    batch_size = 20

    train_X = Variable(torch.Tensor(x_train))
    train_Y = Variable(torch.Tensor(y_train))

    Torch_Traindata = Data.TensorDataset(train_X, train_Y)
    Torch_Traindata = Data.DataLoader(dataset=Torch_Traindata, batch_size=batch_size, shuffle=False,
                                      num_workers=2, drop_last=False)  
  
    test_X = Variable(torch.Tensor(x_test))
    test_Y = Variable(torch.Tensor(y_test))
    Torch_Testdata = Data.TensorDataset(test_X, test_Y)
    Torch_Testdata = Data.DataLoader(dataset=Torch_Testdata, batch_size=batch_size, shuffle=False,
                                      num_workers=2, drop_last=False )  

    # ------------------end------------------#


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is %s" % device)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device is %s" % device)

    net = MyNet()
    learning_rate = 0.001
    num_epochs = 500
    criterion = nn.HuberLoss(reduction='mean').to(device)  # MultiLabelSoftMarginLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

    # scheduler = ExponentialLR(optimizer, gamma=0.95)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=30)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9, last_epoch=-1)


    modle = train_val(Torch_Traindata, num_epochs=num_epochs)
    modle = torch.load(modelpath)
    modle.eval()
    Test(Torch_Testdata, modle)


