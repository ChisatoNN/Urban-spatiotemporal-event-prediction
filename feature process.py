import pandas as pd
import numpy as np
import scipy as sc
import scipy.signal
from scipy import signal
import datetime
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

if __name__ == "__main__":
    Data = pd.read_excel('./data/Zhengzhou data.xlsx')
    Data.columns = ['objiect', 'Time', 'X', 'Y', 'Crime', 'Name']
    Data = Data. sort_values(by=['Time'])
    pp = 9
    # plt.figure()
    # plt.scatter(Data['X'],Data['Y'] )
    # plt.show()
    my_time = Data['Time']
    my_time1 = []

    for i in my_time:
        my_time1.append(i.strftime("%Y-%m-%d") )
    my_time1 = np.array(my_time1)
    Data['Time'] = my_time1.T
    max_X = Data['X'].max()
    min_X = Data['X'].min()

    max_Y = Data['Y'].max()
    min_Y = Data['Y'].min()
    inter_number_x = 50
    inter_number_y = 50
    X_interval = (max_X - min_X)/inter_number_x
    Y_interval = (max_Y - min_Y) / inter_number_y



    hh = Data.groupby(Data['Time'])
    ff = 0
    for name, group in hh:

        gg = Data.iloc[hh.indices[name]]
        print(name)
        Picture = np.zeros([inter_number_x, inter_number_y])

        # if name == '2018-12-23':
        #     ff = 8
        # if ff==8:
        #     for num in range(0, len(group)):
        #
        #         for i in range(0, inter_number_y):
        #             for j in range(0, inter_number_x):
        #                 kk = group['X'].values[1]
        #                 if ((min_X + (j * X_interval)) < group['X'].values[num] < min_X + ((j + 1) * X_interval)):
        #                     if ((min_Y + (i * Y_interval)) < group['Y'].values[num] < (min_Y + ((i + 1) * Y_interval))):
        #                         Picture[j][i] = Picture[j][i] + 1
        #
        #     plt.figure(figsize=(inter_number_x * 0.01, inter_number_y * 0.01))
        #     Picture1 = MinMaxScaler().fit_transform(Picture)
        #     plt.imshow(Picture1, cmap='bwr', aspect='equal')  # coolwarm
        #     plt.axis('off')
        #     plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        #     plt.margins(0, 0)
        #     # plt.show()
        #     plt.savefig('./feature(20)/' + name + '.png', )

        for num in  range(0,len(group)):
            for i in range(0, inter_number_y):
                for j in range(0, inter_number_x):
                    kk = group['X'].values[1]
                    if ((min_X+(j*X_interval)) < group['X'].values[num] < min_X+((j+1)*X_interval)):
                        if ((min_Y + (i*Y_interval)) < group['Y'].values[num] < (min_Y + ((i+1)*Y_interval))):
                            Picture[j][i] = Picture[j][i]+1

        plt.figure(figsize=(inter_number_x*0.01, inter_number_y*0.01))
        # Picture1 = MinMaxScaler().fit_transform(Picture)
        # Picture2 = MinMaxScaler().fit_transform(Picture)
        plt.imshow(Picture, cmap='bwr', aspect='equal')  # coolwarm
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0,left=0, right=1, hspace=0,wspace=0)
        plt.margins(0, 0)
        # plt.show()
        plt.savefig('./feature(50_1)/'+ name + '.png', )


