# -*- coding: utf-8 -*-
# !/usr/bin/env python
# coding=utf-8
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
import re
from dbfread import DBF
import csv
if __name__ == "__main__":
    tabl = DBF('road network.dbf', encoding="gbk")
    ll = 0
   

    data_csv = open("road network.csv", "w", newline="")
   
    write_data = csv.writer(data_csv)
    write_data.writerow(tabl.field_names)

    for r in tabl:
        write_data.writerow(list(r.values()))

