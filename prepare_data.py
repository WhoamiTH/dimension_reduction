import numpy as np
import sklearn.preprocessing as skpre
from sklearn.decomposition import PCA

def loadData(file_name):
    tem = np.loadtxt(file_name, dtype=np.str, delimiter=',', skiprows=1)
    tem_data = tem[:, 1:-1]
    tem_label = tem[:, -1]
    # data = tem_data.astype(np.float).astype(np.int)
    data = tem_data.astype(np.float)
    label = tem_label.astype(np.float).astype(np.int)
    return data, label

file_name = '/home/th/WorkSpace/python/python/Project/GData_test_800.csv'
