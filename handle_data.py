import numpy as np
import random
import sklearn.preprocessing as skpre
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA


def loadData(file_name):
    tem = np.loadtxt(file_name, dtype=np.str, delimiter=',', skiprows=1)
    tem_data = tem[:, 1:-1]
    tem_label = tem[:, -1]
    # data = tem_data.astype(np.float).astype(np.int)
    data = tem_data.astype(np.float)
    label = tem_label.astype(np.float).astype(np.int)
    return data, label


def group(Data):
    i = 0
    j = 1
    l = 1
    t = Data[0][2]
    Ds = {}
    Dl = {}
    Ds[0] = 0
    while j < len(Data):
        if (t != Data[j][2]):
            Dl[i] = l
            Ds[i + 1] = j
            i += 1
            l = 1
            t = Data[j][2]
            j += 1
        elif (j == len(Data) - 1):
            Dl[i] = l + 1
            j += 1
        else:
            l += 1
            j += 1
    return Ds, Dl


def get_the_number(Data,column):
    item_Data = Data[:,column:column+1]
    item_Data = set([int(i) for i in item_Data])
    return len(item_Data)


def transform_the_column(Data,column,n_rows):
    length = get_the_number(Data,column)
    item_column = np.zeros((n_rows,length))
    the_column = Data[:,column:column+1]
    # print(the_column)
    for i in range(n_rows):
        item_column[i,int(the_column[i])-1] = 1
    return item_column


def transform_date_to_age(Data,n_rows):
    item_age = np.zeros((n_rows,1))
    for i in range(n_rows):
        item_age[i,0] = Data[i,1] - Data[i,23]
    return item_age


def transform_data(Data,divide_list,delete_list):
    n_rows,n_column = Data.shape
    item_age = transform_date_to_age(Data,n_rows)
    tem_data = transform_the_column(Data,0,n_rows)
    for column in range(1,n_column):
        if column in divide_list:
            item_column = transform_the_column(Data,column,n_rows)
        elif column in delete_list:
            continue
        else:
            item_column = Data[:,column:column+1]
        tem_data = np.hstack((tem_data,item_column))
    tem_data = np.hstack((tem_data,item_age))
    return tem_data


def data_extend(Data_1, Data_2):
    m = list(Data_1)
    n = list(Data_2)
    return m + n

def condense_data_pca(Data, num_of_components):
    pca = PCA(n_components=num_of_components)
    pca.fit(Data)
    return pca


def condense_data_kernel_pca(Data, num_of_components):
    kernelpca = KernelPCA(n_components=num_of_components)
    kernelpca.fit(Data)
    return kernelpca


def standardize_data(Data):
    scaler = skpre.StandardScaler()
    scaler.fit(Data)
    return scaler


def standarize_PCA_data(train_data, Data, pca_or_not, kernelpca_or_not, num_of_components):
    scaler = standardize_data(Data)
    if pca_or_not :
        new_data = scaler.transform(train_data)
        pca = condense_data_pca(new_data, num_of_components)
        new_data = scaler.transform(Data)
        new_data = pca.transform(new_data)
    elif kernelpca_or_not :
        new_data = scaler.transform(train_data)
        kernelpca = condense_data_kernel_pca(new_data, num_of_components)
        new_data = scaler.transform(Data)
        new_data = kernelpca.transform(new_data)
    else:
        new_data = scaler.transform(Data)
    return new_data

def exchange(test_y):
    ex_ty_list = []
    rank_ty = []
    for i in range(len(test_y)):
        ex_ty_list.append((int(test_y[i]),i+1))
    exed_ty = sorted(ex_ty_list)
    for i in exed_ty:
        rank_ty.append(i[1])
    return rank_ty


def generate_primal_train_data(Data,Label,Ds,Dl,num_of_train):
    train_index_start = random.randint(0,len(Ds)-num_of_train)
    front = Ds[train_index_start]
    end = Ds[train_index_start+num_of_train-1]+Dl[train_index_start+num_of_train-1]
    train_x = Data[front:end,:]
    train_y = Label[front:end]
    return train_index_start,train_x,train_y


def handleData_extend_mirror(Data, Label, start, length, positive_value, negative_value):
    temd = []
    teml = []
    for j in range(length):
        for t in range(length):
            if j != t:
                temd.append(data_extend(Data[start + j], Data[start + t]))
                if Label[start + j] > Label[start + t]:
                    # teml.append([-1])
                    teml.append([negative_value])
                else:
                    teml.append([positive_value])
    return temd, teml


def handleData_extend_not_mirror(Data, Label, start, length, positive_value, negative_value):
    temd = []
    teml = []
    for j in range(length):
        for t in range(j+1,length):
            temd.append(data_extend(Data[start + j], Data[start + t]))
            if Label[start + j] > Label[start + t]:
                teml.append([negative_value])
            else:
                teml.append([positive_value])
    return temd, teml


def generate_all_data(Ds, Dl, Data, Label, train_index_start, num_of_train, mirror_type, positive_value, negative_value):
    tem_data_train = []
    tem_label_train = []
    tem_data_test = []
    tem_label_test = []
    for group_index_start in range(len(Ds)):
        group_start = Ds[group_index_start]
        length = Dl[group_index_start]
        if group_index_start<train_index_start:
            temd,teml = handleData_extend_mirror(Data, Label, group_start, length, positive_value, negative_value)
            tem_data_test = tem_data_test + temd
            tem_label_test = tem_label_test + teml
        elif (group_index_start>=train_index_start and group_index_start<train_index_start+num_of_train):
            if mirror_type == 'mirror':
                temd, teml = handleData_extend_mirror(Data, Label, group_start, length, positive_value, negative_value)
            else:
                temd, teml = handleData_extend_not_mirror(Data, Label, group_start, length, positive_value, negative_value)
            tem_data_train = tem_data_train + temd
            tem_label_train = tem_label_train + teml
        else:
            temd, teml = handleData_extend_mirror(Data, Label, group_start, length, positive_value, negative_value)
            tem_data_test = tem_data_test + temd
            tem_label_test = tem_label_test + teml
    train_data = np.array(tem_data_train)
    train_label = np.array(tem_label_train)
    test_data = np.array(tem_data_test)
    test_label = np.array(tem_label_test)
    return train_data, train_label, test_data, test_label

