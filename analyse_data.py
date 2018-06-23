import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import os
import numpy as np


def list_append(target_list,input_list):
    t_list = []
    for i in input_list:
        t_list.append(float(i))
    target_list.append(t_list)


def initlist():
    gp = []
    gr = []
    ga = []
    agtp = []
    agtea = []
    aga = []
    tt = []
    rt = []
    return gp, gr, ga, agtp, agtea, aga, tt, rt


def read_file(file_name):
    global gplist
    global grlist
    global galist
    global agtplist
    global agtealist
    global agalist
    global ttlist
    global rtlist
    f = open(file_name, 'r')
    t = []
    for line in f:
        t.append(line.strip().split())
    list_append(gplist,t[0])
    list_append(grlist,t[1])
    list_append(galist,t[2])
    list_append(agtplist,t[3])
    list_append(agtealist,t[4])
    list_append(agalist,t[5])
    list_append(ttlist,t[6])
    list_append(rtlist,t[7])


def draw_different_models(feature, evaluate_name):
    global pic_path
    global feature_dic
    # global evaluate_name
    global list_dic
    goal_list = list_dic[evaluate_name]
    feature_index = feature_dic[feature]
    x = [i / 10 for i in range(1, 10)]
    plt.clf()
    plt.plot(x, goal_list[feature_index], label='DT', color='b')
    plt.plot(x, goal_list[feature_index + 7], label='LR', color='g')
    plt.plot(x, goal_list[feature_index + 14], label='NN', color='r')
    plt.plot(x, goal_list[feature_index + 21], label='SVC', color='c')
    if evaluate_name == 'training_time' or evaluate_name == 'running_time':
        plt.yscale('log')
    plt.xlabel('training percent')
    plt.ylabel(evaluate_name)
    plt.title(feature + '_' + evaluate_name)
    plt.legend(loc='upper left')
    plt.savefig(pic_path + feature + '_' + evaluate_name + '.pdf')
    # plt.show()


def draw_the_same_model(evaluate_name, model_name):
    global pic_path
    global list_dic
    global feature_list
    global model_dic
    global color_dic
    x = [i / 10 for i in range(1, 10)]
    goal_list = list_dic[evaluate_name]
    model_index = model_dic[model_name]
    plt.clf()
    max_scale = -1
    min_scale = 10000
    for i, feature in zip(range(7), feature_list):
        max_scale_item = max(goal_list[model_index + i])
        min_scale_item = min(goal_list[model_index + i])
        if max_scale_item > max_scale:
            max_scale = max_scale_item
        if min_scale_item < min_scale:
            min_scale = min_scale_item
        plt.plot(x, goal_list[model_index + i], label=feature, color=color_dic[feature])
        # plt.plot(x, goal_list[model_index + feature_index], label=evaluate_name[0], color='green')
        # plt.plot(x, goal_list[model_index + feature_index], label=evaluate_name[0], color='red')
        # plt.plot(x, goal_list[model_index + feature_index], label=evaluate_name[0], color='cyan')
        # plt.plot(x, goal_list[model_index + feature_index], label=evaluate_name[0], color='magenta')
        # plt.plot(x, goal_list[model_index + feature_index], label=evaluate_name[0], color='black')
        # plt.plot(x, goal_list[model_index + feature_index], label=evaluate_name[0], color='yellow')
        # plt.plot(x, goal_list[model_index + feature_index], label=evaluate_name[0], color='purple')
    if max_scale != min_scale:
        plt.yticks(np.arange(min_scale, max_scale, (max_scale-min_scale)/10))
    plt.xlabel('training percent')
    plt.ylabel(evaluate_name)
    plt.title(model_name + '_' + evaluate_name)
    plt.legend(loc='upper left')
    plt.savefig(pic_path + model_name + '_' + evaluate_name + '.pdf')




data_path = 'result/data'
pic_path = 'picture/'
eva_list = [0, 3, 6, 7]
file_list = os.listdir(data_path)  # 得到文件夹下的所有文件名称
file_list = sorted(file_list)
gplist, grlist, galist, agtplist, agtealist, agalist, ttlist, rtlist = initlist()
# print_list(file_list)
# read_file(path + '/' + file_list[0],file_list[0])
"""
the order of the data is followed by 'DT', 'LR', 'NN', 'SVC'
and in each part, it is divided into 7 files, followed by 'all_features',
'kernel_pca_10', 'kernel_pca_15', 'kernel_pca_20', 'pca_10', 'pca_15', 'pca_20'

"""
feature_dic = {
    'all_features'  : 0,
    'kernel_pca_10' : 1,
    'kernel_pca_15' : 2,
    'kernel_pca_20' : 3,
    'pca_10'        : 4,
    'pca_15'        : 5,
    'pca_20'        : 6   
}

feature_list = [
    'all_features' ,
    'kernel_pca_10',
    'kernel_pca_15',
    'kernel_pca_20',
    'pca_10'       ,
    'pca_15'       ,
    'pca_20'          
]


color_dic = {
    'all_features'  : 'blue'   ,
    'kernel_pca_10' : 'green'  ,
    'kernel_pca_15' : 'red'    ,
    'kernel_pca_20' : 'cyan'   ,
    'pca_10'        : 'magenta',
    'pca_15'        : 'yellow' ,
    'pca_20'        : 'purple' 
}

list_dic = {
    'general_precision'         : gplist,
    'general_recall'            : grlist,
    'general_accuracy'          : galist,
    'group_top_precision'       : agtplist,
    'group_top_exact_accuracy'  : agtealist,
    'group_accuracy'            : agalist,
    'training_time'             : ttlist,
    'running_time'              : rtlist
}



model_dic = {
    'DT'    :0,
    'LR'    :7,
    'NN'    :14,
    'SVC'   :21
}

for file in file_list:
    read_file(data_path + '/' + file)

i = 1
for evaluate_name in list_dic:
    for feature in feature_list:
        print('Drawing the {0}/56 picture, {1}_{2}'.format(i,feature,evaluate_name))
        draw_different_models(feature, evaluate_name)
        i += 1




# i = 1
# for model_name in model_dic:
#     for evaluate_name in list_dic:
#         print('Drawing the {0}/32 picture, {1}_{2}'.format(i,model_name,evaluate_name))
#         draw_the_same_model(evaluate_name, model_name)
#         i += 1

