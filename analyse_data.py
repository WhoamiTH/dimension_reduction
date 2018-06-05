from matplotlib import pyplot as plt
import os


def print_list(l):
    for i in l:
        print(i)


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
    # print_list(t)
    # gplist[name.split('.')[0]] = t[0]
    # grlist[name.split('.')[0]] = t[1]
    # galist[name.split('.')[0]] = t[2]
    # agtplist[name.split('.')[0]] = t[3]
    # agtealist[name.split('.')[0]] = t[4]
    # agalist[name.split('.')[0]] = t[5]
    # ttlist[name.split('.')[0]] = t[6]
    # rtlist[name.split('.')[0]] = t[7]
    gplist.append(t[0])
    grlist.append(t[1])
    galist.append(t[2])
    agtplist.append(t[3])
    agtealist.append(t[4])
    agalist.append(t[5])
    ttlist.append(t[6])
    rtlist.append(t[7])


def draw_different_models(goal_list, feature_index, evaluate_index, feature_name, evaluate_name):
    global pic_path
    x = [i / 10 for i in range(1, 10)]
    plt.plot(x, goal_list[feature_index], label='DT', color='b')
    plt.plot(x, goal_list[feature_index + 7], label='LR', color='g')
    plt.plot(x, goal_list[feature_index + 14], label='NN', color='r')
    plt.plot(x, goal_list[feature_index + 21], label='SVC', color='c')
    plt.xlabel('training percent')
    plt.ylabel(evaluate_name[evaluate_index])
    plt.title(feature_name[feature_index] + '_' + evaluate_name[evaluate_index])
    plt.legend(loc='upper left')
    plt.savefig(pic_path + feature_name[feature_index] + '_' + evaluate_name[evaluate_index])
    plt.show()


def draw_the_same_model(goal_list, model_index, model_name, evaluate_index, feature_name, evaluate_name):
    global pic_path
    x = [i / 10 for i in range(1, 10)]
    color_list = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'yellow', 'purple']
    for i in range(7):
        plt.plot(x, goal_list[model_index + i], label=feature_name[i], color=color_list[i])
        # plt.plot(x, goal_list[model_index + feature_index], label=evaluate_name[0], color='green')
        # plt.plot(x, goal_list[model_index + feature_index], label=evaluate_name[0], color='red')
        # plt.plot(x, goal_list[model_index + feature_index], label=evaluate_name[0], color='cyan')
        # plt.plot(x, goal_list[model_index + feature_index], label=evaluate_name[0], color='magenta')
        # plt.plot(x, goal_list[model_index + feature_index], label=evaluate_name[0], color='black')
        # plt.plot(x, goal_list[model_index + feature_index], label=evaluate_name[0], color='yellow')
        # plt.plot(x, goal_list[model_index + feature_index], label=evaluate_name[0], color='purple')
        plt.xlabel('training percent')
        plt.ylabel(evaluate_name[evaluate_index])
        plt.title(feature_name[i] + '_' + evaluate_name[evaluate_index])
        plt.legend(loc='upper left')
        plt.savefig(pic_path + model_name[model_index] + '_' + feature_name[i] + '_' + evaluate_name[evaluate_index])
        plt.show()




data_path = 'result/data'
pic_path = 'picture/'
feature_name = {
    0: 'all_features',
    1: 'kernel_pca_10',
    2: 'kernel_pca_15',
    3: 'kernel_pca_20',
    4: 'pca_10',
    5: 'pca_15',
    6: 'pca_20'
}

evaluate_name = {
    0: 'general_precision',
    1: 'general_recall',
    2: 'general_accuracy',
    3: 'group_top_precision',
    4: 'group_top_exact_accuracy',
    5: 'group_accuracy',
    6: 'training_time',
    7: 'running_time'
}

model_name = {
    0:'DT',
    1:'LR',
    2:'NN',
    3:'SVC'
}

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
for file in file_list:
    read_file(data_path + '/' + file)

# for i in eva_list:
#     for j in range(7):
#         draw_different_models(agtplist, j, i, feature_name, evaluate_name)

for i in eva_list:
    for j in range(4):
        draw_the_same_model(gplist, j, model_name, i, feature_name, evaluate_name)
