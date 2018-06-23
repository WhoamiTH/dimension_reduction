import math
import sklearn.svm as sksvm
import sklearn.linear_model as sklin
import sklearn.tree as sktree
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from time import clock
import handle_data
import predict_test

def NN_model(layer1_size, layer2_size, x_dim):
  model = Sequential()
  model.add(Dense(layer1_size, input_dim = x_dim, activation = 'sigmoid')) #sigmoid
  model.add(Dense(layer2_size, activation = 'sigmoid')) #sigmoid
  model.add(Dense(1))
  model.add(Dropout(0.5))
  model.add(Activation('sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam')
  model.summary()
  return model

def train_model(train_data, train_label, test_data, test_label):
    start = clock()
    global model_type
    if model_type == 'LR':
        model = sklin.LogisticRegression()
        model.fit(train_data,train_label.flatten())
    if model_type == 'SVC':
        model = sksvm.SVC(C=0.1,kernel='rbf')
        # model = sksvm.SVC(C=0.1,kernel='poly')
        model.fit(train_data, train_label.flatten())
    if model_type == 'DT':
        model = sktree.DecisionTreeClassifier()
        model.fit(train_data, train_label.flatten())
    if model_type == 'NN':
        model = NN_model(15, 5, train_data.shape[1])
        model.fit(train_data, train_label, validation_data=(test_data, test_label), epochs=20, batch_size=1, verbose=2)
    finish = clock()
    return model, finish-start

def crossvalidation(percent, Ds, Dl, Data, Label, record):
    global pca_or_not
    global kernelpca_or_not
    global num_of_components
    global positive_value
    global negative_value
    global threshold_value
    global mirror_type
    num_of_train = math.ceil(len(Ds) * percent)
    for i in range(rep_times):
        all_group_top_precision = []
        all_group_top_exact_accuracy = []
        all_group_exact_accuracy = []
        start = clock()
        train_index_start,train_x,train_y = handle_data.generate_primal_train_data(Data,Label,Ds,Dl,num_of_train)
        new_data = handle_data.standarize_PCA_data(train_x, Data, pca_or_not, kernelpca_or_not, num_of_components)
        train_data,train_label,test_data,test_label= handle_data.generate_all_data(Ds, Dl, new_data, Label, train_index_start, num_of_train, mirror_type, positive_value, negative_value)
        model,training_time = train_model(train_data, train_label, test_data, test_label)
        finish = clock()
        print(model)
        record.write(str(model) + "\n")
        record.write("-------------------------------------------------------------------------------------------\n")
        predict_test.general_test(test_data, test_label, model, positive_value, negative_value, threshold_value, record)
        all_group_top_precision, all_group_top_exact_accuracy, all_group_exact_accuracy = \
            predict_test.group_test(new_data, Label, Ds, Dl, train_index_start, num_of_train, model, threshold_value,
                                    top, all_group_top_precision, all_group_top_exact_accuracy, all_group_exact_accuracy, record)
        running_time = finish-start
        predict_test.cal_average(all_group_top_precision, all_group_top_exact_accuracy, all_group_exact_accuracy, record)
        record.write("the {0} time training time is {1}\n".format(i+1,training_time))
        record.write("the {0} time running time is {1}\n".format(i+1,running_time))
        record.write("-------------------------------------------------------------------------------------------\n\n\n")

# -------------------------------------global parameters---------------------------------------------------------------
# file_name = '/home/th/Workplace/python/python/Project/GData_test_800.csv'
file_name = 'GData_test_200.csv'
delete_list = [1,2,7,15,17,19,23]
divide_list = [0,8,9,12,13,20,22]
# model_type = 'LR'
# model_type = 'SVC'
# model_type = 'DT'
model_type = 'NN'
# mirror_type = "mirror"
mirror_type = "not_mirror"
path = model_type + '_' + mirror_type + '_result_percent_'
top = 3
rep_times = 3
positive_value = 1
negative_value = -1
threshold_value = 0
transform_or_not = False
kernelpca_or_not = True
pca_or_not = False
num_of_components = 20


# ----------------------------------start processing--------------------------------------------------------------------
data, label = handle_data.loadData(file_name)
dicstart, diclength = handle_data.group(data)

if transform_or_not :
    data = handle_data.transform_data(data,divide_list,delete_list)


for i in range(1,10):
    percent = i/10
    # percent = 0.9
    # record_name = path + str(percent) + '_' + str(i) + '.txt'
    record_name = path + str(percent) + '_'
    if transform_or_not :
        record_name += 'transformed '
    if pca_or_not :
        record_name += 'pca_' + str(num_of_components)
    if kernelpca_or_not :
        record_name += 'kernel_pca_' + str(num_of_components)
    record_name += '.txt'
    record = open(record_name,'w')
    print("the percentage of training data is {0}".format(percent))
    crossvalidation(percent, dicstart, diclength, data, label, record)
    print("\n\n\n")
    record.close()
