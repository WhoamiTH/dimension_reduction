import numpy as np
import random
import handle_data


def calacc(rank, label):
    en = 0
    for i in range(len(rank)):
        if rank[i] == label[i]:
            en += 1
    return en / len(rank)


def count_general_pre(y_true, y_pred, positive_value, negative_value, threshold_value):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if int(y_true[i]) == positive_value and y_pred[i] > threshold_value:
            tp += 1
        elif int(y_true[i]) == positive_value and y_pred[i] < threshold_value:
            fn += 1
        elif int(y_true[i]) == negative_value and y_pred[i] > threshold_value:
            fp += 1
    return tp, fp, fn


def count_top(y_true, y_pred, top):
    tp = 0
    exact = 0
    if top <= len(y_true):
        top_true = y_true[:top]
        top_pred = y_pred[:top]
    else:
        top_true = y_true
        top_pred = y_pred
    len_top = len(top_pred)
    for i in range(len_top):
        if top_pred[i] in top_true:
            tp += 1
            if top_pred[i] == top_true[i]:
                exact += 1
    group_top_pre = tp/top
    group_top_exact_accuracy = exact/top
    return group_top_pre, group_top_exact_accuracy


def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


def general_test(test_data, test_label, model, positive_value, negative_value, threshold_value, record):
    general_predict = model.predict(test_data)
    general_tp, general_fp, general_fn = count_general_pre(test_label, general_predict, positive_value, negative_value, threshold_value)
    general_precision, general_recall = precision_recall(general_tp, general_fp, general_fn)
    general_accuracy = (len(test_label) - general_fp - general_fn) / len(test_label)
    # --------------------------------------------------------------------
    record.write(f"the general true positive is {general_tp}\n")
    record.write(f"the general false positive is {general_fp}\n")
    record.write(f"the general false negative is {general_fn}\n")
    general_tn = len(test_label) - general_fp - general_fn - general_tp
    record.write(f"the general true negative is {general_tn}\n")
    # --------------------------------------------------------------------
    record.write(f"the general precision is {general_precision}\n")
    record.write(f"the general recall is {general_recall}\n")
    record.write(f"the general accuracy is {general_accuracy}\n")
    record.write("-------------------------------------------------------------------------------------------\n")


def record_middle_result(name, list, record):
    record.write(name)
    for i in list:
        record.write(f'{i:2d}\t')
    record.write('\n')


def rank_the_group(group_data, reference, model, threshold, record):
    low = []
    high = []
    if len(reference) <= 1:
        return reference
    else:
        pivot = random.choice(reference)
        # record.write(f"the pivot is          {pivot:2d}\n")
        reference.remove(pivot)
        for i in reference:
            t = handle_data.data_extend(group_data[pivot-1], group_data[i-1])
            t = np.array(t).reshape((1,-1))
            if model.predict(t) > threshold:
                low.append(i)
            else:
                high.append(i)
    # record_middle_result('low', low, record)
    # record_middle_result('high', high, record)
    low = rank_the_group(group_data, low, model, threshold, record)
    high = rank_the_group(group_data, high, model, threshold, record)
    high.append(pivot)
    return high + low


def record_rank_reference(reference, rank, predict_rank, record):
    t = [i for i in range(1,len(rank)+1)]
    record_middle_result('                      ', t, record)
    record_middle_result('the random order is   ', reference, record)
    record_middle_result('the true rank is      ', rank, record)
    record_middle_result('the predict rank is   ', predict_rank, record)

def group_test_relative(group_data, group_label, length, model, threshold_value, record):
    graph = []
    for j in range(length):
        edge_set = [-1 for i in range(length)]
        for t in range(length):
            if j != t:
                temd = handle_data.data_extend(group_data[j], group_data[t])
                temi = handle_data.data_extend(group_data[t], group_data[j])
                temd = np.array(temd).reshape((1, -1))
                temi = np.array(temi).reshape((1, -1))
                if model.predict(temd) > threshold_value:
                    edge_set[t] = 1
                    rtd = 1
                    record.write(f'{j+1:2d}\t{t+1:2d}\t 1({model.predict_proba(temd)})\t')
                else:
                    rtd = 0
                    record.write(f'{j+1:2d}\t{t+1:2d}\t-1({model.predict_proba(temd)})\t')
                if group_label[j] > group_label[t]:
                    record.write('-1\n')
                else:
                    record.write('1\n')
                if model.predict(temi) > threshold_value:
                    rti = 1
                    record.write(f'{t+1:2d}\t{j+1:2d}\t 1({model.predict_proba(temi)})\t')
                else:
                    rti = 0
                    record.write(f'{t+1:2d}\t{j+1:2d}\t-1({model.predict_proba(temi)})\t')
                if group_label[t] > group_label[j]:
                    record.write('-1\n')
                    record.write('---------------------------------------\n')
                else:
                    record.write('1\n')
                    record.write('---------------------------------------\n')
                if rtd == rti:
                    record.write('there is a conflict!!!!!!!!!!!!!!!!!!!!\n')
        graph.append(edge_set)

def group_test(Data, Label, Ds, Dl, train_index_start, num_of_train, model, threshold_value, top, all_group_top_precision, all_group_top_exact_accuracy, all_group_exact_accuracy, record):
    for group_index_start in range(len(Ds)):
        if group_index_start<train_index_start:
            group_start = Ds[group_index_start]
        elif (group_index_start>=train_index_start and group_index_start<train_index_start+num_of_train):
            continue
        else:
            group_start = Ds[group_index_start]
        length = Dl[group_index_start]
        group_end = group_start + length
        group_data = Data[group_start:group_end, :]
        group_label = Label[group_start:group_end]
        reference = [t for t in range(1, length + 1)]
        random.shuffle(reference)
        predict_rank = rank_the_group(group_data, reference, model, threshold_value, record)
        group_rank = handle_data.exchange(group_label)
        reference = [t for t in range(1, length + 1)]
        record_rank_reference(reference, group_rank, predict_rank, record)
        # --------------------------------------------------------------------
        # group_test_relative(group_data, group_label, length, model, threshold_value, record)
        # --------------------------------------------------------------------
        group_top_precision, group_top_exact_accuracy = count_top(group_rank, predict_rank, top)
        group_exact_accuracy = calacc(group_rank, predict_rank)
        all_group_top_precision.append(group_top_precision)
        all_group_top_exact_accuracy.append(group_top_exact_accuracy)
        all_group_exact_accuracy.append(group_exact_accuracy)
        record.write(f"the group top precision is {group_top_precision}\n")
        record.write(f"the group top exact accuracy is {group_top_exact_accuracy} \n")
        record.write(f"the group accuracy is {group_exact_accuracy}\n")
        record.write("-------------------------------------------------------------------------------------------\n")
    return all_group_top_precision, all_group_top_exact_accuracy, all_group_exact_accuracy


def cal_average(all_group_top_precision, all_group_top_exact_accuracy, all_group_accuracy, record):
    totle = len(all_group_top_precision)
    average_group_top_precision = sum(all_group_top_precision)/totle
    average_group_top_exact_accuracy = sum(all_group_top_exact_accuracy)/totle
    average_group_accuracy = sum(all_group_accuracy)/totle
    record.write(f"\nthe average group top precision is {average_group_top_precision}\n")
    record.write(f"the average group top exact accuracy is {average_group_top_exact_accuracy}\n")
    record.write(f"the average group accuracy is {average_group_accuracy}\n")
    record.write("-------------------------------------------------------------------------------------------\n\n")

