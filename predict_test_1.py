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
    # top_true = y_true[:top]
    # top_pred = y_pred[:top]
    # print(y_pred)
    # print(top_pred)
    for i in range(len_top):
        if top_pred[i] in top_true:
            tp += 1
            if top_pred[i] == top_true[i]:
                exact += 1
    group_pro = tp/top
    group_top_exact_accuracy = exact/top
    return group_pro, group_top_exact_accuracy


def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


def general_test(test_data, test_label, model, positive_value, negative_value, threshold_value, record):
    general_predict = model.predict(test_data)
    general_tp, general_fp, general_fn = count_general_pre(test_label, general_predict, positive_value, negative_value, threshold_value)
    general_precision, general_recall = precision_recall(general_tp, general_fp, general_fn)
    general_accuracy = (len(test_label) - general_fp - general_fn) / len(test_label)
    record.write(f"the general precision is {general_precision}\n")
    record.write(f"the general recall is {general_recall}\n")
    record.write(f"the general accuracy is {general_accuracy}\n")
    record.write("-------------------------------------------------------------------------------------------\n")

def rank_the_group(group_data, length, reference, model, record):
    low = []
    high = []
    global handle_type
    if len(reference) <= 1:
        return reference
    else:
        # pivot = reference.pop()
        # print('the reference is ',reference)
        # print('the length is ', length)
        pivot = random.choice(reference)
        # if len(group_data) == length:
        #     record.write(f"the pivot is          {pivot:2d}\n")
        record.write(f"the pivot is          {pivot:2d}\n")
        reference.remove(pivot)
        # print("pivot is ",pivot)
        for i in reference:
            t = handle_data.data_extend(group_data[pivot-1], group_data[i-1])
            t = np.array(t).reshape((1,-1))
            if model.predict(t) > 0:
                low.append(i)
            else:
                high.append(i)
    record.write('low  is ')
    for i in low:
        record.write(f'{i:2d}\t')
    record.write('\n')
    record.write('high is ')
    for i in high:
        record.write(f'{i:2d}\t')
    record.write('\n')
    low = rank_the_group(group_data, len(low), low, model)
    high = rank_the_group(group_data, len(high), high, model)
    high.append(pivot)
    # print('high+low :',high+low)
    return high + low

# def compdata(group_data,model,standscaler,i):
#     global handle_type
#     if handle_type == 'extend':
#         t = data_extend(group_data[i], group_data[i+1])
#     else:
#         t = data_diff(group_data[i], group_data[i+1])
#     t = np.array(t).reshape((1,-1))
#     # print(i)
#     standscaler.transform(t)
#     if model.predict(t) > 0:
#         return True
#     else:
#         return False
#
# def rank_the_group(group_data, length, reference, model, standscaler):
#     if len(reference) <= 1:
#         return reference
#     else:
#         for unsortednum in range(len(reference)-1,0,-1):
#             record.write(f'the unsortednum is {unsortednum:2d}\n')
#             for i in range(unsortednum):
#                 if compdata(group_data,model,standscaler,i):
#                     temp = reference[i]
#                     reference[i] = reference[i+1]
#                     reference[i+1] = temp
#             record.write(f"the {len(reference)-unsortednum:2d} order is       ")
#             for item in reference:
#                 record.write(f'{item:2d}\t')
#             record.write('\n')
#     # print(reference)
#     reference.reverse()
#     # print(reference)
#     return reference

def record_rank_reference(rank, reference, record):
    record.write("                      ")
    for m in range(1,len(rank)+1):
        record.write(f"{m:2d}\t")
    record.write("\n")
    record.write("the true rank is      ")
    for i in rank:
        record.write(f"{int(i):2d}\t")
    record.write("\n")
    record.write("the predict rank is   ")
    for t in reference:
        record.write(f"{int(t):2d}\t")
    record.write("\n")

def test_cycle(graph):
    cycle = []
    handled_set = []
    for i in range(len(graph)):
        dfs(graph,cycle,handled_set,i)

        handled_set.append(i)

def dfs(graph, cycle, handled_set, node, record):
    if node not in handled_set:
        cycle.append(node)
        for edge in range(len(graph[node])):
            if graph[node][edge] == 1:
                if edge == cycle[0]:
                    record.write("there is a cycle:")
                    cycle.append(edge)
                    for item in cycle:
                        # print(item,end='\t')
                        record.write(f'{(item + 1):2d}\t')
                    record.write('\n')
                    # print()
                    cycle.pop()
                    continue
                if edge not in cycle:
                    dfs(graph,cycle,handled_set,edge)
        cycle.pop()

def group_test_relative(group_start, length, Data, Label, model, record):
    global handle_type
    graph = []
    record.write(f'the start line is {group_start}\n')
    for j in range(length):
        edge_set = [-1 for i in range(length)]
        for t in range(length):
            if j != t:
                if handle_type == 'extend':
                    temd = handle_data.data_extend(Data[group_start + j], Data[group_start + t])
                    temi = handle_data.data_extend(Data[group_start + t], Data[group_start + j])
                # else:
                #     temd = data_diff(Data[group_start + j], Data[group_start + t])
                #     temi = data_diff(Data[group_start + t], Data[group_start + j])
                temd = np.array(temd).reshape((1, -1))
                temi = np.array(temi).reshape((1, -1))
                # print(model.predict_proba(temd))
                # print(model.predict_proba(temi))
                if model.predict(temd) > 0:
                    edge_set[t] = 1
                    record.write(f'{j+1:2d}\t{t+1:2d}\t 1\t')
                else:
                    record.write(f'{j+1:2d}\t{t+1:2d}\t-1\t')
                if Label[group_start + j] > Label[group_start + t]:
                    record.write('-1\n')
                else:
                    record.write('1\n')
                if model.predict(temi) > 0:
                    record.write(f'{t+1:2d}\t{j+1:2d}\t 1\t')
                else:
                    record.write(f'{t+1:2d}\t{j+1:2d}\t-1\t')
                if Label[group_start + t] > Label[group_start + j]:
                    record.write('-1\n')
                    record.write('---------------------------------------\n')
                else:
                    record.write('1\n')
                    record.write('---------------------------------------\n')
                if model.predict(temd) == model.predict(temi):
                    record.write('there is a conflict!!!!!!!!!!!!!!!!!!!!\n')
        graph.append(edge_set)
    test_cycle(graph)


def group_test(Data, Label, Ds, Dl, train_index_start, num_of_train, model, record):
    for group_index_start in range(1,len(Ds)+1):
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
        # reference = [t for t in range(1, length + 1)]
        # reference1 = rank_the_group(group_data, length, reference, model)
        # group_test_relative(group_start, length, Data, Label, model)
        for time in range(100):
            reference = [t for t in range(1, length + 1)]
            random.shuffle(reference)
            record.write("the random order is   ")
            for t in reference:
                record.write(f"{int(t):2d}\t")
            record.write("\n")
            # print(reference)
            reference = rank_the_group(group_data, length, reference, model)
            # if not operator.eq(reference,reference1):
            #     # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #     record.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            #     record.write("the predict rank is   ")
            #     for t in reference:
            #         record.write(f"{int(t):2d}\t")
            #     record.write('\n')
            #     record.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

            # reference1 = reference
            # print(reference)
            # record.write("the predict rank is   ")
            # for t in reference:
            #     record.write(f"{int(t):2d}\t")
            # record.write("\n")
        # print(reference)
        group_rank = handle_data.exchange(group_label)
        # print(group_rank)
        record_rank_reference(group_rank,reference)
        group_precision, group_top_exact_accuracy = count_top(group_rank, reference)
        group_accuracy = calacc(group_rank, reference)
        all_group_top_precision.append(group_precision)
        all_group_top_exact_accuracy.append(group_top_exact_accuracy)
        all_group_accuracy.append(group_accuracy)
        record.write(f"the group top precision is {group_precision}\n")
        record.write(f"the group top exact accuracy is {group_top_exact_accuracy} \n")
        record.write(f"the group accuracy is {group_accuracy}\n")
        record.write("-------------------------------------------------------------------------------------------\n")

def calaverage(all_group_top_precision, all_group_top_exact_accuracy, all_group_accuracy, record):
    totle = len(all_group_top_precision)
    average_group_top_precision = sum(all_group_top_precision)/totle
    average_group_top_exact_accuracy = sum(all_group_top_exact_accuracy)/totle
    average_group_accuracy = sum(all_group_accuracy)/totle
    record.write(f"\nthe average group top precision is {average_group_top_precision}\n")
    record.write(f"the average group top exact accuracy is {average_group_top_exact_accuracy}\n")
    record.write(f"the average group accuracy is {average_group_accuracy}\n")
    record.write("-------------------------------------------------------------------------------------------\n\n")

