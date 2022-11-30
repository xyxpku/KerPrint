import argparse
import pickle
import copy
import json
import csv
import pandas as pd
import numpy as np
# from utils.file_com import read_file_2dict, read_file_2list
from itertools import combinations
import os
from skmultilearn.model_selection import IterativeStratification
from sklearn import preprocessing
from scipy.sparse import coo_matrix

middle_file = "./EHR_MIMIC/data_middle.pkl"
quadra_list_file = "./quadra/quadra_list_final_hop3.pkl"
candidate_knowledge_path = "./EHR_MIMIC/patient_knowledge.pkl"

def load_middle_data(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict["patient_visit_list"],data_dict["label_list"],data_dict["time_list"],data_dict["all_pair_list"]

def _label_normalize(label_list):
    enc = preprocessing.MultiLabelBinarizer()
    return enc.fit_transform(label_list)


def iterative_train_test_split(X, y, train_size):
    """Custom iterative train test split which
    'maintains balanced representation with respect
    to order-th label combinations.'
    """
    stratifier = IterativeStratification(
        n_splits=2, order=1, sample_distribution_per_fold=[1.0-train_size, train_size, ])
    train_indices, test_indices = next(stratifier.split(X, y))
    X_train, y_train = np.array(X)[train_indices], np.array(y)[train_indices]
    X_test, y_test = np.array(X)[test_indices], np.array(y)[test_indices]
    return X_train, X_test, y_train, y_test

def build_sequence_array(patient_visit_list,patient_num,max_visit_len,move_num):
    print('building sequence array ...')
    x = []
    lens = np.zeros((patient_num, ), dtype=int)
    lens_dim2 = np.ones((patient_num, max_visit_len),dtype=int)
    mask = np.zeros((patient_num, max_visit_len), dtype=np.float32)
    mask_final = np.zeros((patient_num, max_visit_len), dtype=np.float32)

    for index,patient in enumerate(patient_visit_list):
        row_list = []
        col_list = []
        data_list = []
        for index_visit, codes in enumerate(patient):
            if len(codes) > 0:
                for code in codes:
                    row_list.append(index_visit)
                    col_list.append(code)
                    data_list.append(1.0)
                lens_dim2[index][index_visit] = len(codes)
        lens[index] = len(patient)
        row_array = np.array(row_list)
        col_array = np.array(col_list)
        data_array = np.array(data_list)
        sparse_move_matrix = coo_matrix((data_array, (row_array, col_array)), shape=(max_visit_len, move_num+1))
        x.append(sparse_move_matrix)

    for i in range(patient_num):
        mask[i,0:lens[i]-1]=1
        max_visit = lens[i] - 1
        mask_final[i, max_visit] = 1
    return x,lens,lens_dim2,mask,mask_final

def dump_pickle_file(file_path,data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f,protocol = 4)

def dump_sequence_pkl(sequence_all_array, sequence_len_array, sequence_len_dim2_array,label_array,mask,mask_final,seq_time,flag,dir_path):
    data_dict={}
    data_dict["sequence_all_array"] = sequence_all_array
    data_dict["sequence_len_array"] = sequence_len_array
    data_dict["sequence_len_dim2_array"] = sequence_len_dim2_array
    data_dict["label_array"] = label_array
    data_dict["mask"] = mask
    data_dict["mask_final"] = mask_final
    data_dict["seq_time"] = seq_time
    dump_pickle_file(dir_path+flag+".pkl",data_dict)

def adjust_hita_input(diagnosis_codes, time_step, n_diagnosis_codes):
    time_step = copy.deepcopy(time_step)
    diagnosis_codes = copy.deepcopy(diagnosis_codes)
    for ind in range(len(diagnosis_codes)):
        time_step[ind].append(0)
        diagnosis_codes[ind].append([n_diagnosis_codes])
    return diagnosis_codes, time_step

def pad_time(seq_time_step):
    lengths = np.array([len(seq) for seq in seq_time_step])
    maxlen = np.max(lengths)
    for k in range(len(seq_time_step)):
        while len(seq_time_step[k]) < maxlen:
            seq_time_step[k].append(100000)

    return seq_time_step

if __name__ == "__main__":
    #random split parser
    parser = argparse.ArgumentParser(description="preprocess data for model TAKGAT for renmin dataset.")
    parser.add_argument('--seed', type=int, default=629,
                        help='Random seed.')
    parser.add_argument('--max_node_num', type=int, default=25476,
                        help='Max node num.')
    parser.add_argument('--max_code_num', type=int, default=33355,
                        help='Max code num.')
    args = parser.parse_args()

    model_input_dir_final = "./model_input_data_new_{}/".format(args.seed)
    if not os.path.exists(model_input_dir_final):
        os.makedirs(model_input_dir_final)

    #generate data
    max_node_num = args.max_node_num
    patient_visit_list, label_list, time_list, all_pair_list = load_middle_data(middle_file)
    #add hitanet preprocess
    patient_visit_list,time_list = adjust_hita_input(patient_visit_list,time_list,args.max_code_num)
    print("Example hita data:",patient_visit_list[0])
    time_list = np.array(pad_time(time_list))
    print("Example time list:", time_list[0])

    #original preprocess
    label_list = _label_normalize(label_list)
    max_visit_len = max([len(patient_visit) for patient_visit in patient_visit_list])
    max_visit_move_len = max([len(visit_move) for patient_visit in patient_visit_list for visit_move in patient_visit])
    move_num = max_node_num
    patient_num = len(patient_visit_list)

    print("max_node_num:", max_node_num)
    print("max_visit_len:", max_visit_len)
    print("max_visit_move_len:", max_visit_move_len)
    print("move_num:",move_num)
    print("patient_num:",patient_num)
    miss_EHR_move_num = args.max_code_num - max_node_num -1
    print("miss_EHR_move_num:",miss_EHR_move_num)

    sequence_all_array,sequence_len_array,sequence_len_dim2_array,mask,mask_final = build_sequence_array(patient_visit_list,patient_num,max_visit_len,args.max_code_num) #for hitanet  5580  global_index:5581

    print("splitting train/val/test data ...")

    np.random.seed(args.seed)
    patient_list = list(range(0, patient_num, 1))
    X_train, X_, y_train, y_ = iterative_train_test_split(
        patient_list, label_list, train_size=0.7)
    X_val, X_test, y_val, y_test = iterative_train_test_split(
        X_, y_, train_size=0.5)

    train_index = list(X_train)
    val_index = list(X_val)
    test_index = list(X_test)

    sequence_all_array_train, sequence_len_array_train, sequence_len_dim2_array_train, label_array_train, mask_train, mask_final_train, seq_time_train = \
    [sequence_all_array[i] for i in train_index], sequence_len_array[train_index] \
        , sequence_len_dim2_array[train_index], label_list[train_index], mask[train_index], mask_final[train_index], \
    time_list[train_index]

    sequence_all_array_val, sequence_len_array_val, sequence_len_dim2_array_val, label_array_val, mask_val, mask_final_val, seq_time_val = \
    [sequence_all_array[i] for i in val_index], sequence_len_array[val_index] \
        , sequence_len_dim2_array[val_index], label_list[val_index], mask[val_index], mask_final[val_index], time_list[
        val_index]

    sequence_all_array_test, sequence_len_array_test, sequence_len_dim2_array_test, label_array_test, mask_test, mask_final_test, seq_time_test = \
    [sequence_all_array[i] for i in test_index], sequence_len_array[test_index] \
        , sequence_len_dim2_array[test_index], label_list[test_index], mask[test_index], mask_final[test_index], \
    time_list[test_index]

    dump_sequence_pkl(sequence_all_array_train, sequence_len_array_train, sequence_len_dim2_array_train,label_array_train,mask_train,mask_final_train,seq_time_train,"train",model_input_dir_final)
    dump_sequence_pkl(sequence_all_array_val, sequence_len_array_val, sequence_len_dim2_array_val,label_array_val,mask_val,mask_final_val,seq_time_val,"val",model_input_dir_final)
    dump_sequence_pkl(sequence_all_array_test, sequence_len_array_test, sequence_len_dim2_array_test,label_array_test,mask_test,mask_final_test,seq_time_test,"test",model_input_dir_final)

    dump_pickle_file(model_input_dir_final+"miss_pair.pkl",all_pair_list)
    move_array_dict = {}
    move_array_dict["miss_EHR_move_num"] = miss_EHR_move_num
    dump_pickle_file(model_input_dir_final+"move_dict.pkl",move_array_dict)

    with open(quadra_list_file, 'rb') as f:
        quadra_list = pickle.load(f)

    quadra_train_val_test_dict = {}
    quadra_train_val_test_dict["train"] = [quadra_list[i] for i in train_index]
    quadra_train_val_test_dict["val"] = [quadra_list[i] for i in val_index]
    quadra_train_val_test_dict["test"] = [quadra_list[i] for i in test_index]
    dump_pickle_file(model_input_dir_final+"quadra_train_val_test_dict.pkl",quadra_train_val_test_dict)

    with open(candidate_knowledge_path, 'rb') as f:
        candidate_knowledge_list = pickle.load(f)

    canknowledge_train_val_test_dict = {}
    canknowledge_train_val_test_dict["train"] = [candidate_knowledge_list[i] for i in train_index]
    canknowledge_train_val_test_dict["val"] = [candidate_knowledge_list[i] for i in val_index]
    canknowledge_train_val_test_dict["test"] = [candidate_knowledge_list[i] for i in test_index]
    dump_pickle_file(model_input_dir_final+"canknowledge_train_val_test_dict.pkl",canknowledge_train_val_test_dict)