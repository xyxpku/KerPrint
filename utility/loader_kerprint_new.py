import os
import random
import collections
import pickle

import dgl
import torch
import numpy as np
import pandas as pd
import math

class DataLoaderKerPrint(object):
    def __init__(self, args, logging):
        self.args = args
        self.model_input_data_dir = args.model_input_data_dir     
        self.kg_data_dir = args.kg_data_dir      
        self.pretrain_embedding_dir = args.pretrain_embedding_dir

        self.sequence_batch_size = args.sequence_batch_size
        self.kg_batch_size = args.kg_batch_size

        self.train_sequence_file_path = self.model_input_data_dir + "train.pkl"
        self.val_sequence_file_path = self.model_input_data_dir + "val.pkl"
        self.test_sequence_file_path = self.model_input_data_dir + "test.pkl"
        self.move_dict_file_path = self.model_input_data_dir + "move_dict.pkl"
        self.miss_pair_file_path = self.model_input_data_dir + "miss_pair.pkl"
        self.quadra_file_path =self.model_input_data_dir + "quadra_train_val_test_dict.pkl"
        self.canknowledge_file_path = self.model_input_data_dir + "canknowledge_train_val_test_dict.pkl"

        self.concept_file_path = args.concept_file_path
        self.relation_file_path = args.relation_file_path
        self.triple_file_path = args.triple_file_path
        self.triple_corrupt_file_path = args.triple_corrupt_file_path

          
        self.cuda_choice = args.cuda_choice
        self.graph_type_choice = args.graph_type_choice

        self._load_input_data()
        self._load_kg_data()
        self.g_global = self.create_global_graph()

    def _load_input_data(self):
        with open (self.train_sequence_file_path,"rb") as fin:
            data_dict_train = pickle.load(fin)
        with open (self.val_sequence_file_path,"rb") as fin:
            data_dict_val = pickle.load(fin)
        with open (self.test_sequence_file_path,"rb") as fin:
            data_dict_test = pickle.load(fin)
        with open (self.move_dict_file_path,"rb") as fin:
            move_array_dict = pickle.load(fin)
        with open (self.miss_pair_file_path,"rb") as fin:
            self.miss_pair = pickle.load(fin)
        with open (self.quadra_file_path,"rb") as fin:
            self.quadra_train_val_test_dict = pickle.load(fin)
        with open (self.canknowledge_file_path,"rb") as fin:
            self.canknowledge_train_val_test_dict = pickle.load(fin)
          
        self.sequence_train_array = [data_dict_train["sequence_all_array"],data_dict_train["sequence_len_array"],
                                     data_dict_train["sequence_len_dim2_array"],data_dict_train["label_array"],data_dict_train["mask"],data_dict_train["mask_final"],data_dict_train["seq_time"]]
        self.sequence_val_array = [data_dict_val["sequence_all_array"], data_dict_val["sequence_len_array"],
                                     data_dict_val["sequence_len_dim2_array"], data_dict_val["label_array"],data_dict_val["mask"],data_dict_val["mask_final"],data_dict_val["seq_time"]]
        self.sequence_test_array = [data_dict_test["sequence_all_array"], data_dict_test["sequence_len_array"],
                                     data_dict_test["sequence_len_dim2_array"], data_dict_test["label_array"],data_dict_test["mask"],data_dict_test["mask_final"],data_dict_test["seq_time"]]
        self.miss_EHR_move_num = move_array_dict["miss_EHR_move_num"]
        self.max_visit_len = data_dict_train["sequence_all_array"][0].get_shape()[0]
        self.move_num = data_dict_train["sequence_all_array"][0].get_shape()[1]

        self.sequence_array_dict ={}
        self.sequence_array_dict["train"] = self.sequence_train_array
        self.sequence_array_dict["val"] =self.sequence_val_array
        self.sequence_array_dict["test"] = self.sequence_test_array

    def _load_kg_data(self):
        self.concept_dic = {}
        with open(self.concept_file_path, "r") as f:
            concept_list = f.readlines()
            concept_list = [line.strip() for line in concept_list]
            for line in concept_list[1:]:
                line_list = line.split('\t')
                renmen_node_name = line_list[0]
                item_id = int(line_list[1])
                self.concept_dic[renmen_node_name] = item_id
        self.kg_node_num = len(list(self.concept_dic.keys()))
        self.valid_move_num = self.kg_node_num

        self.relation_dic = {}
        with open(self.relation_file_path, "r") as f:
            relation_list = f.readlines()
            relation_list = [line.strip() for line in relation_list]
            for line in relation_list[1:]:
                line_list = line.split('\t')
                renmen_rel_name = line_list[0]
                item_id = int(line_list[1])
                self.relation_dic[renmen_rel_name] = item_id
        self.kg_edge_num = len(list(self.relation_dic.keys()))    

        self.all_h_list, self.all_t_list, self.all_r_list, self.all_v_list = [], [], [], []
        with open(self.triple_file_path, "r") as f:
            triple_list = f.readlines()
            triple_list = [line.strip() for line in triple_list]
              
            for line in triple_list[1:]:
                sp_list = line.split(' ')
                h_id = int(sp_list[0])
                t_id = int(sp_list[1])
                r_id = int(sp_list[2])
                self.all_h_list.append(h_id)
                self.all_t_list.append(t_id)
                self.all_r_list.append(r_id)
                self.all_v_list.append(1.0)
                  

        self.all_corrupt_list = []
        with open(self.triple_corrupt_file_path, "r") as f:
            triple_corrupt_list = f.readlines()
            triple_corrupt_list = [line.strip() for line in triple_corrupt_list]
            for line in triple_corrupt_list[1:]:
                sp_list = line.split(' ')
                h_id = int(sp_list[0])
                corrt_id = int(sp_list[1])
                r_id = int(sp_list[2])
                self.all_corrupt_list.append(corrt_id)

    def sequence_batch_iter(self,flag,args,shuffle = False):
        sequence_all_array = self.sequence_array_dict[flag]
        quadra_list = self.quadra_train_val_test_dict[flag]
          
        canknowledge_list = self.canknowledge_train_val_test_dict[flag]

        patient_num  = len(sequence_all_array[0])
        batch_num = math.ceil(patient_num / self.sequence_batch_size)
        miss_batch_size = math.floor(len(self.miss_pair) / batch_num)      
        index_array = list(range(patient_num))

        if shuffle:
            np.random.shuffle(index_array)

        for i in range(batch_num):
            indices = index_array[i * self.sequence_batch_size: (i + 1) * self.sequence_batch_size]
            x_batch_origin = sequence_all_array[0][i * self.sequence_batch_size: (i + 1) * self.sequence_batch_size]
            x_batch = np.array([sparse_matrix.toarray() for sparse_matrix in x_batch_origin])
            s_batch = sequence_all_array[1][indices]
            s_batch_dim2 = sequence_all_array[2][indices]
            y_batch = sequence_all_array[3][indices]
            q_batch = quadra_list[i * self.sequence_batch_size: (i + 1) * self.sequence_batch_size]
            batch_mask = sequence_all_array[4][indices]
            batch_mask_final = sequence_all_array[5][indices]
            seq_time_batch = sequence_all_array[6][indices]
            miss_batch = self.miss_pair[i*miss_batch_size: (i+1)*miss_batch_size]   
            canknowledge_batch = canknowledge_list[i * self.sequence_batch_size: (i + 1) * self.sequence_batch_size]
              

            h_batch = []
            r_batch = []
            t_batch = []
            time_batch = []
            pair_batch = []
              
            for patient in q_batch:
                for quadra in patient:
                    pair_tmp = []
                    h_batch.append(quadra[0])
                    r_batch.append(quadra[1])
                    t_batch.append(quadra[2])
                    time_batch.append(quadra[3])
                      
                      
                      
                      
                    pair_tmp.append(quadra[0])
                    pair_tmp.append(quadra[2])
                    pair_batch.append(pair_tmp)
            g_personal = self.create_batch_graph(h_batch,r_batch,t_batch,time_batch)

            yield x_batch, s_batch, s_batch_dim2, y_batch,g_personal,miss_batch,batch_mask,batch_mask_final,seq_time_batch,canknowledge_batch

    def kg_batch_iter(self,shuffle = False):
        triple_num = len(self.all_h_list)
        batch_num = math.ceil(triple_num / self.kg_batch_size)
        index_array = list(range(batch_num))

        if shuffle:
            np.random.shuffle(index_array)

        for i in range(batch_num):
            h_batch = self.all_h_list[i * self.kg_batch_size: (i + 1) * self.kg_batch_size]
            r_batch = self.all_r_list[i * self.kg_batch_size: (i + 1) * self.kg_batch_size]
            t_batch = self.all_t_list[i * self.kg_batch_size: (i + 1) * self.kg_batch_size]
            corrupt_batch = self.all_corrupt_list[i * self.kg_batch_size: (i + 1) * self.kg_batch_size]

            yield h_batch,r_batch,t_batch,corrupt_batch


    def create_global_graph(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.kg_node_num+1)
        g.add_edges(torch.tensor(self.all_h_list),torch.tensor(self.all_t_list))
        g.ndata['id'] = torch.arange(self.kg_node_num+1, dtype=torch.long)
        g.edata['type'] = torch.LongTensor(self.all_r_list)
        g.to(self.cuda_choice)
        return g

    def create_batch_graph(self,h_batch,r_batch,t_batch,time_batch):
          
        g_batch = dgl.DGLGraph()
        g_batch.add_nodes(self.kg_node_num+1)
        g_batch.add_edges(torch.tensor(h_batch),torch.tensor(t_batch))
        g_batch.ndata['id'] = torch.arange(self.kg_node_num+1, dtype=torch.long)
        g_batch.edata['type'] = torch.LongTensor(r_batch)
        g_batch.edata['time'] = torch.LongTensor(time_batch)
        return g_batch

