import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.softmax import edge_softmax
from utility.helper import edge_softmax_fix
import math
import numpy as np
from torch.autograd import Variable
from model.transformer_hita import TransformerTime

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)    


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if aggregator_type == 'gcn':
            self.W = nn.Linear(self.in_dim, self.out_dim)         
        elif aggregator_type == 'graphsage':
            self.W = nn.Linear(self.in_dim * 2, self.out_dim)     
        elif aggregator_type == 'bi-interaction':
            self.W1 = nn.Linear(self.in_dim, self.out_dim)        
            self.W2 = nn.Linear(self.in_dim, self.out_dim)        
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()
        self._init_weight(aggregator_type)

    def _init_weight(self,aggregator_type):
        if aggregator_type == 'gcn':
            nn.init.xavier_uniform_(self.W.weight)
            nn.init.zeros_(self.W.bias)
        elif aggregator_type == 'graphsage':
            nn.init.xavier_uniform_(self.W.weight)
            nn.init.zeros_(self.W.bias)
        elif aggregator_type == 'bi-interaction':
            nn.init.xavier_uniform_(self.W1.weight)
            nn.init.xavier_uniform_(self.W2.weight)
            nn.init.zeros_(self.W1.bias)
            nn.init.zeros_(self.W2.bias)
        else:
            raise NotImplementedError

    def forward(self, mode, g, entity_embed):
        g = g.local_var()
        g.ndata['node'] = entity_embed

        if mode == 'predict':
            g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), dgl.function.sum('side', 'N_h'))

        else:
            g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), dgl.function.sum('side', 'N_h'))

        if self.aggregator_type == 'gcn':
            out = self.activation(self.W(g.ndata['node'] + g.ndata['N_h']))

        elif self.aggregator_type == 'graphsage':
            out = self.activation(self.W(torch.cat([g.ndata['node'], g.ndata['N_h']], dim=1)))

        elif self.aggregator_type == 'bi-interaction':
            out1 = self.activation(self.W1(g.ndata['node'] + g.ndata['N_h']))
            out2 = self.activation(self.W2(g.ndata['node'] * g.ndata['N_h']))
            out = out1 + out2
        else:
            raise NotImplementedError

        out = self.message_dropout(out)
        return out

class TA_Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(TA_Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if aggregator_type == 'gcn':
            self.W = nn.Linear(self.in_dim, self.out_dim)
        elif aggregator_type == 'graphsage':
            self.W = nn.Linear(self.in_dim * 2, self.out_dim)
        elif aggregator_type == 'bi-interaction':
            self.W1 = nn.Linear(self.in_dim, self.out_dim)
            self.W2 = nn.Linear(self.in_dim, self.out_dim)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()
        self._init_weight(aggregator_type)

    def _init_weight(self,aggregator_type):
        if aggregator_type == 'gcn':
            nn.init.xavier_uniform_(self.W.weight)
            nn.init.zeros_(self.W.bias)
        elif aggregator_type == 'graphsage':
            nn.init.xavier_uniform_(self.W.weight)
            nn.init.zeros_(self.W.bias)
        elif aggregator_type == 'bi-interaction':
            nn.init.xavier_uniform_(self.W1.weight)
            nn.init.xavier_uniform_(self.W2.weight)
            nn.init.zeros_(self.W1.bias)
            nn.init.zeros_(self.W2.bias)
        else:
            raise NotImplementedError

    def forward(self, mode, g, entity_embed):
        g = g.local_var()
        g.ndata['node'] = entity_embed

        if mode == 'predict':
            g.update_all(dgl.function.u_mul_e('node', 'att_time', 'side_time'),
                         dgl.function.sum('side_time', 'N_h_time'))
        else:
            g.update_all(dgl.function.u_mul_e('node', 'att_time', 'side_time'), dgl.function.sum('side_time', 'N_h_time'))

        if self.aggregator_type == 'gcn':
              
            out = self.activation(self.W(g.ndata['node'] + g.ndata['N_h_time']))

        elif self.aggregator_type == 'graphsage':
              
            out = self.activation(self.W(torch.cat([g.ndata['node'], g.ndata['N_h_time']], dim=1)))

        elif self.aggregator_type == 'bi-interaction':
              
            out1 = self.activation(self.W1(g.ndata['node'] + g.ndata['N_h_time']))
            out2 = self.activation(self.W2(g.ndata['node'] * g.ndata['N_h_time']))
            out = out1 + out2
        else:
            raise NotImplementedError

        out = self.message_dropout(out)
        return out

class MLP(nn.Module):
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh , 'leakyrelu':nn.LeakyReLU}

    def __init__(self, input_size, hidden_size_list, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='leakyrelu',use_dropout = False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.use_dropout = use_dropout

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size_list[i-1]
            n_out = self.hidden_size_list[i] if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                if use_dropout:
                    self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

          
        def init_weight(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.layers.apply(init_weight)

    def forward(self, input):
        return self.layers(input)

class KerPrint(nn.Module):

    def __init__(self, args,
                 node_num,relation_num, move_num, valid_move_num, max_visit_len,miss_EHR_move_num = 1447,miss_start_index = 4134
                 ):

        super(KerPrint, self).__init__()
        self.hita_time_scale  = args.hita_time_scale
        self.device = torch.device(args.cuda_choice if torch.cuda.is_available() else "cpu")
        self.node_num = node_num
        self.relation_num = relation_num
        self.move_num = move_num
        self.valid_move_num = valid_move_num   
        self.miss_EHR_move_num = miss_EHR_move_num
        self.miss_start_index = miss_start_index
        self.ablation = args.ablation
        self.max_visit_len = max_visit_len
        print("move_num：", self.move_num)
        print("valid_move_num：", self.valid_move_num)
        print("max_visit_len：",self.max_visit_len)

        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim

        self.aggregation_type = args.aggregation_type
        self.kgat_dim_list = [args.entity_dim] + eval(args.kgat_dim_list)
        self.kgat_mess_dropout = [args.kgat_mess_dropout] * len(eval(args.kgat_dim_list))
        self.n_layers_KGAT = len(eval(args.kgat_dim_list))

        self.takgat_dim_list = [args.entity_dim] + eval(args.takgat_dim_list)    
        self.takgat_mess_dropout = [args.takgat_mess_dropout] * len(eval(args.takgat_dim_list))
        self.n_layers_TAKGAT = len(eval(args.takgat_dim_list))

          
        self.train_dropout_rate = args.hita_dropout_prob
        self.hita_input_size = np.sum(np.array(self.takgat_dim_list))
        self.hita_time_selection_layer_global = [args.hita_time_selection_layer_global_embed,args.global_query_size]
        self.hita_time_selection_layer_encoder = [args.hita_time_selection_layer_encoder_embed,self.hita_input_size]
        self.transformerEncoder = TransformerTime(args,self.hita_input_size,self.max_visit_len,self.move_num,
                                                  self.hita_time_selection_layer_global, self.hita_time_selection_layer_encoder)


        self.classfier_fc = nn.Linear(self.hita_input_size+self.kgat_dim_list[-1],args.label_num)

          
        self.delta_MLP_embedsize = eval(args.delta_MLP_embedsize)
        self.time_MLP_embedsize = eval(args.time_MLP_embedsize)
        self.time_MLP_layer_num =  len(eval(args.time_MLP_embedsize)) - 1
        self.time_MLP = MLP(input_size=2*self.relation_dim + eval(args.delta_MLP_embedsize)[-1], hidden_size_list=self.time_MLP_embedsize,output_size = 1,num_layers = self.time_MLP_layer_num,dropout = self.train_dropout_rate,layer_norm=False,use_dropout=False)

          
        self.kg_l2loss_lambda = args.kg_l2loss_lambda

          
        self.loss_func = nn.BCEWithLogitsLoss(reduction="mean")

          
        self.relation_embed = nn.Embedding(self.relation_num+1, self.relation_dim,padding_idx=0)
        self.node_embed = nn.Embedding(self.node_num+1, self.entity_dim,padding_idx=0)

        self.W_R = nn.Parameter(torch.Tensor(self.relation_num+1, self.entity_dim, self.relation_dim))
        self.miss_embedding = nn.Parameter(torch.Tensor(self.miss_EHR_move_num,np.sum(np.array(self.takgat_dim_list))))
        self.global_embedding = nn.Parameter(torch.Tensor(1,np.sum(np.array(self.takgat_dim_list))))   
        self.element_weight_matrix = nn.Parameter(torch.Tensor(args.candidate_knowledge_num,self.kgat_dim_list[-1],self.kgat_dim_list[-1]))
        self.patient_transform_fc = nn.Linear(self.hita_input_size, self.kgat_dim_list[-1])

        self.time_interval_layer = nn.Linear(1,self.delta_MLP_embedsize[0])
        self.time_feature_layer = nn.Linear(self.delta_MLP_embedsize[0],self.delta_MLP_embedsize[1])
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(self.train_dropout_rate)
        self._init_weight()

        self.aggregator_layers_KGAT = nn.ModuleList()
        for k in range(self.n_layers_KGAT):
            self.aggregator_layers_KGAT.append(Aggregator(self.kgat_dim_list[k], self.kgat_dim_list[k + 1], self.kgat_mess_dropout[k], self.aggregation_type))

        self.aggregator_layers_TAKGAT = nn.ModuleList()
        for k in range(self.n_layers_TAKGAT):
            self.aggregator_layers_TAKGAT.append(TA_Aggregator(self.takgat_dim_list[k], self.takgat_dim_list[k + 1], self.takgat_mess_dropout[k], self.aggregation_type))

    def _init_weight(self):
        nn.init.xavier_uniform_(self.classfier_fc.weight)
        nn.init.zeros_(self.classfier_fc.bias)
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.miss_embedding, gain=nn.init.calculate_gain('relu'))

          
        nn.init.xavier_uniform_(self.global_embedding, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.element_weight_matrix, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.patient_transform_fc.weight)
        nn.init.zeros_(self.patient_transform_fc.bias)

        nn.init.xavier_uniform_(self.time_interval_layer.weight)
        nn.init.zeros_(self.time_interval_layer.bias)
        nn.init.xavier_uniform_(self.time_feature_layer.weight)
        nn.init.zeros_(self.time_feature_layer.bias)
        nn.init.xavier_uniform_(self.node_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)

    def att_score(self, edges):
          
        r_mul_h = torch.matmul(self.node_embed(edges.src['id']), self.W_r)                         
        r_mul_t = torch.matmul(self.node_embed(edges.dst['id']), self.W_r)                         
        r_embed = self.relation_embed(edges.data['type'])                                                 
        att = torch.bmm(r_mul_t.unsqueeze(1), torch.tanh(r_mul_h + r_embed).unsqueeze(2)).squeeze(-1)     
        return {'att': att}

    def att_score_time(self,edges):
        rel_index = edges.data['type'].unsqueeze(1).unsqueeze(2)
        rel_index = rel_index.expand(rel_index.size(0),self.entity_dim,self.relation_dim)
        W_r = torch.gather(self.W_R,0,rel_index)
        r_mul_h = torch.bmm(self.node_embed(edges.src['id']).unsqueeze(1), W_r).squeeze(1)                         
        r_mul_t = torch.bmm(self.node_embed(edges.dst['id']).unsqueeze(1), W_r).squeeze(1)                   
        r_embed = self.relation_embed(edges.data['type'])                                          
        time_interval = edges.data['time'].unsqueeze(1) / self.hita_time_scale      
        time_feature = 1 - self.tanh(torch.pow(self.time_interval_layer(time_interval),2))
        time_feature = self.time_feature_layer(time_feature)
        all_feature = torch.cat([r_mul_h,r_mul_t,time_feature],dim=1)
        att_time = self.time_MLP(all_feature)
        return {'att_time': att_time}


    def compute_attention(self, g):
        g = g.local_var()
        for i in range(1,self.relation_num+1):
            edge_idxs = g.filter_edges(lambda edge: edge.data['type'] == i)
            self.W_r = self.W_R[i]
            g.apply_edges(self.att_score, edge_idxs)

        g.edata['att'] = edge_softmax_fix(g, g.edata.pop('att'))
        return g.edata.pop('att')

    def compute_attention_time(self, g):
        g = g.local_var()
        g.apply_edges(self.att_score_time)
        g.edata['att_time'] = edge_softmax_fix(g, g.edata.pop('att_time'))
        return g.edata.pop('att_time')


    def calc_kg_loss(self, h, r, pos_t, neg_t):
        r_embed = self.relation_embed(r)                   
        W_r = self.W_R[r]                                  

        h_embed = self.node_embed(h)                
        pos_t_embed = self.node_embed(pos_t)        
        neg_t_embed = self.node_embed(neg_t)        

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)               
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)       
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)       

          
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)       
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)       

          
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def calc_move_embedding(self,node_embedding):
        replace_embed,rest_embed = torch.split(node_embedding,[1,node_embedding.size(0)-1],dim=0)
        zero_embed_for_node = torch.zeros_like(replace_embed,requires_grad=False)
        all_embed = torch.cat((zero_embed_for_node,rest_embed),dim=0)

          
        move_embedding_final = torch.cat([all_embed,self.miss_embedding,self.global_embedding],dim=0)   

        return move_embedding_final

    def calc_visit_embedding(self,move_embedding,x_batch,s_batch_dim2):
        replace_embed, rest_embed = torch.split(move_embedding, [1, move_embedding.size(0) - 1], dim=0)
        zero_embed_for_move = torch.zeros_like(replace_embed, requires_grad=False)
        move_embed = torch.cat((zero_embed_for_move,rest_embed),dim=0)

        sequence_embedding = torch.matmul(x_batch,move_embed)

        s_batch_dim2 = s_batch_dim2.unsqueeze(2)    
        sequence_embedding = torch.div(sequence_embedding,s_batch_dim2)
        return sequence_embedding

    def calc_knowledge_embed(self,sequence_embedding_final,canknowledge_embed):
        candidate_knowledge_num = canknowledge_embed.size(1)
        patient_element_matrix = sequence_embedding_final.unsqueeze(2).expand(-1, -1, sequence_embedding_final.size(1))
        patient_element_matrix_expand = patient_element_matrix.unsqueeze(1).expand(-1, candidate_knowledge_num, -1, -1)
        knowledge_element_matrix = canknowledge_embed.unsqueeze(2).expand(-1, -1,
                                                                                 sequence_embedding_final.size(1), -1)
        interaction_matrix = patient_element_matrix_expand * knowledge_element_matrix + patient_element_matrix_expand - knowledge_element_matrix
        element_attention_matrix = torch.softmax(torch.sum(interaction_matrix * self.element_weight_matrix, dim=2), dim=1)
        result_embedding = torch.sum(canknowledge_embed * element_attention_matrix, dim=1)
        return result_embedding,element_attention_matrix


    def calc_sequence_embedding(self, mode,  g_global, g_personal , x_batch ,s_batch ,
                                s_batch_dim2 , batch_mask, batch_mask_final, seq_time_batch, canknowledge_batch):
        g_global = g_global.local_var()
        g_personal = g_personal.local_var()

          
        ego_embed = self.node_embed(g_global.ndata['id'])
        all_embed = [ego_embed]

        for i,layer in enumerate(self.aggregator_layers_TAKGAT):
            ego_embed = layer(mode, g_personal, ego_embed)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)
        all_embed = torch.cat(all_embed, dim=1)

          
        move_embedding_all = self.calc_move_embedding(all_embed)

          
        sequence_embedding = self.calc_visit_embedding(move_embedding_all,x_batch,s_batch_dim2)

        sequence_embedding_final,self_weight = self.transformerEncoder(sequence_embedding, seq_time_batch, batch_mask,batch_mask_final,s_batch,self.device)     

          
          
        global_node_embed = self.node_embed(g_global.ndata['id'])
        kgat_embed = []

        for i, layer in enumerate(self.aggregator_layers_KGAT):
            global_node_embed = layer(mode, g_global, global_node_embed)
            norm_embed = F.normalize(global_node_embed, p=2, dim=1)
            kgat_embed = [norm_embed]
        kgat_embed = torch.cat(kgat_embed, dim=0)

        canknowledge_index = canknowledge_batch.unsqueeze(2).expand(-1,-1,kgat_embed.size(1))
        kgat_embed = kgat_embed.expand(canknowledge_index.size(0), kgat_embed.size(0), kgat_embed.size(1))
        canknowledge_embed = torch.gather(kgat_embed,1,canknowledge_index)
        sequence_embedding_transform = self.relu(self.patient_transform_fc(sequence_embedding_final))
        knowledge_embed,element_attention_matrix = self.calc_knowledge_embed(sequence_embedding_transform,canknowledge_embed)

        sequence_embedding_all = torch.cat((sequence_embedding_final,knowledge_embed),dim=1)
        sequence_embedding_all = self.dropout(sequence_embedding_all)

        return sequence_embedding_all,move_embedding_all

    def calc_sequence_logits(self, mode, g_global, g_personal, x_batch, s_batch, s_batch_dim2, miss_pair,
                             batch_mask, batch_mask_final, seq_time_batch, canknowledge_batch):

        all_sequence_embed,move_embedding_all = self.calc_sequence_embedding(mode,  g_global, g_personal , x_batch ,s_batch , s_batch_dim2 ,
                                                                              batch_mask, batch_mask_final, seq_time_batch, canknowledge_batch)    
          
        all_sequence_logits = self.classfier_fc(all_sequence_embed)      
          
        if mode == "calc_sequence_logits" and "no_miss" not in self.ablation:
            miss_loss = self.calc_miss_loss(miss_pair,move_embedding_all)
            return all_sequence_logits,miss_loss
        elif mode == "calc_sequence_logits" and "no_miss" in self.ablation:
            miss_loss = 0.0
            return all_sequence_logits, miss_loss
        else:
            return all_sequence_logits

    def calc_miss_loss(self,miss_pair,move_embedding_all):
        pair_valid,pair_invalid = torch.split(miss_pair,[1,1],dim=1)
        idx_valid = pair_valid.expand(pair_valid.size(0),move_embedding_all.size(1))
        idx_invalid = pair_invalid.expand(pair_invalid.size(0), move_embedding_all.size(1))
        embedding_valid = torch.gather(move_embedding_all, 0, idx_valid)
        embedding_invalid = torch.gather(move_embedding_all, 0, idx_invalid)
        embedding_miss = self.miss_embedding.permute(1,0)
        embedding_pair_product = torch.exp(torch.sum(embedding_valid * embedding_invalid,dim=1,keepdim=True))
        embedding_miss_product = torch.sum(torch.exp(torch.mm(embedding_valid,embedding_miss)),dim=1,keepdim=True)
        loss_vector = torch.log(torch.div(embedding_pair_product,embedding_miss_product))
        loss = - torch.sum(loss_vector) / miss_pair.size(0)
        return loss


    def forward(self, mode, *input):
        if mode == 'calc_att':
            return self.compute_attention(*input)
        if mode == 'calc_att_time':
            return self.compute_attention_time(*input)
        if mode == 'calc_sequence_logits':
            return self.calc_sequence_logits(mode, *input)
        if mode == 'calc_kg_loss':
            return self.calc_kg_loss(*input)
        if mode == 'predict':
            return self.calc_sequence_logits(mode, *input)
        if mode == 'calc_miss_loss':
            return self.calc_miss_loss(*input)

