import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.init as init
import copy

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=256, num_heads=4, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention

class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=256, ffn_dim=1024, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, model_dim=256, num_heads=4, ffn_dim=1024, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):

        super(PositionalEncoding, self).__init__()

        self.max_seq_len = max_seq_len
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(self.max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding.astype(np.float32))

        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        self.position_encoding = nn.Embedding(self.max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len, device):


        # max_len = torch.max(input_len)
        # tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor

        pos = np.zeros([len(input_len), self.max_seq_len])
        for ind, length in enumerate(input_len):
            for pos_ind in range(1, length + 1):
                pos[ind, pos_ind - 1] = pos_ind
        input_pos = torch.LongTensor(pos).to(device)
        return self.position_encoding(input_pos), input_pos

def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask

class EncoderNew(nn.Module):
    def __init__(self,args,hita_input_size,max_seq_len,move_num,hita_time_selection_layer_encoder,
                 dropout=0.0):
        super(EncoderNew, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(hita_input_size, args.hita_encoder_head_num, args.hita_encoder_ffn_size, dropout) for _ in
             range(args.hita_encoder_layer)])

        self.bias_embedding = torch.nn.Parameter(torch.Tensor(hita_input_size))
        bound = 1 / math.sqrt(move_num)
        init.uniform_(self.bias_embedding, -bound, bound)

        # self.weight_layer = torch.nn.Linear(model_dim, 1)
        self.pos_embedding = PositionalEncoding(hita_input_size, max_seq_len)
        self.hita_time_selection_layer_encoder = hita_time_selection_layer_encoder
        self.time_layer = torch.nn.Linear(self.hita_time_selection_layer_encoder[0], self.hita_time_selection_layer_encoder[1])
        self.selection_layer = torch.nn.Linear(1, self.hita_time_selection_layer_encoder[0])
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, sequence_embedding, seq_time_step, mask_mult, lengths,device):
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature)
        output = sequence_embedding + self.bias_embedding
        output += time_feature
        output_pos, ind_pos = self.pos_embedding(lengths.unsqueeze(1),device)
        # print(output_pos)
        # print(output_pos.shape)
        output += output_pos
        self_attention_mask = padding_mask(ind_pos, ind_pos)

        attentions = []
        outputs = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask) #attention size(ffn_size,max_seq_len,max_seq_len)
            attentions.append(attention)
            outputs.append(output)
        # weight = torch.softmax(self.weight_layer(outputs[-1]), dim=1)
        # weight = weight * mask - 255 * (1 - mask)
        return output

class TransformerTime(nn.Module):
    def __init__(self, args, hita_input_size,max_seq_len,move_num,hita_time_selection_layer_global,hita_time_selection_layer_encoder):
        super(TransformerTime, self).__init__()
        # self.prior_encoder = PriorEncoder(batch_size, options)
        self.time_encoder = TimeEncoder(args,hita_time_selection_layer_global)
        self.feature_encoder = EncoderNew(args, hita_input_size, max_seq_len, move_num,hita_time_selection_layer_encoder)
        self.self_layer = torch.nn.Linear(hita_input_size, 1)
        # self.classify_layer = torch.nn.Linear(256, 2)
        self.quiry_layer = torch.nn.Linear(hita_input_size, args.global_query_size)
        self.quiry_weight_layer = torch.nn.Linear(hita_input_size, 2)
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = args.hita_dropout_prob
        self.dropout = nn.Dropout(dropout_rate)

    def get_self_attention(self, features, query, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill(mask, -np.inf), dim=1)
        # attention = torch.sum(key * query, 2, keepdim=True) / 8
        return attention

    def forward(self, sequence_embedding, seq_time_step, mask_mult,mask_final,lengths,device):
        # sequence_embedding: [batch_size, length, embedding_size]
        # seq_time_step: [batch_size, length] the day times to the final visit
        #features size:(batch_size, max_seq_len, hita_input_size)
        #mask_final size:(batch_size, max_seq_len, 1)
        features = self.feature_encoder(sequence_embedding, seq_time_step, mask_mult, lengths,device)
        final_statues = features * mask_final
        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues)) #(batch_size,1,global_qurey_size)

        # prior_weight = self.prior_encoder(seq_dignosis_codes, maxlen, quiryes, options, mask_mult)
        self_weight = self.get_self_attention(features, quiryes, mask_mult) #(batch_size,max_visit_len,1) #no global
        time_weight = self.time_encoder(seq_time_step, quiryes, mask_mult) #(batch_size,max_visit_len,1)
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2)

        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        # averaged_features = self.dropout(averaged_features) #after canknowledge select
        # predictions = self.classify_layer(averaged_features)
        # labels = torch.LongTensor(labels)
        # if options['use_gpu']:
        #     labels = labels.cuda()
        return averaged_features, self_weight

class TimeEncoder(nn.Module):
    def __init__(self,args,hita_time_selection_layer_global):
        super(TimeEncoder, self).__init__()
        self.hita_time_selection_layer_global = hita_time_selection_layer_global
        self.selection_layer = torch.nn.Linear(1, self.hita_time_selection_layer_global[0])
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight_layer = torch.nn.Linear(self.hita_time_selection_layer_global[0], self.hita_time_selection_layer_global[1])

    def forward(self, seq_time_step, final_queries, mask):
        selection_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        selection_feature = self.relu(self.weight_layer(selection_feature))
        selection_feature = torch.sum(selection_feature * final_queries, 2, keepdim=True) / (self.hita_time_selection_layer_global[1] ** 0.5)
        selection_feature = selection_feature.masked_fill_(mask, -np.inf)
        # time_weights = self.weight_layer(selection_feature)
        return torch.softmax(selection_feature, 1)