import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import json
import random
import logging
import argparse
from time import time

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist


from model.KerPrint import KerPrint
from utility.parser_kerprint import *
from utility.log_helper import *
from utility.loader_kerprint_new import DataLoaderKerPrint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,auc, roc_curve
from scipy import interp

def get_metrics(predict_all,targets_all,flag,epoch_id,result_path):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
      
      
      
    for i in range(predict_all.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(
            targets_all[:, i], predict_all[:, i])
        fpr[i][np.isnan(fpr[i])] = 0
        tpr[i][np.isnan(tpr[i])] = 0
        roc_auc[i] = auc(fpr[i], tpr[i])
      
    fpr["micro"], tpr["micro"], _ = roc_curve(
        targets_all.ravel(), predict_all.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
      
    all_fpr = np.unique(np.concatenate(
        [fpr[i] for i in range(predict_all.shape[1])]))    
      
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(predict_all.shape[1]):    
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

      
    mean_tpr /= predict_all.shape[1]

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    with open(result_path, "a") as f:
        if flag == "train":
            f.write("train_epoch{}:".format(epoch_id) + str(roc_auc) + '\n')
        elif flag == "val":
            f.write("val_epoch{}:".format(epoch_id) + str(roc_auc) + '\n')
        elif flag == "test":
            f.write("test_epoch{}:".format(epoch_id)+str(roc_auc)+'\n')

    return roc_auc["micro"],roc_auc["macro"]

def evaluate(args,device,model,dataLoader,g_global,result_path,flag,epoch):
    model.eval()

    with torch.no_grad():
        att = model('calc_att', g_global)
    g_global.edata['att'] = att

    y_true_evaluate = []
    y_pred_evaluate = []
    with torch.no_grad():
        for step,(x_batch, s_batch, s_batch_dim2, y_batch,g_personal,miss_batch,batch_mask,batch_mask_final,seq_time_batch,canknowledge_batch) in enumerate(dataLoader.sequence_batch_iter(flag=flag,args=args)):
              
            x_batch = torch.FloatTensor(x_batch).to(device)
            s_batch = torch.LongTensor(s_batch).to(device)
            s_batch_dim2 = torch.Tensor(s_batch_dim2).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)
            g_personal = g_personal.to(device)
            miss_pair = torch.LongTensor(miss_batch).to(device)
              
            seq_time_batch = torch.Tensor(seq_time_batch).to(device).unsqueeze(2) / args.hita_time_scale
            mask_mult = torch.BoolTensor(1-batch_mask).to(device).unsqueeze(2)    
            mask_final = torch.Tensor(batch_mask_final).to(device).unsqueeze(2)    
            canknowledge_batch = torch.LongTensor(canknowledge_batch).to(device)

              
            with torch.no_grad():
                att_time = model('calc_att_time', g_personal)
            g_personal.edata['att_time'] = att_time

            logits = model('predict', g_global, g_personal, x_batch, s_batch, s_batch_dim2,
                           miss_pair, mask_mult, mask_final, seq_time_batch, canknowledge_batch)
            real_logits = torch.sigmoid(logits)

            logits_cpu = real_logits.data.cpu().numpy()
            labels_cpu = y_batch.data.cpu().numpy()
            y_true_evaluate.append(labels_cpu)
            y_pred_evaluate.append(logits_cpu)

    y_true_evaluate = np.vstack(y_true_evaluate)
    y_pred_evaluate = np.vstack(y_pred_evaluate)
    micro_auc, macro_auc = get_metrics(y_pred_evaluate, y_true_evaluate, flag, epoch, result_path)

    return micro_auc,macro_auc


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    use_cuda = torch.cuda.is_available()
    device = torch.device(args.cuda_choice if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    dataLoader = DataLoaderKerPrint(args, logging)
    model =KerPrint(args,dataLoader.kg_node_num,dataLoader.kg_edge_num,dataLoader.move_num,dataLoader.valid_move_num,
                  dataLoader.max_visit_len,miss_EHR_move_num=dataLoader.miss_EHR_move_num)

    model.to(device)
    logging.info(model)
    with open(args.save_dir+"params.json",mode = "w") as f:
        json.dump(args.__dict__,f,indent=4)

    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if ('transformerEncoder' in n)],
          'lr': args.hita_encoder_lr},
        {'params': [p for n, p in model.named_parameters() if (not 'transformerEncoder' in n)],
          'lr': args.base_lr}
    ]
      
    optimizer_1 = optim.Adam(grouped_parameters)
    optimizer_2 = optim.Adam(grouped_parameters)

      
    g_global = dataLoader.g_global
    g_global = g_global.to(device)

      
    result_path = args.save_dir+"metrics.txt"

    best_dev_epoch = 0
    best_dev_auc, final_micro_auc, final_macro_auc = 0.0, 0.0, 0.0
    model_path =os.path.join(args.save_dir, 'model.pt')
    loss_func = nn.BCEWithLogitsLoss(reduction='mean')
    for epoch in range(1,args.n_epoch+1):
          
        time0 = time()
        model.train()

          
        with torch.no_grad():
            att = model('calc_att', g_global)
        g_global.edata['att'] = att
        logging.info('Update attention scores: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

          
        time1 = time()
        sequence_total_loss = 0.0
        miss_total_loss = 0.0

        y_true_train = []
        y_pred_train = []
        for step,(x_batch, s_batch, s_batch_dim2, y_batch,g_personal,miss_batch,batch_mask,batch_mask_final,seq_time_batch,canknowledge_batch) in enumerate(dataLoader.sequence_batch_iter(flag="train",args=args)):
            optimizer_1.zero_grad()

            x_batch = torch.FloatTensor(x_batch).to(device)   
            s_batch = torch.LongTensor(s_batch).to(device)
            s_batch_dim2 = torch.Tensor(s_batch_dim2).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)
            g_personal = g_personal.to(device)
            miss_pair = torch.LongTensor(miss_batch).to(device)
              
            seq_time_batch = torch.Tensor(seq_time_batch).to(device).unsqueeze(2) / args.hita_time_scale
            mask_mult = torch.BoolTensor(1-batch_mask).to(device).unsqueeze(2)    
            mask_final = torch.Tensor(batch_mask_final).to(device).unsqueeze(2)    
            canknowledge_batch = torch.LongTensor(canknowledge_batch).to(device)

              
            with torch.no_grad():
                att_time = model('calc_att_time', g_personal)
            g_personal.edata['att_time'] = att_time

            logits, miss_loss = model('calc_sequence_logits', g_global, g_personal, x_batch, s_batch, s_batch_dim2,
                                       miss_pair,mask_mult,mask_final,seq_time_batch,canknowledge_batch)
            real_logits = torch.sigmoid(logits)
            sequence_loss = loss_func(logits,y_batch.float())
            if "no_miss" not in args.ablation:
                loss = sequence_loss + miss_loss
            else:
                loss = sequence_loss
            loss.backward()
            sequence_total_loss += sequence_loss.item()
            if "no_miss" not in args.ablation:
                miss_total_loss += miss_loss.item()

            if args.clip != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)    
            optimizer_1.step()

            logits_cpu = real_logits.data.cpu().numpy()
            labels_cpu = y_batch.data.cpu().numpy()
            y_true_train.append(labels_cpu)
            y_pred_train.append(logits_cpu)

        y_true_train = np.vstack(y_true_train)
        y_pred_train = np.vstack(y_pred_train)
        micro_auc, macro_auc = get_metrics(y_pred_train, y_true_train, "train", epoch,result_path)

        logging.info(
                'Sequence Training: Epoch {:04d} | Total Time {:.1f}s | Total Loss {:.4f}'.format(epoch,time() - time1,sequence_total_loss))
        if "no_miss" not in args.ablation:
            logging.info(
                'Sequence Training: Epoch {:04d} | Total Time {:.1f}s | Total Miss Loss {:.4f}'.format(epoch,time() - time1,miss_total_loss))
          
        time1= time()
        kg_total_loss =0.0
        for step,(h_batch,r_batch,t_batch,corrupt_batch) in enumerate(dataLoader.kg_batch_iter()):
            optimizer_2.zero_grad()
            h_batch = torch.LongTensor(h_batch).to(device)
            r_batch = torch.LongTensor(r_batch).to(device)
            t_batch = torch.LongTensor(t_batch).to(device)
            corrupt_batch = torch.LongTensor(corrupt_batch).to(device)
            kg_batch_loss = model('calc_kg_loss', h_batch, r_batch, t_batch,
                                  corrupt_batch)
            kg_batch_loss.backward()
            optimizer_2.step()
            kg_total_loss += kg_batch_loss.item()

        logging.info(
            'KG Training: Epoch {:04d} | Total Time {:.1f}s | Total Loss {:.4f}'.format(epoch,time() - time1,kg_total_loss ))

          
        time1 = time()
        dev_micro_auc, dev_macro_auc = evaluate(args, device, model, dataLoader, g_global, result_path, "val",
                                                 epoch)
        logging.info(
            'Val Evaluation: Epoch {:04d} | Total Time {:.1f}s'.format(
                epoch, time() - time1))
        time1 = time()
        test_micro_auc, test_macro_auc = evaluate(args, device, model, dataLoader, g_global, result_path, "test",
                                                   epoch)
        logging.info(
            'Test Evaluation: Epoch {:04d} | Total Time {:.1f}s'.format(
                epoch, time() - time1))
        if dev_macro_auc >= best_dev_auc:
            best_dev_auc = dev_macro_auc
            best_dev_epoch = epoch
            final_micro_auc = test_micro_auc
            final_macro_auc = test_macro_auc
            torch.save([model, args], model_path)
            print(f'model saved to {model_path}')

        logging.info("Epoch: {}".format(epoch))
        logging.info('best test micro auc: {:.4f}'.format(final_micro_auc))
        logging.info('best test macro auc: {:.4f}'.format(final_macro_auc))

        if epoch > args.unfreeze_epoch and epoch - best_dev_epoch >= args.max_epochs_before_stop:
            break

    logging.info('best dev auc: {:.4f} (at epoch {})'.format(best_dev_auc, best_dev_epoch))
    logging.info('final test micro auc: {:.4f}'.format(final_micro_auc))
    logging.info('final test macro auc: {:.4f}'.format(final_macro_auc))


if __name__ == "__main__":
    args = parse_takgat_args()
    train(args)









