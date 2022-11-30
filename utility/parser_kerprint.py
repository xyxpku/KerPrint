import argparse
import datetime

def parse_takgat_args():
    parser = argparse.ArgumentParser(description="Run TAKGAT for renmin.")

    parser.add_argument('--seed', type=int, default=629,
                        help='Random seed.')

      

    parser.add_argument('--model_input_data_dir', nargs='?', default='./data_preprocess/model_input_data_new_629/',
                        help='Input model input data path')
    parser.add_argument('--kg_data_dir', nargs='?', default='./data_preprocess/graph_preprocessed/',
                        help='Input kg data path.')

    parser.add_argument('--use_pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth',
                        help='Path of stored model.')

    parser.add_argument('--concept_file_path', nargs='?', default='./data_preprocess/graph_preprocessed/entity2id_1hop.txt',
                        help='Path of concept file.')
    parser.add_argument('--relation_file_path', nargs='?', default='./data_preprocess/graph_preprocessed/relation2id_1hop.txt',
                        help='Path of relation file.')
    parser.add_argument('--triple_file_path', nargs='?', default='./data_preprocess/graph_preprocessed/train2id_1hop.txt',
                        help='Path of triple file.')
    parser.add_argument('--triple_corrupt_file_path', nargs='?', default='./data_preprocess/graph_preprocessed/train2id_1hop_corrupt.txt',
                        help='Path of corrupt triple file.')

    parser.add_argument('--cuda_choice', nargs='?', default='cuda:0',
                        help='GPU choice.')
    parser.add_argument('--ablation', default=[],
                        choices=['no_miss','no_time_attention','no_global_knowledge'], nargs='*', help='run ablation test')


    parser.add_argument('--sequence_batch_size', type=int, default=256,
                        help='sequence batch size.')
    parser.add_argument('--kg_batch_size', type=int, default=256,
                        help='KG batch size.')

    parser.add_argument('--entity_dim', type=int, default=64,
                        help='Node Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=96,
                        help='Relation Embedding size.')

    parser.add_argument('--aggregation_type', nargs='?', default='gcn',
                        help='Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}.')
    parser.add_argument('--kgat_dim_list', nargs='?', default='[64]',
                        help='Output sizes of every aggregation layer in KGAT.')
    parser.add_argument('--kgat_mess_dropout', type=float, default=0.5,
                        help='Dropout probability for kgatw.r.t. message dropout for each deep layer. 0: no dropout.')
    parser.add_argument('--takgat_dim_list', nargs='?', default='[64,64,64]',
                        help='Output sizes of every aggregation layer in TAKGAT.')
    parser.add_argument('--takgat_mess_dropout', type=float, default=0.5,
                        help='Dropout probability for takgat w.r.t. message dropout for each deep layer. 0: no dropout.')


    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--clip', type=int, default=5,
                        help='Clip Value for gradient.')

    parser.add_argument('--hita_encoder_lr', type=float, default=0.0001,
                        help='Hita Encoder Learning rate.')
    parser.add_argument('--base_lr', type=float, default=0.001,
                        help='Base Learning rate.')


    parser.add_argument('--n_epoch', type=int, default=400,
                        help='Number of epoch.')
    parser.add_argument('--label_num', type=int, default=150,
                        help='Number of multi label.')
    parser.add_argument('--delta_MLP_embedsize', nargs='?', default='[64,64]',
                        help='Output sizes of delta.')
    parser.add_argument('--time_MLP_embedsize', nargs='?', default='[64,32,1]',
                        help='Output sizes of every MLP layer in Time-attention.')


      
    parser.add_argument('--hita_dropout_prob', type=float, default=0.5,
                        help='HitaNet Dropout rate')
    parser.add_argument('--hita_encoder_layer', type=int, default=1,
                        help='HitaNet Encoder layer')
    parser.add_argument('--hita_encoder_head_num', type=int, default=4,
                        help='HitaNet Encoder Head Num')
    parser.add_argument('--hita_encoder_ffn_size', type=int, default=1024,
                        help='HitaNet Encoder ffn size')
    parser.add_argument('--global_query_size', type=int, default=64,
                        help='HitaNet Global query size.')
    parser.add_argument('--hita_time_selection_layer_global_embed', type=int, default=64,
                        help='HitaNet Global time selection layer.')
    parser.add_argument('--hita_time_selection_layer_encoder_embed', type=int, default=64,
                        help='HitaNet Encoder time selection layer.')
    parser.add_argument('--hita_time_scale', type=int, default=28,
                        help='HitaNet Time Scale')

    parser.add_argument('--candidate_knowledge_num', type=int, default=180,
                        help='Candidate Knowledge Num.')

    parser.add_argument('--cf_print_every', type=int, default=1,
                        help='Iter interval of printing sequence loss.')
    parser.add_argument('--kg_print_every', type=int, default=1,
                        help='Iter interval of printing KG loss.')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating multi-label.')

    parser.add_argument('--unfreeze_epoch', default=3, type=int)
    parser.add_argument('--max_epochs_before_stop', default=30, type=int,
                        help='stop training if dev does not increase for N epochs')


    args = parser.parse_args()

    save_dir = 'trained_model/TAKGAT/{}/entitydim{}_relationdim{}_{}_layer{}_kgat{}_tlayer{}_takgat{}_ablation-{}_sequenceBatch{}_kgBatch{}_kglossLambda{}_kgatDrop{}_takgatDrop{}_clip{}_hitalr{}_baselr{}_timeMLP{}_dropout{}_hitalayer{}_hitahead{}_hitaffn{}_hitascale{}_seed{}_{}/'.format(
        str(datetime.datetime.now().strftime('%Y-%m-%d')), args.entity_dim, args.relation_dim, args.aggregation_type,
        len(eval(args.kgat_dim_list)),args.kgat_dim_list,len(eval(args.takgat_dim_list)),args.takgat_dim_list,"-".join(args.ablation),args.sequence_batch_size,
        args.kg_batch_size,args.kg_l2loss_lambda,args.kgat_mess_dropout,args.takgat_mess_dropout,args.clip,
        args.hita_encoder_lr,args.base_lr,args.time_MLP_embedsize,args.hita_dropout_prob,args.hita_encoder_layer,
        args.hita_encoder_head_num,args.hita_encoder_ffn_size,args.hita_time_scale,
        args.seed,args.cuda_choice)

    args.save_dir = save_dir

    return args


