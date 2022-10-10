
# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/

import parser
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=int,default=0,help="0: data pre-processing mode; \n"
                                                         "1: training mode;\n "
                                                         "2: inference mode")
    #configure training param settings
    parser.add_argument('-F',type=str, required=True,help='dataset directory path')
    parser.add_argument('-M', type=str,default='Train_Model', help='model path for evluation and the path to save model')
    parser.add_argument('--resume', type=int, default=0, help='reload trained model to continue training')

    parser.add_argument('--dataset',type=str,default='B',help="Dataset name: \n B:breakfast dataset;\n"
                                                              "S: 50 Salads dataset;\n I: INRIA dataset ")

    parser.add_argument('--start_epoch', default=0, type=int, help='startring epoch for resuming situation')
    parser.add_argument('--epochs', default=30, type=int,help='number of total epochs to run')
    parser.add_argument('--lr',  default=0.002, type=float,help='initial learning rate')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')

    parser.add_argument('--seed', type=int, default=888, help='manual seed')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training/evaluation')
    parser.add_argument('--gpu', type=str, default='', help='specified gpu id')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--log_path', type=str, default='train_log', help='training log record and model save path')
    #configure self supervised learning model
    parser.add_argument('--encode_feature',type=int,default=1024,help="encoding feature dimensions of frames")
    parser.add_argument('-moco_T',default=0.2,type=float,help="Temperature for contrastive loss")
    parser.add_argument('--moco_k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')

    # configure masked video model
    parser.add_argument("--window_length",type=int,default=10,help="frame length to input as a window for CoSeg")
    parser.add_argument('--mask_length',default=1,type=int,help="number of frames to be reconstructed")
    parser.add_argument("--num_attention",default=8,type=int,help="number of attention heads")
    parser.add_argument("--dropout",default=0.1,type=float,help="dropout rate for dropout layers")
    parser.add_argument("--num_layers",default=2,type=int,help="transformer layers settings")

    parser.add_argument('--beta',default=1,type=float,help="coefficient to balance contrastive loss and reconstruction loss")

    #configure the error detection module
    #please do a search for different dataset
    parser.add_argument("--smooth_factor",default=10,type=int,help="smooth factor for error detection module")
    parser.add_argument("--range",default=70,type=int,help="range parameter M for boundary detection module")

    parser.add_argument('--tensorboard',default=0,type=int,help="use tensorboard to record logs or not")

    args = parser.parse_args()
    params = vars(args)
    return params
