import time
import argparse
import os
import logging
import pandas as pd
import torch
import random

from src.encoder import Encoder
from src.dataloader import DataLoader
from src.optimizer import build_optim
from src.trainer import Trainer, build_trainer


logger = logging.getLogger(__name__)

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-result_path", default='results/')
    parser.add_argument("-pickle_files_path", default="pickle_files/")
    parser.add_argument("-data_path", default="data/nl_data/")
    parser.add_argument("-lower",type=bool,default=True)
    parser.add_argument("-max_src_ntokens_per_sent",type=int,default=150)
    parser.add_argument("-min_src_ntokens_per_sent",type=int,default=3)

    parser.add_argument("-query_batch_size", default=140, type=int)
    parser.add_argument("-max_pos", default=512, type=int)

    # Funtionality for pretraining query model 
    #
    # parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    # parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    # parser.add_argument("-beta1", default= 0.9, type=float)
    # parser.add_argument("-beta2", default=0.999, type=float)
    # parser.add_argument("-warmup_steps", default=8000, type=int)
    # parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    # parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    # parser.add_argument("-max_grad_norm", default=0, type=float)
    # parser.add_argument("-label_smoothing", default=0.1, type=float)
    # parser.add_argument("-train_steps", default=1000, type=int)
    # parser.add_argument("-lr_bert", default=2e-3, type=float)

    # Funtionality for Document Level Quering 

    parser.add_argument("-enc_ff_size", default=512, type=int)

    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)

    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)

    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='logs/model.log')
    parser.add_argument('-seed', default=666, type=int)

    # Active Learning
    parser.add_argument("-nos_pool_batch_size", default=25, type=int)
    parser.add_argument("-train_docs_path", default="", type=str)
    parser.add_argument("-encoder", default="baseline", type=str, choices=['baseline'])
    parser.add_argument("-encoder_checkpoint", type=str, default="")
    parser.add_argument("-query_model", default="bert", type=str, choices=['bert'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-max_len", default=80, type=int)
    parser.add_argument("-vocab_size", default=10000, type=int)
    parser.add_argument("-train_batch_size", default=32, type=int)
    parser.add_argument("-val_batch_size", default=32, type=int)
    parser.add_argument("-enc_hidden_size", default=256, type=int)
    parser.add_argument("-rnn_type", default="LSTM", type=str)
    parser.add_argument("-enc_emd_size", default=300, type=int)
    parser.add_argument("-enc_layers", default=2, type=int)
    parser.add_argument("-enc_dropout", default=0.4, type=float)
    parser.add_argument("-enc_out_size", default=2, type=int)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    logger.info('Args : {}'.format(args))

    # add condition to check if already stored dataloader. Will save time in intializing term mappings.
    dataLoader = DataLoader(args)
    dataLoader.store_data()
    logger.info('Storing dataloader to %s' % args.pickle_files_path + "dataloader_" + time.time() + ".pt")
    #add storing funtionality


    if (args.mode == 'train'):
        train_ext(args, device_id)
    else: 
        test_ext(args, device_id)

def train_ext(args, device_id):

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.encoder_checkpoint != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.encoder_checkpoint,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    def get_dataloaders():
        dL = DataLoader(args)
        dL.store_data()
        encoder_dataloaders = dL.get_encoder_dataloader()
        query_dataloaders = dL.get_query_dataloader()

        return {
            "train" : encoder_dataloaders[0],
            "pool" : encoder_dataloaders[1],
            "valid" : encoder_dataloaders[2],
            "query_pool" : query_dataloaders[0],
            "query_valid" : query_dataloaders[1]
        }

    model = Encoder(args, device, checkpoint)
    optim = build_optim(args, model, checkpoint)

    logger.info(model)
    
    trainer = build_trainer(args, device_id, model, optim)
    trainer.train(get_dataloaders())