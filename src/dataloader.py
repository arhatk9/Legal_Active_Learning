from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import TensorDataset, ConcatDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from src.utils import *
import json
import numpy as np
from sklearn import preprocessing
import pandas as pd
import os
import pickle

def make_data_unbiased(Feature, Label):
    return Feature, Label

class DataLoader:
    def __init__(self, args):
        self.vocab_size = args.vocab_size
        self.max_length = args.max_len
        self.args = args
        self.csv_data = {}
        

    def get_feature_list(self, big_json):

        word_list, label_list = self.read_json_to_list(big_json)
        word_idx_list = []
        for wl in word_list:
            f = self.sentence_to_features(wl)
            word_idx_list.append(f)
        return word_idx_list,label_list
    
    def initialize_term_mappings(self, big_json):
        word_list, label_list = self.read_json_to_list(big_json)
        word_idx_list = []
        self.id2term, self.term2id = self.create_term_idx_mapping(word_list)
        for wl in word_list:
            f = self.sentence_to_features(wl)
            word_idx_list.append(f)

        
        return word_idx_list,label_list

    def read_json_to_list(self, big_json):

        with open(big_json, 'r') as f:
            all_documents = json.load(f)

        word_list = []
        label_list = []

        for key in all_documents:
            if all_documents[key]['judgementSents'] == None:
                continue

            for s in all_documents[key]['judgementSents']:
                word_list.append([w.lower() for w in word_tokenize(s['sentText'])])
                label_list.append(s['sentPseudoRelevance'])

        return word_list, label_list

    def read_json_to_sentences(self, big_json):

        with open(big_json, 'r') as f:
            all_documents = json.load(f)

        sent_list = []
        label_list = []

        for key in all_documents:
            if all_documents[key]['judgementSents'] == None:
                continue

            for s in all_documents[key]['judgementSents']:
                sent_list.append(s['sentText'])
                label_list.append(s['sentPseudoRelevance'])

        return sent_list, label_list

    def create_term_idx_mapping(self, all_tokens):
        '''
            input: list of lists for documents tokens
        '''
        tf = {}
        for s in all_tokens:
            for w in s["words"]:
                tf[w] = tf.get(w,0) + 1
        total_docs = len(all_tokens)

        sorted_tf = sorted(tf.items(), key=lambda x: x[1], reverse=True)[0:self.vocab_size]

        term2id = {}
        id2term = {}
        unique_words = 2
        term2id['<UNK>'] = 1
        term2id['<PAD>'] = 0
        id2term[0] = '<UNK>'
        id2term[1] = '<PAD>'

        for w,_ in sorted_tf:
            term2id[w] = unique_words
            id2term[unique_words] = w
            unique_words += 1
            
        return id2term, term2id

    def store_data(self):

        Feature, Label = self.initialize_term_mappings(self.args.data_path + "train.json")
        Feature_v, Label_v = self.get_feature_list(self.args.data_path + "valid.json")

        print(len(Feature), len(Label), len(Feature_v), len(Label_v))

        data_path = os.path.join(self.args.data_path, '%s.pt' % "tensor_data")
        data = {
            "Feature" : torch.Tensor(np.array(Feature)).type(torch.LongTensor),
            "Label" : torch.Tensor(Label),
            "Feature_val" : torch.Tensor(np.array(Feature_v)).type(torch.LongTensor),
            "Label_val" :  torch.Tensor(Label_v)
        }

        if (not os.path.exists(data_path)):
            torch.save(data, data_path)

    def get_encoder_dataloader(self):

        checkpoint = torch.load(self.args.data_path + "tensor_data.pt", map_location=lambda storage, loc: storage)
        train_data_size = self.args.nos_pool_batch_size * self.args.train_batch_size
        Feature, Label = make_data_unbiased(checkpoint["Feature"], checkpoint["Label"])
        
        train_dataset = TensorDataset(Feature[:train_data_size], Label[:train_data_size])
        pool_dataset = TensorDataset(Feature[train_data_size:], Label[train_data_size:])
        validation_dataset = TensorDataset(checkpoint["Feature_val"], checkpoint["Label_val"])

        train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset, generator=torch.Generator().manual_seed(seed=12)),
            batch_size = self.args.train_batch_size
        )
        pool_dataloader = DataLoader(
            pool_dataset,
            sampler = SequentialSampler(train_dataset),
            batch_size = self.args.train_batch_size
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            sampler = SequentialSampler(validation_dataset),
            batch_size = self.args.val_batch_size
        )

        return train_dataloader, validation_dataloader, pool_dataloader

    def get_query_dataloader(self):

        words, _ = self.read_json_to_list(self.args.data_path + "train.json")
        query_tokenizer = AutoTokenizer.from_pretrained(self.args.model_type)

        inputs = query_tokenizer.encode_plus(
            words,
            None,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        train_data_size = self.args.nos_pool_batch_size * self.args.train_batch_size
        pool_dataset = TensorDataset(ids[train_data_size:], mask[train_data_size:], token_type_ids[train_data_size:])

        pool_dataloader = DataLoader(
            pool_dataset,
            sampler = SequentialSampler(pool_dataset),
            batch_size = self.args.train_batch_size
        )

        words, _ = self.read_json_to_list(self.args.data_path + "val.json")

        inputs = query_tokenizer.encode_plus(
            words,
            None,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        validation_dataset = TensorDataset(ids, mask, token_type_ids)

        validation_dataloader = DataLoader(
            validation_dataset,
            sampler = SequentialSampler(validation_dataset),
            batch_size = self.args.val_batch_size
        )
        
        return validation_dataloader, pool_dataloader

    def get_test_dataloader(self):
        pass

    # create function for returning test dataloader doc wise

def get_custom_dataloader(args, pool_data_idx, dataloader):
    
    train_data = dataloader['train'].dataset
    new_train_data = [dataloader['pool'].dataset.__getitem__(idx) for idx in pool_data_idx]
    new_pool_data = [dataloader['pool'].dataset.__getitem__(idx) for idx in range(dataloader['pool'].dataset.__len__()) if idx not in pool_data_idx]
    new_query_pool_data = [dataloader['query_pool'].dataset.__getitem__(idx) for idx in range(dataloader['query_pool'].dataset.__len__()) if idx not in pool_data_idx]

    train_dataloader = DataLoader(
        ConcatDataset(train_data, new_train_data),
        sampler = RandomSampler(new_train_data),
        batch_size = args.train_batch_size
    )

    pool_dataloader = DataLoader(
        new_pool_data,
        sampler = SequentialSampler(new_pool_data),
        batch_size = args.train_batch_size
    )

    query_pool_dataloader = DataLoader(
        new_query_pool_data,
        sampler = SequentialSampler(new_query_pool_data),
        batch_size = args.train_batch_size
    )

    dataloader['train'] = train_dataloader
    dataloader['pool'] = pool_dataloader
    dataloader['query_pool'] = query_pool_dataloader

    return dataloader