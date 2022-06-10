
import numpy as np
from transformers import AutoModel
import torch.nn as nn
import torch

from src.dataloader import get_custom_dataloader

class QueringModel(nn.Module):
    def __init__(self, args):
        super(QueringModel, self).__init__()
        self.Model = Model(args.model_type, args.temp_dir, args.finetune)
        self.Model_drop = nn.Dropout(0.3)
        self.AvgPool = nn.AdaptiveAvgPool2d(output_size=(1,768))
        self.Dense1 = nn.Linear(768,128)
        self.Dense2 = nn.Linear(128,64)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(64,2)

    def forward(self, ids, mask, token_type_ids):
        o1 = self.Model(ids, attention_mask=mask, token_type_ids=token_type_ids)[0]
        bo = self.Model_drop(o1)
        bo = self.AvgPool(bo).squeeze(1)
        bo = self.relu(self.Dense1(bo))
        bo = self.relu(self.Dense2(bo))
        bo = self.out(bo)
        output = self.sigmoid(bo)
        return output

class Model(nn.Module):
    def __init__(self, model_type, temp_dir, finetune=False):
        super(Model, self).__init__()
        self.model = AutoModel.from_pretrained(model_type, cache_dir=temp_dir)
        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec

def query_data(args, dataloader):

    if args.query_technique == "transformer":
        queried_ids = query_annotated_data(args, dataloader["query_pool"])
    elif args.query_technique == "clustering":
        queried_ids = None
    return get_custom_dataloader(args, queried_ids, dataloader)

def query_annotated_data(args, dataloader):
        
    query_model = QueringModel(args)
    query_model.eval()
    pool_predictions = []
    
    with torch.no_grad():

        for i,batch in enumerate(dataloader['pool']):
        
            ids = batch[0].cuda()
            mask = batch[1].cuda()
            token_type_ids = batch[2].cuda()
            
            predictions = query_model(ids, mask, token_type_ids)
            pool_predictions.extend(np.concatenate(predictions.detach().cpu().numpy(), np.array(range(i*args.query_batch_size, (i+1)*args.query_batch_size), axis = 1)))
   
    pool_predictions.sort(reverse=True,key=confidence_criteria)

    return [pred[2] for i,pred in enumerate(pool_predictions) if i < args.query_thres]

def confidence_criteria(prediction):
    return np.abs(prediction[0]-prediction[1])