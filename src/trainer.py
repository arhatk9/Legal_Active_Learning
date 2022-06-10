import torch
import torch.nn as nn
import torchmetrics
from src.query import query_data

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

def build_trainer(args, device_id, model, optim):

    trainer = Trainer(args, model, optim)

    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)
    return trainer

class Trainer(object):

    def __init__(self, args, model, optim):
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.loss = nn.CrossEntropyLoss(torch.tensor([0.5,1]).cuda())
        if (model):
            self.model.train()

    def train(self, dataloader):

        while dataloader['pool'].dataset.__len__() > 0:
            for _ in range(self.args.train_epochs):
                loss, acc, f1, precision, recall = self.train_fn(dataloader['train'], self.args)
                # Adding logging 
            loss, acc, f1, precision, recall = self.validate_fn(dataloader['valid'], self.args)
            # Adding logging 
            # Quering 
            dataloader = query_data(dataloader)       

    def train_fn(self, iterator, epochs):
    
        epoch_loss = 0
        self.model.train()
        
        # Metrics
        train_accuracy = torchmetrics.Accuracy()
        train_f1_score = torchmetrics.F1Score()
        train_recall = torchmetrics.Recall()
        train_precision = torchmetrics.Precision()

        for i, batch in enumerate(iterator):
            
            self.optim.zero_grad()
            text = batch[0].cuda()
            target = batch[1].type(torch.LongTensor).cuda()
            
            predictions = self.model(text) 
            
            loss = self.loss(predictions, target)
            batch_acc = train_accuracy(predictions, target)
            batch_f1 = train_f1_score(predictions, target)
            batch_recall = train_recall(predictions, target)
            batch_precision = train_precision(predictions, target)
            
            loss.backward() 
            
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            self.optim.step()
            
            epoch_loss += loss.item()
            
            if (i % 10000) == 0:
                print('Epoch : {} Accuracy : {} Loss : {} Batch:{}'.format(epochs, train_accuracy.compute(), epoch_loss/i, i))
            
        return epoch_loss / len(iterator), train_accuracy.compute(), train_f1_score.compute(), train_precision.compute(), train_recall.compute()

    def validate_fn(self, iterator, epochs):
        
        epoch_loss = 0
        self.model.eval()
        
        # Metrics
        valid_accuracy = torchmetrics.Accuracy()
        valid_f1_score = torchmetrics.F1Score()
        valid_recall = torchmetrics.Recall()
        valid_precision = torchmetrics.Precision()

        for i, batch in enumerate(iterator):
            
            text = batch[0].cuda()
            target = batch[1].type(torch.LongTensor).cuda()
            
            predictions = self.model(text) 
            
            loss = self.loss(predictions, target)
            batch_acc = valid_accuracy(predictions, target)
            batch_f1 = valid_f1_score(predictions, target)
            batch_recall = valid_recall(predictions, target)
            batch_precision = valid_precision(predictions, target)
                        
            epoch_loss += loss.item()
            
            if (i % 1000) == 0:
                print('Epoch : {} Accuracy : {} Loss : {} Batch:{}'.format(epochs, valid_accuracy.compute(), epoch_loss/i, i))
            
        return epoch_loss / len(iterator), valid_accuracy.compute(), valid_f1_score.compute(), valid_precision.compute(), valid_recall.compute()

    def test(self):
        pass