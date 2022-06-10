import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, args, device, checkpoint):
        super(Encoder, self).__init__()

        self.args = args
        self.device = device

        if args.encoder_type == "BiLSTM" :
            self.model = BiLSTM(args)
        else:
            self.model = AttentionModel(args)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        self.to(device)

        def forward(self, text):
            top_vec = self.model(text)
            return top_vec

class BiLSTM(nn.Module):
    
    def __init__(self, args):
        super(BiLSTM,self).__init__()
        
        self.embedding = nn.Embedding(args.vocab_size, args.enc_emd_size)
        self.rnn =  getattr(nn, args.rnn_type)(args.enc_emd_size, args.enc_hidden_size, num_layers = args.enc_layers, bidirectional = True, dropout = args.enc_dropout)
        self.Linear = nn.Linear(args.enc_hidden_size*2,args.enc_out_size)
        self.Dropout = nn.Dropout(p=args.enc_dropout)
        self.max_len = args.max_len
        self.act = nn.Sigmoid()
        self.init_weights()

    def forward(self,text):
        
        embedded = self.embedding(text)        
        embedded = embedded.permute(1, 0, 2)
        output, (hidden,cell) = self.rnn(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = self.Dropout(hidden)
        dense_outputs = self.Linear(hidden)
        outputs = dense_outputs
        outputs = self.act(dense_outputs)
        
        return outputs
    
    def init_weights(self):
        ''' Initialise weights of embeddings '''
        torch.nn.init.xavier_normal_(self.embedding.weight)
        ''' Initialise weights of encoder RNN ''' 
        torch.nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        torch.nn.init.xavier_normal_(self.lstm.weight_hh_l0)
        torch.nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        torch.nn.init.xavier_normal_(self.lstm.weight_hh_l0)
        ''' Dense '''
        torch.nn.init.xavier_normal_(self.Linear.weight)

class AttentionModel(nn.Module):

    def __init__(self, args):
        super(AttentionModel, self).__init__()
        
        self.bidirectional = True
        self.ntoken = args.vocab_size
        self.nlayers = args.enc_layers,
        self.embedding_size = args.enc_emd_size
        self.rnn_hidden_size = args.enc_hidden_size
        self.rnn_type = args.rnn_type
        self.dropout_p = args.enc_dropout
        self.embedding = nn.Embedding(self.ntoken, self.embedding_size)
        self.embedding.weight.requires_grad=True
        
        self.encoder_i = getattr(nn, rnn_type)(self.embedding_size, self.rnn_hidden_size, dropout=self.dropout_p, bidirectional=bidirectional)
        self.encoder_h = getattr(nn, rnn_type)(self.lstm_hidden_size, self.rnn_hidden_size, dropout=self.dropout_p, bidirectional=bidirectional)
        
        self.initial_embeddings_tensor = iembed_tensor
        self.dropout = nn.Dropout(self.dropout_p)
        self.init_weights()
        self.label = nn.Linear(hidden_size, output_size)
        #self.label1 = nn.Linear(2,1)

    def init_weights(self):
        ''' Initialise weights of embeddings '''
        if self.initial_embeddings_tensor != None:
            self.embedding.weight.data.copy_(torch.from_numpy(self.initial_embeddings_tensor))
        else:
            torch.nn.init.xavier_normal_(self.embedding.weight.data)
        ''' Initialise weights of encoder RNN '''
        torch.nn.init.xavier_normal_(self.encoder_i.weight_ih_l0)
        torch.nn.init.xavier_normal_(self.encoder_i.weight_hh_l0)
        torch.nn.init.xavier_normal_(self.encoder_h.weight_ih_l0)
        torch.nn.init.xavier_normal_(self.encoder_h.weight_hh_l0)
        
    def attention_layer(self,encoder_output,final_hidden_state):
        
        hidden = final_hidden_state.squeeze(0)
        #print('Hidden_Attention Size : '+str(hidden.size()))
        
        attn_weights = torch.bmm(encoder_output, hidden.unsqueeze(2)).squeeze(2)
        #print('Attention Weights : '+str(attn_weights.size()))
        
        soft_attn_weights = F.softmax(attn_weights,0)
        #print('Soft Attention Weights : '+str(soft_attn_weights.size()))
        
        new_hidden_state = torch.bmm(encoder_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state,attn_weights

    def forward(self, input,input1):
        
        embedded = self.embedding(input)
        embedded = embedded.permute(1, 0, 2)
        embedded = self.dropout(embedded)
        

        output, (hidden) = self.encoder_i(embedded)
        for _ in range(self.nlayers-1):
            output, (hidden) = self.encoder_h(output,hidden)
            
        
        #print('Hidden Size : '+str(hidden[0].size()))
        
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        else:
            hidden = hidden[-1]
                    
        #print('Output Size : '+str(output.size()))
        output = output.permute(1, 0, 2)
        
        #print('Permuted Output Size : '+str(output.size()))
        #print('Hidden Size : '+str(hidden.size()))
        
        attn_output,attn_weights = self.attention_layer(output,hidden)
        #print('Attn_output : '+str(attn_output.size()))
        
        #Avg = (attn_output + input1)/2;
        #print(input1.size())
        logits = self.label(attn_output)
        #print('Logits : '+str(logits.size()))
        #print(logits.size())
        #logits = self.label1(logits)
        return logits,attn_weights

    def initHidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(1, bsz, self.lstm_hidden_size).zero_().cuda()),
                    Variable(weight.new(1, bsz, self.lstm_hidden_size).zero_().cuda()))
        else:
            return Variable(weight.new(1, bsz, self.lstm_hidden_size).zero_())