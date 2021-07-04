import torch 
from torch import nn 
import pandas as pd 
from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.nn.utils.rnn import pad_sequence 
from transformers import AutoModel,AutoTokenizer
from torch.nn import functional as F 
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Vocabulary:
    def __init__(self):
        self.vocab_size=100000
        self.tokenizer=Tokenizer(num_words=self.vocab_size,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n',
            oov_token='<unk>'
            )
        
    def __len__(self):
        return len(self.tokenizer.word_index)
        
    def build_vocab(self,texts):
        self.tokenizer.fit_on_texts(texts)
        self.tokenizer.word_index['<pad>']=0
        self.tokenizer.index_word[0]='<pad>'
        self.stoi=self.tokenizer.word_index
        self.itos=self.tokenizer.index_word 
    
    def convert_text_to_int(self,text):
        return self.tokenizer.texts_to_sequences([text])[0]

class News(Dataset):
    def __init__(self,file_data,vocab=None):
        self.data=pd.read_csv(file_data)
        self.texts=self.data.text
        self.labels=self.data.label
        if vocab is None:
            self.vocab=Vocabulary()
            self.vocab.build_vocab(self.texts.tolist())
        else:
            self.vocab=vocab
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        text=self.texts[idx]
        label=self.labels[idx]
        text_to_int=self.vocab.convert_text_to_int(text)
        return torch.tensor(text_to_int),torch.tensor([label])

class MyCollate:
    def __init__(self,pad_idx):
        self.pad_idx=pad_idx

    def __call__(self,batch):
        texts=[item[0] for item in batch]
        labels=[item[1] for item in batch]
        labels=torch.cat(labels,dim=0)
        texts=pad_sequence(texts,batch_first=True,padding_value=self.pad_idx)
        return texts,labels
    
class BILSTM(nn.Module):
    def __init__(self,embedding_size,hidden_size,vocab_size,num_class,drop_prob=0.2):
        super(BILSTM,self).__init__()
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.num_class=num_class
        self.embedding=nn.Embedding(vocab_size,self.embedding_size)
        self.lstm=nn.LSTM(input_size=embedding_size,
                        hidden_size=hidden_size,
                        batch_first=True,
                        num_layers=2,
                        bidirectional=True)
        self.dropout=nn.Dropout(drop_prob)
        self.norm=nn.LayerNorm(hidden_size*4)
        self.linear=nn.Linear(hidden_size*4,num_class)
    
    def forward(self,texts):
        #texts: batch_size*seq_length
        embedded=self.embedding(texts)#batch_size*seq_length*embedding_size
        #lstm_out: batch_size*seq_length*(2*hidden_size)
        lstm_out,_=self.lstm(embedded)
        avg_hidden=torch.mean(lstm_out,dim=1)
        max_hidden,_=torch.max(lstm_out,dim=1)
        input_linear=torch.cat((avg_hidden,max_hidden),dim=1)
        out=self.linear(self.norm(input_linear))
        return out


class LabelSmoothingCrossEntropyLossv1(nn.Module):
    def __init__(self,n_classes,smoothing_label=0.1):
        super(LabelSmoothingCrossEntropyLossv1,self).__init__()
        self.smoothing_label=smoothing_label
        self.n_classes=n_classes
    
    def forward(self,preds,targets):
        one_hot=torch.zeros(preds.size(0),preds.size(1)).to(device)
        one_hot[torch.arange(preds.size(0)),targets]=1
        one_hot_smoothing=(1-self.smoothing_label)*one_hot+self.smoothing_label/self.n_classes
        log_prob=F.log_softmax(preds,dim=1)
        temp=one_hot_smoothing.mul(log_prob).sum(dim=1)#element-wise
        return temp.mean()

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self,smoothing_value=0.1,reduction='mean'):
        super(LabelSmoothingCrossEntropyLoss,self).__init__()
        self.smoothing_value=smoothing_value
        self.reduction=reduction
    
    def reduce_loss(self,loss,reduction='mean'):
        return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss 

    def forward(self,outputs,targets):
        n_classes=outputs.size(1)
        log_preds=F.log_softmax(outputs,dim=1)
        loss=self.reduce_loss(-log_preds.sum(dim=-1),self.reduction)
        nll=F.nll_loss(log_preds,targets,reduction=self.reduction)
        return (1-self.smoothing_value)*nll+ self.smoothing_value*(loss/n_classes)


class BERT(nn.Module):
    def __init__(self,n_classes):
        super(BERT,self).__init__()
        self.n_classes=n_classes
        self.bert=AutoModel.from_pretrained("vinai/phobert-base")
        self.norm=nn.LayerNorm(self.bert.config.hidden_size)
        self.linear=nn.Linear(self.bert.config.hidden_size,n_classes)

    def fine_tune_bert(self,fine_tune=False):
            for child in self.bert.children():
                for param in child.parameters():
                    param.requires_grad=fine_tune

    def forward(self,x_batch):
        cls_embedding = self.bert(x_batch)[1]
        out=self.norm(cls_embedding)
        out=self.linear(out)
        return out

class BERT_LSTM(nn.Module):
    def __init__(self,n_classes,drop_prob=0.2):
        super(BERT_LSTM,self).__init__()
        self.n_classes=n_classes
        self.bert=AutoModel.from_pretrained("vinai/phobert-base")
        self.tokenizer=AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.lstm=nn.LSTM(input_size=self.bert.config.hidden_size,
                      hidden_size=512,
                      batch_first=True,
                      num_layers=2,
                      bidirectional=True)
        self.linear=nn.Sequential(
            nn.Linear(512*4,512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512,self.n_classes)
        )
        #self.fine_tune_bert(False)
      
    def forward(self,x_batch):
        word_embedding = self.bert(x_batch)[0]
        lstm_out,_=self.lstm(word_embedding)
        avg_hidden=torch.mean(lstm_out,dim=1)
        max_hidden,_=torch.max(lstm_out,dim=1)
        input_linear=torch.cat((avg_hidden,max_hidden),dim=1)
        out=self.linear(input_linear)
        return out
    
    def fine_tune_bert(self,fine_tune=False):
        for child in self.bert.children():
            for param in child.parameters():
                param.requires_grad=fine_tune
        
    