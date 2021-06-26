from urllib import parse
from pandas.core.frame import DataFrame
import torch 
from torch import nn 
import numpy as np 
from torch import optim 
import time 
import argparse
from model import *
from utils import * 
from transformers import AdamW,get_constant_schedule,get_linear_schedule_with_warmup
from collections import defaultdict

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser=argparse.ArgumentParser()
parser.add_argument('--model',type=str,required=True,help="Select model deeplearning")
parser.add_argument('--batch_size',type=int,required=True)
parser.add_argument('--n_epochs',type=int,required=True,help="Number epoch for trainig")
parser.add_argument('--n_classes',type=int,default=10,help="Number class")
parser.add_argument('--embedding_dim',type=int,default=300,help="Embedding size for word in LSTM")
parser.add_argument('--hidden_dim',type=int,default=256,help="Hidden size for LSTM")
parser.add_argument('--lr',type=float,default=1e-5,help="Learning rate")
parser.add_argument('--n_epoch_to_saved',type=int,default=4,help="The number epoch that we want to save checkpoint")
arg=parser.parse_args()


#criterion=nn.CrossEntropyLoss()
criterion=LabelSmoothingCrossEntropyLoss()

if arg.model=="BERT":
    train_loader,test_loader,val_loader=get_loader_bert(arg.batch_size)
    model=Model_BERT(arg.n_classes).to(device)
    optimizer=AdamW(model.parameters(),lr=arg.lr)
    model.fine_tune_bert(False)

elif arg.model=="LSTM":
    train_loader,test_loader,val_loader,train_dataset=get_loader_LSTM(arg.batch_size)
    embedding_size=arg.embedding_dim
    hidden_size=arg.hidden_dim
    vocab_size=train_dataset.vocab.vocab_size
    model=Model_RNN(embedding_size,hidden_size,vocab_size,arg.n_classes).to(device)
    optimizer=optim.Adam(model.parameters(),lr=arg.lr)

def train_LSTM():
    history=defaultdict(list)
    for epoch in range(arg.n_epochs):
        start_time=time.time()
        train_acc,train_loss=train_model(model,train_loader,optimizer,criterion)        
        val_acc,val_loss=evaluate(model,val_loader,criterion)
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch}----Train acc:{train_acc}----Train loss:{train_loss}---Val acc:{val_acc}---Val loss:{val_loss}---Time:{time.time()-start_time}")
        if epoch!=0 and epoch%5==0:
            print("--------------Time for testing-----------------")
            testing()  
            save_checkpoint(model,optimizer,history,epoch)


def train_BERT():
    frozen=True
    scheduler=get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader)*(arg.n_epochs-1)
    )
    scheduler_frozen=get_constant_schedule(optimizer)
    #x_batch,y_batch=next(iter(train_loader))
    history=defaultdict(list)
    for epoch in range(arg.n_epochs):

        if epoch > 0 and frozen:
            model.fine_tune_bert(True)
            del scheduler_frozen
            torch.cuda.empty_cache()
            frozen=False 

        start_time=time.time()
        if frozen:
            train_acc,train_loss=train_model(model,train_loader,optimizer,criterion,scheduler_frozen)
        else:
            train_acc,train_loss=train_model(model,train_loader,optimizer,criterion,scheduler)
            
        val_acc,val_loss=evaluate(model,val_loader,criterion)
        print(f"Epoch {epoch}----Train acc:{train_acc}----Train loss:{train_loss}---Val acc:{val_acc}---Val loss:{val_loss}---Time:{time.time()-start_time}")
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        if epoch!=0 and epoch%arg.n_epoch_to_saved==0:
            print("--------------Time for testing-----------------")
            testing(test_loader) 
            save_checkpoint(model,optimizer,epoch,history,scheduler)


if __name__=="__main__":
    if arg.model=="BERT":
        train_BERT()
        
    elif arg.model=="LSTM":
        train_LSTM()
