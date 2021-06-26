from torch.utils.data import DataLoader
import pandas as pd 
from model import News,MyCollate
import pickle
import torch 
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt 

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_sequence_length=256
def split_data(path_file):
    train=pd.read_csv(path_file)
    print("Shape train:",train.shape)
    val=pd.DataFrame()
    n_classes=len(set(train.label))
    for i in range(n_classes):
        temp=train[train.label==i]
        temp=temp.sample(int(len(temp)*0.2))
        val=pd.concat([val,temp],axis=0)

    train=train[~train.index.isin(val.index)]
    train.to_csv(path_file+"/train.csv")
    val.to_csv(path_file+"/val.csv")


def data_loader_LSTM(data_set,batch_size):
    pad_idx=data_set.vocab.stoi['<pad>']
    return DataLoader(dataset=data_set,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True,
                    collate_fn=MyCollate(pad_idx))


def get_loader_LSTM(batch_size):
    train_dataset=News('News/train.csv')
    val_dataset=News('News/val.csv',train_dataset.vocab)
    test_dataset=News('News/test_n.csv',train_dataset.vocab)
    train_loader=data_loader_LSTM(train_dataset,batch_size)
    test_loader=data_loader_LSTM(test_dataset,batch_size)
    val_loader=data_loader_LSTM(val_dataset,batch_size)
    return train_loader,test_loader,val_loader,train_dataset


def save_checkpoint(model,optimizer,history,epoch,is_bert=False,scheduler=None):
    with open('/history{}.pickle'.format(epoch),'wb') as file:
        pickle.dump(history,file,protocol=pickle.HIGHEST_PROTOCOL)
    print("Save history done")
    model_state={
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict()
    }
    if scheduler is not None:
        model_state['scheduler']=scheduler.state_dict()
    if is_bert is True:
        torch.save(model_state,"Model/Bert/model_bert{}.pth".format(epoch))
    else :
        torch.save(model_state,"Model/LSTM/model_lstm{}.pth".format(epoch))
    print("Save model done")

def encode_text(text,tokenizer):
    temp=tokenizer.encode(text)
    if len(temp)<max_sequence_length:
        temp+=[1]*(max_sequence_length-len(temp))
    else:
        temp=temp[:max_sequence_length]
        temp[-1]=tokenizer.eos_token_id
    
    return temp

def encode_data(data,tokenizer):
    result=[]
    for text in data.text.values:
        temp=tokenizer.encode(text)
        if len(temp)<max_sequence_length:
            temp+=[1]*(max_sequence_length-len(temp))
        else:
            temp=temp[:max_sequence_length]
            temp[-1]=tokenizer.eos_token_id
        result.append(temp)
    return result


def get_loader_bert(batch_size):
    train=pd.read_csv('News/train.csv')
    val=pd.read_csv('News/val.csv')
    test=pd.read_csv('News/test_n.csv')
    train['data']=encode_data(train)
    test['data']=encode_data(test)
    val['data']=encode_data(val)
    train_dataset_tensor=torch.utils.data.TensorDataset(torch.tensor(train['data'],dtype=torch.long),torch.tensor(train['label'],dtype=torch.long))
    test_dataset_tensor=torch.utils.data.TensorDataset(torch.tensor(test['data'],dtype=torch.long),torch.tensor(test['label'],dtype=torch.long))
    val_dataset_tensor=torch.utils.data.TensorDataset(torch.tensor(val['data'],dtype=torch.long),torch.tensor(val['label'],dtype=torch.long))
    train_loader=DataLoader(train_dataset_tensor,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(test_dataset_tensor,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_dataset_tensor,batch_size=batch_size,shuffle=True)
    return train_loader,test_loader,val_loader


def testing(model,test_loader):
    print("--------------Time for testing-----------------")
    model.eval()
    final_preds=[]
    final_targets=[]
    with torch.no_grad():
        for idx,(x_batch,y_batch) in enumerate(test_loader):
            x_batch=x_batch.to(device)
            y_batch=y_batch.to(device)
            outputs=model(x_batch)
            outputs=torch.softmax(outputs,dim=1)
            preds=torch.argmax(outputs,dim=1)
            final_preds.extend(preds.cpu().numpy().tolist())
            final_targets.extend(y_batch.cpu().numpy().tolist())
            if idx%100==0:
                print(idx,end=" ")
    print()
    print("Accuracy for data test:",accuracy_score(final_targets,final_preds))


def evaluate(model,data_loader,criterion):
    print("------------------------Evaluate---------------------------")
    model.eval()
    loss_val=0
    final_preds=[]
    final_true=[]
    for idx,(x_batch,y_batch) in enumerate(data_loader):
        x_batch=x_batch.to(device)
        y_batch=y_batch.to(device)
        outputs=model(x_batch)
        loss=criterion(outputs,y_batch)
        outputs=torch.softmax(outputs,dim=1)
        preds=torch.argmax(outputs,dim=1)
        loss_val+=loss.item()
        final_preds.extend(preds.cpu().numpy().tolist())
        final_true.extend(y_batch.cpu().numpy().tolist())
        if idx%100==0:
            print(idx,end=' ')

    print()
    return accuracy_score(final_true,final_preds),loss_val/len(data_loader)


def train_model(model,data_loader,optimizer,criterion,scheduler_model=None):
    print("-----------------------Training--------------------------")
    model.train()
    loss_train=0
    final_preds=[]
    final_true=[]
    for idx,(x_batch,y_batch) in enumerate(data_loader):    
        x_batch=x_batch.to(device)
        y_batch=y_batch.to(device)
        outputs=model(x_batch)
        loss=criterion(outputs,y_batch)
        outputs=torch.softmax(outputs,dim=1)
        preds=torch.argmax(outputs,dim=1)
        final_preds.extend(preds.cpu().numpy().tolist())
        final_true.extend(y_batch.cpu().numpy().tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler_model is not None:
            scheduler_model.step()
        loss_train+=loss.item()
        if idx%100==0:
            print(idx,end=' ')
    print()  
    return accuracy_score(final_true,final_preds),loss_train/len(data_loader)

