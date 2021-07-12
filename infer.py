import torch 
import argparse
import numpy as np 
import pickle 
from processing import process_data
from transformers import AutoTokenizer
from model import BERT_LSTM,BILSTM, Vocabulary,LabelSmoothingCrossEntropyLoss
from utils import encode_text
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels={0:"Van hoa",1:"Chinh tri Xa hoi",2:"Khoa hoc",3:"Phap luat",4:"Vi tinh",5:"Suc khoe",6:"Kinh doanh",7:"The thao",8:"The gioi",9:"Doi song"}

parser=argparse.ArgumentParser()
parser.add_argument('--model',type=str,required=True,help="Select model for inference")
parser.add_argument('--file_test',type=str,required=True,help="Select file to test")
arg=parser.parse_args()


file=open("Test/"+arg.file_test,'r',encoding='utf-8')
text=file.read()
text=process_data(text)

if arg.model=="NB":
    with open("Model/model_naive_bayes.pickle",'rb') as file:
        model=pickle.load(file)
elif arg.model=="KNN":
    with open("Model/model_knn.pickle",'rb') as file:
        model=pickle.load(file)

elif arg.model=="LSTM":
    with open("Model/LSTM/vocab.pickle","rb") as file:
        vocab=pickle.load(file)
    model_state=torch.load('Model/LSTM/model39.pth',map_location='cpu')
    model=BILSTM(embedding_size=300,hidden_size=256,vocab_size=vocab.vocab_size,num_class=10).to(device)
    model.load_state_dict(model_state['model'])
    model.eval()

elif arg.model=="BERT":
    model_state=torch.load('Model/Bert/model21.pth',map_location='cpu')
    model=BERT_LSTM(n_classes=10).to(device)
    model.load_state_dict(model_state['model'])
    model.eval()


def get_label():
    label=None
    if arg.model=="NB" or arg.model=="KNN":
        result=model.predict([text])
        label=labels.get(result[0])

    elif arg.model=="LSTM":
        temp=vocab.convert_text_to_int(text)
        data=torch.tensor(temp).to(device)
        out=model(data.unsqueeze(dim=0))
        index=torch.argmax(out,dim=1).item()
        label=labels[index]

    elif arg.model=="BERT":
        tokenizer=AutoTokenizer.from_pretrained('vinai/phobert-base')
        temp=encode_text(text,tokenizer)
        data=torch.tensor(temp,dtype=torch.long).view(1,256).to(device)
        out=model(data)
        index=torch.argmax(out,dim=1).item()
        label=labels[index]
    
    return label 


if __name__=="__main__":
    label=get_label()
    print(label)
