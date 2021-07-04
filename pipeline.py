from model import Vocabulary,BERT_LSTM,BILSTM
from utils import encode_text
from transformers import AutoTokenizer
import pickle 
import torch 
from processing import process_data
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels={0:"Van hoa",1:"Chinh tri Xa hoi",2:"Khoa hoc",3:"Phap luat",4:"Vi tinh",5:"Suc khoe",6:"Kinh doanh",7:"The thao",8:"The gioi",9:"Doi song"}
tokenizer=None
vocab=None
flag=4
if flag==1:
    with open("Model/model_naive_bayes.pickle",'rb') as file:
        model=pickle.load(file)
        model_name="Naive Bayes"

elif flag==2:
    with open("Model/model_knn.pickle",'rb') as file:
        model=pickle.load(file)
        model_name="KNN"

elif flag==3:
    with open("Model/LSTM/vocab.pickle",'rb') as file:
        vocab=pickle.load(file)
    
    model_state=torch.load('Model/LSTM/model39.pth',map_location="cpu")
    model=BILSTM(embedding_size=300,hidden_size=256,vocab_size=vocab.vocab_size,num_class=10)
    model.load_state_dict(model_state['model'])
    model.eval()
    model_name="BiLSTM"

    
elif flag==4:
    tokenizer=AutoTokenizer.from_pretrained('vinai/phobert-base')
    model_state=torch.load('Model/Bert/model21.pth',map_location="cpu")
    model=BERT_LSTM(n_classes=10).to(device)
    model.load_state_dict(model_state['model'])
    model.eval()
    model_name="Bert +BiLSTM"
    
    


def get_label(text):
    text=process_data(text)
    if flag==1 or flag==2:
        result=model.predict([text])
        label=labels.get(result[0])


    elif flag==3:
        temp=vocab.convert_text_to_int(text)
        data=torch.tensor(temp).to(device)
        out=model(data.unsqueeze(dim=0))
        index=torch.argmax(out,dim=1).item()
        label=labels[index]

    if flag==4:
        temp=encode_text(text,tokenizer)
        data=torch.tensor(temp,dtype=torch.long).view(1,256).to(device)
        out=model(data)
        index=torch.argmax(out,dim=1).item()
        label=labels[index]
    
    return label,model_name
