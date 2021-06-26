from flask import Flask, jsonify, request
from flask.templating import render_template
import pickle 
import torch 
from processing import process_data
from model import Model_BERT,Model_RNN
from utils import encode_text
from transformers import AutoTokenizer
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

flag=4
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels={0:"Van hoa",1:"Chinh tri Xa hoi",2:"Khoa hoc",3:"Phap luat",4:"Vi tinh",5:"Suc khoe",6:"Kinh doanh",7:"The thao",8:"The gioi",9:"Doi song"}
mode_name=""
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
    
elif flag==4:
    tokenizer=AutoTokenizer.from_pretrained('vinai/phobert-base')
    model_state=torch.load('Model/Bert/model16.pth',map_location="cpu")
    model_bert=Model_BERT(n_classes=10).to(device)
    model_bert.load_state_dict(model_state['model'])
    model_bert.eval()
    model_name="Bert +BiLSTM"



def get_label(text):
    label=None
    if flag==1 or flag==2:
        result=model.predict([text])
        label=labels.get(result[0])


    elif model=="LSTM":
        temp=vocab.convert_text_to_int(text)
        data=torch.tensor(temp).to(device)
        out=model(temp)
        index=torch.argmax(out,dim=1).item()
        label=labels[index]

    if flag==4:
        temp=encode_text(text,tokenizer)
        data=torch.tensor(temp,dtype=torch.long).view(1,256).to(device)
        out=model_bert(data)
        index=torch.argmax(out,dim=1).item()
        label=labels[index]
    
    return label 

app=Flask(__name__)
@app.route("/")
def index():
    return render_template('index.html')
            #data=[{'name':'Navie Bayes'}, {'name':'KNN'}, {'name':'BiLSTM'},{'name':'BERT + BiLSTM'}])

@app.route("/", methods=['GET', 'POST'])
def form_post():
    text=request.form['inputtext']
    text=process_data(text)
    label=get_label(text)
    return render_template("submit.html",model_name=model_name, result=label)

if __name__ == '__main__':
    app.run(debug=True)

#sudo netstat -tulnp | grep :5000
#sudo kill pid