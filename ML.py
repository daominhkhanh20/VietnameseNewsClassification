import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from matplotlib import pyplot as plt 
import pickle 
import os 
import argparse 

labels={0:"Van hoa",1:"Chinh tri Xa hoi",2:"Khoa hoc",3:"Phap luat",4:"Vi tinh",5:"Suc khoe",6:"Kinh doanh",7:"The thao",8:"The gioi",9:"Doi song"}
parser=argparse.ArgumentParser()
parser.add_argument('--model',type=str,required=True,help="Select model to train")
parser.add_argument('--path_file_train',type=str,default='News/train.csv')
parser.add_argument('--path_file_test',type=str,default="News/test_n.csv")
parser.add_argument('--path_file_val',type=str,default='News/val.csv')
parser.add_argument('--n_centers',type=int,default=10,help="Number center for KNN")
parser.add_argument('--path_to_save_result',type=str,default="Result")
parser.add_argument('--path_to_save_model',type=str,default='Model/')
parser.add_argument('--plot_result',type=bool,default=False,help="whether plot result")
parser.add_argument('--n_components',type=int,default=300)
parser.add_argument('--file_name',type=str,default="Model",help="File name image to saved")
parser.add_argument('--save_model',type=bool,default=False)
arg=parser.parse_args()

class NaiveBayes():
    def __init__(self,path_train,path_val,path_test):
        self.data_train=pd.read_csv(path_train)
        self.data_val=pd.read_csv(path_val)
        self.data_test=pd.read_csv(path_test)
        self.x_train=self.data_train.text
        self.y_train=self.data_train.label
        self.x_val=self.data_val.text
        self.y_val=self.data_val.label 
        self.x_test=self.data_test.text
        self.y_test=self.data_test.label 
        self.model=Pipeline([
                ('vect',CountVectorizer(ngram_range=(1,1),max_features=None)),
                #('tfidf',TfidfTransformer()),
                ('clf',MultinomialNB())
            ])

    def train(self):
        self.model.fit(self.x_train,self.y_train)
    
    def predict(self,x,y,target_names=None,plot_result=False,file_name=None):
        y_preds=self.model.predict(x)
        if plot_result is True:
            self.plot(y,y_preds,target_names,file_name)
        return accuracy_score(y,y_preds),confusion_matrix(y,y_preds)

    def plot(self,y_true,y_preds,target_names,file_name):
        print(target_names)
        cof=confusion_matrix(y_true,y_preds)
        df=pd.DataFrame(cof,index=target_names,columns=target_names)
        plt.figure(figsize=(12,9))
        heatmap=sns.heatmap(df,annot=True,fmt="d")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),rotation=0,ha='right')
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),rotation=45,ha='right')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        path=arg.path_to_save_result 
        if os.path.exists(path) is False:
            os.mkdir(path)

        plt.savefig(path+'/'+file_name+'.png')
    
    def save_model(self):
        with open(arg.path_to_save_model+'model_naive_bayes.pickle','wb') as file:
            pickle.dump(self.model,file,protocol=pickle.HIGHEST_PROTOCOL)
    
    def get_report(self,y_true,y_preds,target_names):
        print(classification_report(y_true,y_preds,target_names=target_names))



class KNN:
    def __init__(self,path_train,path_val,path_test,n_centers):
        self.data_train=pd.read_csv(path_train)
        self.data_val=pd.read_csv(path_val)
        self.data_test=pd.read_csv(path_test)
        self.x_train=self.data_train.text
        self.y_train=self.data_train.label
        self.x_val=self.data_val.text
        self.y_val=self.data_val.label 
        self.x_test=self.data_test.text
        self.y_test=self.data_test.label  
        self.n_center=n_centers
        self.model=Pipeline([
                    ('vect',CountVectorizer(ngram_range=(1,1),max_features=None)),
                    ('svd',TruncatedSVD(n_components=arg.n_components,random_state=42)),
                    ('clf',KNeighborsClassifier(n_neighbors=n_centers))
        ])
        

    def train(self):
        self.model.fit(self.x_train,self.y_train)
    
    def predict(self,x,y,target_names=None,plot_result=False,file_name=None):
        y_preds=self.model.predict(x)
        if plot_result is True:
            self.plot(y,y_preds,target_names,file_name)
        return accuracy_score(y,y_preds),confusion_matrix(y,y_preds)

    def plot(self,y_true,y_preds,target_names,file_name):
        cof=confusion_matrix(y_true,y_preds)
        df=pd.DataFrame(cof,index=target_names,columns=target_names)
        plt.figure(figsize=(12,9))
        heatmap=sns.heatmap(df,annot=True,fmt="d")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),rotation=0,ha='right')
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),rotation=45,ha='right')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        path=arg.path_to_save_result
        if os.path.exists(path) is False:
            os.mkdir(path)

        plt.savefig(path+'/'+file_name+'.png')

    def save_model(self):
        with open(arg.path_to_save_model+'model_knn.pickle','wb') as file:
            pickle.dump(self.model,file,protocol=pickle.HIGHEST_PROTOCOL)
    
    def get_report(self,y_true,y_preds,target_names):
        print(classification_report(y_true,y_preds,target_names=target_names))



if __name__=="__main__":
    if arg.model=='NB':
        model=NaiveBayes(arg.path_file_train,arg.path_file_val,arg.path_file_test)  

    elif arg.model=="KNN":
        model=KNN(arg.path_file_train,arg.path_file_val,arg.path_file_test,arg.n_centers)
    
    model.train()
    acc,conf_matrix=model.predict(model.x_val,model.y_val,plot_result=arg.plot_result,target_names=labels.values(),file_name=arg.file_name)
    print("Accuracy for validation set:{:.3f}".format(acc))
    if arg.save_model:
        model.save_model()

