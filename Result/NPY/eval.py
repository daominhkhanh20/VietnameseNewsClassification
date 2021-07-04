import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score,recall_score
import pandas as pd 
import seaborn as sns 
from matplotlib import pyplot as plt 
def plot(final_true,final_preds,file_name):
  temp=confusion_matrix(final_true,final_preds)
  df=pd.DataFrame(temp,index=labels.values(),columns=labels.values()).astype(int)
  plt.figure(figsize=(12,8))
  heatmap=sns.heatmap(df,annot=True,fmt="d")
  heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),rotation=0,ha='right')
  heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),rotation=45,ha='right')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  result=accuracy_score(final_true,final_preds)
  plt.title("Accuracy for test:{:0.5f}".format(result))
  plt.savefig(file_name)
  #plt.savefig('/content/drive/MyDrive/Viettel3/Bert/{}'.format(file_name))

y_true=np.load('true.npy').tolist()
naive=np.load('naive.npy')
lstm=np.load('lstm.npy')
bert=np.load('bert.npy')
labels={0:"Van hoa",1:"Chinh tri Xa hoi",2:"Khoa hoc",3:"Phap luat",
4:"Vi tinh",5:"Suc khoe",6:"Kinh doanh",7:"The thao",
8:"The gioi",9:"Doi song"
}
temp=np.asarray([bert,naive,lstm])
result=stats.mode(temp)[0][0]
print("Accuracy:",accuracy_score(y_true,result))
print(classification_report(y_true,result,target_names=list(labels.values())))
print("Precision score:",precision_score(y_true,result,average='weighted'))
print("Recall score",recall_score(y_true,result,average='weighted'))
plot(y_true,result,'ensemble.png')
