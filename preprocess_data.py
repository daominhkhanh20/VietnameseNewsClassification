import os 
from processing import process_data

path_10_label="/media/daominhkhanh/D:/Data/Project/ViettelProject/Data/10Label"

def preprocessing(folder_data_raw,folder_data,file_data):
    paths=os.path.join(path_10_label,folder_data_raw)

    path_temp="/media/daominhkhanh/D:/Data/Project/ViettelProject/"+folder_data
    if os.path.exists(path_temp) is False:
        os.mkdir(path_temp)

    file_s=open('/{}/{}.csv'.format(path_temp,file_data),'w')
    file_s.write('file_name,text,label\n')
    for idx,label in enumerate(os.listdir(paths)):
        path_to_label=os.path.join(paths,label)
        for file_name in os.listdir(path_to_label):
            file=open(path_to_label+'/'+file_name,"r",encoding="utf-16")
            text=file.read()
            text=process_data(text)
            file_s.write(f"{file_name},{text},{idx}\n")
        
        print(f"Done {label}")

    file_s.close()

preprocessing('Train_Full','News','train_n')
preprocessing('Test_Full','News','test_n')