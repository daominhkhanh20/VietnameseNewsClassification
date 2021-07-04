## VietNamese Classification News Project 

## Dataset
A Large-scale Vietnamese News Text Classification Corpus ([link](https://github.com/duyvuleo/VNTC))

|                | Train set  | Test set  |
|----------------|:----------:|----------:|
|Chinh tri xa hoi|   5219     |  7567     |
|Doi song        |   3159     |  2036     |
|Khoa hoc        |   1820     |  2096     |
|Kinh doanh      |   2552     |  5276     |
|Phap luat       |   3868     |  3788     |
|Suc khoe        |   3384     |  5417     |
|The gioi        |   2898     |  6716     |
|The thao        |   5298     |  6667     |
|Van hoa         |   3080     |  6250     |
|Vi tinh         |   2481     |  4560     |

## Insall package requirement
<pre>
pip install -r requirements.txt 
</pre>

## Preprocessing News
- Convert all word into unicode composed 
- Convert all words into the rules for placing the old version bar marks vietnamese ([Reference](https://vi.wikipedia.org/wiki/Quy_t%E1%BA%AFc_%C4%91%E1%BA%B7t_d%E1%BA%A5u_thanh_trong_ch%E1%BB%AF_qu%E1%BB%91c_ng%E1%BB%AF))
- Tokenizer VietNamese word(Đại học Bách Khoa Hà Nội ==> Đại_học Bách_Khoa Hà_Nội)
- Remove all punctuation ,number and stop word 
- Remove all word which has length <3
- Concat all file into 1 file csv

## Model
- Naive Bayes
- KNN
- BiLSTM
- BERT 
- BERT+BiLSTM

## Training
- DeepLearning
<pre>
python3 train.py --model "BERT"//"LSTM" [REQUIRED] 
                 --batch_size 32 [REQUIRED] 
                 --n_epochs 100 [REQUIRED]  
                 --embedding_dim 300 [OPTIONAL]  
                 --hidden_dim 256 [OPTIONAL] 
                 --lr 1e-5 [OPTIONAL] 

</pre>

- MachineLearning 
<pre>
python3 ML.py --model "NB"//"KNN" [REQUIRED]  
              --n_centers 20 [OPTIONAL]
              --n_components 300 [OPTIONAL]
              --plot_result False [OPTIONAL]        
</pre>

## Testing
- Running in terminal with file text
<pre>
python3 infer.py --model "NB"//"KNN"//"LSTM"//"BERT" [REQUIRED]
                 --file_test "a.txt"   
</pre>
- Running with web UI
<pre>
python3 runner.py 
</pre>

