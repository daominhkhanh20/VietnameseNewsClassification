import unicodedata 
import regex 
from pyvi import ViTokenizer
from string import punctuation
import regex
from underthesea import word_tokenize
from matplotlib import pyplot as plt 

with open('stopword.txt',encoding='utf-8') as file:
    stop_words=[stop_word.replace(" ","_") for stop_word in file.read().splitlines()]

bang_nguyen_am=[['a', 'à', 'á', 'ả', 'ã', 'ạ'],
                ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ'],
                ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ'],
                ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ'],
                ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ'],
                ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị'],
                ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ'],
                ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ'],
                ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ'],
                ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ'],
                ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự'],
                ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ']]

nguyen_am_to_ids = {}

for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)

def convert_unicode(text):
    return unicodedata.normalize("NFC",text)

def tokenizer_word(text):
    return ViTokenizer.tokenize(text) 



def remove_redundancy(text):
    new_punctuation=punctuation.replace('_','')
    table=str.maketrans('','',new_punctuation)
    text=text.translate(table)
    text=regex.sub("\s\s+"," ",text)
    return text

def chuan_hoa_dau_tu_tieng_viet(word):
    if not is_valid_vietnam_word(word):
        return word
 
    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 0:
            nguyen_am_index.append(index)
    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else bang_nguyen_am[9][dau_cau]
            return ''.join(chars)
        return word
 
    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = bang_nguyen_am[x][dau_cau]
            return ''.join(chars)
 
    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
        else:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    else:
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    return ''.join(chars)
 
 
def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True

def chuan_hoa_dau_cau_tieng_viet(sentence):
    #sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = regex.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
        if len(cw) == 3:
            cw[1] = chuan_hoa_dau_tu_tieng_viet(cw[1])
        words[index] = ''.join(cw)
    return ' '.join(words)

def process_data(text):
    text=convert_unicode(text)
    text=chuan_hoa_dau_cau_tieng_viet(text)
    text=tokenizer_word(text)
    text=remove_redundancy(text)
    words=[]
    temp=text.strip().split(' ')
    for word in temp:
        if len(word)<3 or word.isnumeric()== True or word in stop_words or word.lower() in stop_words:
            continue
        words.append(word)
    text=' '.join(words)
    return text