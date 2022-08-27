#coding:utf-8
from torch.utils import data
import pickle,os
import csv
import nltk
import numpy as np


class AG_Data(data.DataLoader):

    def __init__(self,data_path,min_count,max_length,n_gram=2,word2id = None,uniwords_num = 0):
        self.path =data_path
        self.n_gram = n_gram
        self.load(data_path)
        if word2id==None:
            self.get_word2id(self.data,min_count)
        else:
            self.word2id = word2id
            self.uniwords_num = uniwords_num
        self.data = self.convert_data2id(self.data,max_length)
        self.data = np.array(self.data)
        self.y = np.array(self.y)

    def load(self,data_path,lowercase=True):
        self.label = []
        self.data = []
        with open(self.path,"r") as f:
            datas = list(csv.reader(f,delimiter=',', quotechar='"'))
            for row in datas:
                self.label.append(int(row[0])-1)
                txt = " ".join(row[1:])
                if lowercase:
                    txt = txt.lower()
                txt = nltk.word_tokenize(txt)
                new_txt=  []
                for i in range(0,len(txt)):
                    for j in range(self.n_gram):
                        if j<=i:
                            new_txt.append(" ".join(txt[i-j:i+1])) #添加ngram信息
                self.data.append(new_txt)
        self.y = self.label


    def get_word2id(self,datas,min_count=3):
        word_freq = {}
        for data in datas:
            for word in data:
                if word_freq.get(word)!=None:
                    word_freq[word]+=1
                else:
                    word_freq[word] = 1
        word2id = {"<pad>":0,"<unk>":1}
        for word in word_freq:
            if word_freq[word]<min_count or " " in word:
                continue
            word2id[word] = len(word2id)
        self.uniwords_num = len(word2id)  #添加unigram信息

        for word in word_freq:
            if word_freq[word]<min_count or " " not in word:
                continue
            word2id[word] = len(word2id) #添加bigram信息
        self.word2id = word2id


    def convert_data2id(self,datas,max_length):
        for i,data in enumerate(datas):
            for j,word in enumerate(data):
                if " " not in word:
                    datas[i][j] = self.word2id.get(word,1)
                else:
                    datas[i][j] = self.word2id.get(word, 1)%100000+self.uniwords_num
                    #datas[i][j] = self.word2id.get(word, 1)
            datas[i] = datas[i][0:max_length]+[0]*(max_length-len(datas[i]))
        return datas
    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.y[idx]
        return X, y

    def __len__(self):
        return len(self.label)

if __name__=="__main__":
    #nltk.download('punkt') #下载到c盘去了，可以选择指定的目录进行下载
    if not os.path.exists("train_data.pkl"):
        ag_data_train = AG_Data("AG/train.csv",3,100)
        ag_data_test=AG_Data("AG/test.csv",3,100,word2id=ag_data_train.word2id,uniwords_num=ag_data_train.uniwords_num)

    else:
        with open("./test_data.pkl","rb") as f:
            ag_data_test=pickle.load(f)
            print(len(ag_data_test))
    # with open("./train_data.pkl","wb") as f:
    #     pickle.dump(ag_data_train,f)

    # with open("./test_data.pkl","wb") as f:
    #     pickle.dump(ag_data_test,f)
