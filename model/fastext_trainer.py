import torch.nn as nn
import torch
import json
import torch.optim as optim
import pickle
from model import logger
from base.basetrainer import BaseTrainer
import numpy as np
from  model.fasttext import FastTextModel
from torch.utils.data import  DataLoader

class FastextTrainer(BaseTrainer):

    def __init__(self,vocab_size,embed_dim,max_length,label_size,learning_rate=0.3,use_cuda=True,save_dir="../config/train.json"):
        self.embed_dim=embed_dim
        self.vocab_size=vocab_size
        self.max_length=max_length
        self.label_size=label_size
        self.save_dir=save_dir
        self.cuda=use_cuda
        self.learning_rate=learning_rate
        self.fastextmodel=FastTextModel(vocab_size=vocab_size,embed_dim=embed_dim,max_length=max_length,
                                        label_size=label_size) #实例化一个模型


    def _save_config(self):
        info={
            "vocab_size":self.vocab_size,
            "embed_dim":self.embed_dim,
            "max_length":self.max_length,
            "label_size":self.label_size,
            "learning_rate":self.learning_rate
        }
        with open(self.save_dir+"/train_config.json","w") as f:
            f.write(json.dumps(info,indent=4))

    def train(self,epoches,train_data_iter,batch_size=64):
        if self.cuda and torch.cuda.is_available():
            self.fastextmodel=self.fastextmodel.cuda()
        optimizer=optim.Adam(self.fastextmodel.parameters(), lr=self.learning_rate)
        total_loss=0
        cretierion=nn.CrossEntropyLoss()
        best_loss = float("inf")
        loss_buff = []  # 保存最近的10个valid loss
        max_loss_num = 10
        for epoch in range(epoches):
            self.fastextmodel.train()
            for step,batch_data,labels in enumerate(train_data_iter):
                labels=labels.view(-1)
                if self.cuda and torch.cuda.is_available():
                    batch_data=batch_data.cuda()
                    labels = labels.cuda()
                logits=self.fastextmodel(batch_data)
                loss=cretierion(logits.view(-1,self.label_size),torch.autograd.Variable(labels).long())
                total_loss+=loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logger.info("current epoch is {},current step is {}.loss is {}".format(epoch,step,loss.item()))
            state_dict=self.fastextmodel.state_dict()
            torch.save(state_dict,"{}/{}".format(self.save_dir,"fastext.bin"))


    def validate(self,validate_dataset,batch_size):
        validate_data = DataLoader(validate_dataset, shuffle=True, batch_size=batch_size, num_workers=0)
        self.fastextmodel.eval()
        with torch.no_grad():
            logits=self.fastextmodel()
            pass
        pass

    def get_test_result(self,test_data_iter, testdata_len):
        # 生成测试结果
        self.fastextmodel.eval()
        true_sample_num = 0
        for data, label in test_data_iter:
            if self.cuda and torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()
            else:
                data = torch.autograd.Variable(data).long()
            out = self.fastextmodel(data)
            true_sample_num += np.sum((torch.argmax(out, 1) == label.long()).cpu().numpy())
        acc = true_sample_num / testdata_len
        return acc
if __name__ == '__main__':
    with open("../data/train_data.pkl","rb") as f:
        Train_dataset=pickle.load(f)
    with open("../data/test_data.pkl","rb") as f:
        Test_dataset=pickle.load(f)

    Train_iter=DataLoader(Train_dataset,shuffle=True,batch_size=64)
    Test_iter=DataLoader(Test_dataset,shuffle=True,batch_size=64)
    model_trainer=FastextTrainer(vocab_size=Train_dataset.uniwords_num+100000,embed_dim=32,max_length=100,label_size=54,learning_rate=1e-3,use_cuda=False,save_dir="../config")
    model_trainer.train(10,Train_iter,batch_size=64)
