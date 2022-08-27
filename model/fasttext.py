import torch
import torch.nn as nn
import numpy as np


class FastTextModel(nn.Module):

    def __init__(self,vocab_size,embed_dim,max_length,label_size):
        super().__init__()
        self.vocab_size=vocab_size
        self.embed_dim=embed_dim
        self.max_length=max_length
        self.label_size=label_size

        self.embedding=nn.Embedding(self.vocab_size,self.embed_dim) #词嵌入矩阵
        self.avg_pool=nn.AvgPool1d(kernel_size=max_length,stride=1) #对于文本分类来说，平均池化就像是一种对文本整体的抽象表征
        self.linear=nn.Linear(self.embed_dim,self.label_size)

    def forward(self,x):
        """
        :param x: [Batch_size,seqleng]
        :return:
        """
        self.x=self.embedding(x) #[Batch_size,seqlen,embed_dim]
        self.out=self.avg_pool(self.x.permute(0,2,1)) #[Batch_size,embed_dim,seqlen]======>avg=====>[Batch_size,embed_dim,1]
        self.out=self.out.squeeze(-1) #[Batch_size,embed_dim]
        self.logits=self.linear(self.out) #[Batch_size,num_classes]
        return self.logits  #返回未经过softmax的logits

if __name__ == '__main__':
    fastext=FastTextModel(vocab_size=100,embed_dim=128,max_length=20,label_size=5)
    input=np.random.randint(low=0,high=10,size=(16,20))
    input=torch.LongTensor(input) #embedding要求输入得是torch.Tensor，不能是numpy类型的数据
    logits=fastext(input)
    print(logits)