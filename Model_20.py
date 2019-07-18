import torch.nn as nn
import torch 

from collections import OrderedDict


class Model_20(nn.Module):

    def __init__(self, vocab_size, dim, embeddings):
        super(Model_20, self).__init__()
        self.vocab_size = vocab_size 
        self.dim = dim
        self.embedding = nn.Embedding(self.vocab_size, self.dim)
        self.convnet = nn.Sequential(OrderedDict([
            #('embed1', nn.Embedding(self.vocab_size, self.dim)),
            ('c1', nn.Conv1d(100, 128, 5)),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool1d(5)),
            ('c2', nn.Conv1d(128, 128, 5)),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool1d(5)),
            ('c3', nn.Conv1d(128, 128, 5)),
            ('relu3', nn.ReLU()),
            ('maxpool3', nn.MaxPool1d(35)),
        ]))
    
        self.embedding.weight = nn.Parameter(torch.FloatTensor(embeddings))
        #copy_((embeddings))
        self.embedding.weight.requires_grad = False
    
        self.fc = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(128, 128)),
            ('relu4', nn.ReLU()),
            ('f5', nn.Linear(128, 20)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        
        output = self.embedding(img)
        output.transpose_(1,2)
        output = self.convnet(output)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        
        return output

