import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self):
        # 训练配置
        self.seed = 22
        #self.batch_size = 64
        #self.lr = 1e-3
        #self.weight_decay = 1e-4
        self.num_epochs =15 #100 #我们的epoch是15
        self.early_stop = 512 #如果距离上次改进间隔了512次之后都没有改进，就停下来
        self.max_seq_length = 12   # 128 #这个参数等价于我们的   SENTENCE_LENGTH = 12
        self.save_path = './RNN_SA.bin'

        # 模型配置
        self.filter_sizes = (3, 4, 5)
        self.num_filters = 100
        self.dense_hidden_size = 128
        self.dropout = 0.5
        self.embed_size = 768 #词向量
        self.num_outputs = 2




import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, embed,config):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embed, freeze=False)

        self.rnn = nn.RNN(config.embed_size, config.dense_hidden_size )

        self.fc = nn.Linear(config.dense_hidden_size, config.num_outputs)


    def forward(self, text):

        # text = [sent_len, batch_size]
        #print(text.shape)
        embedded = self.embedding(text)
        #print("embedded" + str(embedded.shape))
        # embedded = [sent_len, batch_size, embedding_dim]

        embedded=embedded.permute(1,0,2)  #这里是对调维数，因为rnn需要把时间维数提到最前面，就是12个词为一个句子，对应一个标签。把（32，12）改为（12，32）
        #print("embedded2" + str(embedded.shape))
        output, hidden = self.rnn(embedded)
        #print("out"+str(output.shape))
        #print(hidden)
        # output = [sent_len, batch_size, hid_dim]
        # hidden = [1, batch_size, hid_dim]
        #print("hidden" + str(hidden.shape))

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        #print("hidden(0)" + str(hidden.squeeze(0).shape))
        return self.fc(hidden.squeeze(0))




