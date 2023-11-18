import torch as t
import torch.nn as nn
import torch.utils.data
from sklearn.model_selection import train_test_split

"""
#这里要将这些文件同存在的目录文件夹mark为root 文件才可以导入包；或者按照网上说的用import sys来修改，再或者是在site package中添加py文件的绝对路径（但不要带py）
# 或者将用os将工作目录设为当前同级目录，再import
import sys
sys.path.append('.')

import os
os.chdir('/Users/hongruoyu/PycharmProjects/package111111/自然语言处理/sentimen-analysis-based-on-sentiment-lexicon-and-deep-learning-master/data')
"""

#import os
#os.chdir('/Users/hongruoyu/PycharmProjects/package111111/自然语言处理/sentimen-analysis-based-on-sentiment-lexicon-and-deep-learning-master/fruit')

#import sys
#sys.path.append('.')

from data_loader import MyData
from model import SLCABG
import data_util
from models_CNN_SA import CNN
from models_CNN_SA import Config


device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
SENTENCE_LENGTH = 15
WORD_SIZE = 24389    #35000
EMBED_SIZE = 768


if __name__ == '__main__':
    sentences, label, word_vectors = data_util.process_data(SENTENCE_LENGTH, WORD_SIZE, EMBED_SIZE)
    x_train, x_test, y_train, y_test = train_test_split(sentences, label, test_size=0.2)

    train_data_loader = torch.utils.data.DataLoader(MyData(x_train, y_train), 32, True)
    test_data_loader = torch.utils.data.DataLoader(MyData(x_test, y_test), 32, False)

    config=Config()
    net = CNN( word_vectors,config).to(device)
    optimizer = t.optim.Adam(net.parameters(), 0.01)
    criterion = nn.CrossEntropyLoss() #交叉熵损失
    tp = 1
    tn = 1
    fp = 1
    fn = 1

    require_improvement = config.early_stop
    dev_best_loss = float('inf')
    # Record the iter of batch that the loss of the last improvement
    last_improve = 0
    # Whether the result has not improved for a long time
    flag = False
    j = 0

    for epoch in range(15):
        for i, (cls, sentences) in enumerate(train_data_loader):
            optimizer.zero_grad()
            sentences = sentences.type(t.LongTensor).to(device)
            cls = cls.type(t.LongTensor).to(device)
            out = net(sentences)
            _, predicted = torch.max(out.data, 1)
            predict = predicted.cpu().numpy().tolist()
            pred = cls.cpu().numpy().tolist()
            for f, n in zip(predict, pred):
                if f == 1 and n == 1:
                    tp += 1
                elif f == 1 and n == 0:
                    fp += 1
                elif f == 0 and n == 1:
                    fn += 1
                else:
                    tn += 1
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * r * p / (r + p)
            acc = (tp + tn) / (tp + tn + fp + fn)
            loss = criterion(out, cls).to(device)
            loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0:
                print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())
                print('acc', acc, 'p', p, 'r', r, 'f1', f1)

                dev_loss=loss.item()

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    #torch.save(model.state_dict(), config.save_path)
                    last_improve = j

               # model.train()
                #model = model.to(device)

            if j - last_improve > require_improvement:
                # Stop training if the loss of dev dataset has not dropped
                # exceeds args.early_stop batches
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break #两次break跳出两层循环
            j += 1
        if flag:
            break



    net.eval()
    print('==========================================================================================')
    with torch.no_grad():
        tp = 1
        tn = 1
        fp = 1
        fn = 1
        for cls, sentences in test_data_loader:
            sentences = sentences.type(t.LongTensor).to(device)
            cls = cls.type(t.LongTensor).to(device)
            out = net(sentences)
            _, predicted = torch.max(out.data, 1)
            predict = predicted.cpu().numpy().tolist()
            pred = cls.cpu().numpy().tolist()
            for f, n in zip(predict, pred):
                if f == 1 and n == 1:
                    tp += 1
                elif f == 1 and n == 0:
                    fp += 1
                elif f == 0 and n == 1:
                    fn += 1
                else:
                    tn += 1
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * r * p / (r + p)
        acc = (tp + tn) / (tp + tn + fp + fn)
        print('acc', acc, 'p', p, 'r', r, 'f1', f1)
