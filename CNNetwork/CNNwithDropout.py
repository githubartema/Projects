import numpy as np # linear algebra
import torch
import matplotlib.pyplot as plt
import glob
import cv2
import torch.optim as optim
import torch.nn as nn
import random
import torch.nn.functional as F
from random import shuffle

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.determenistic = True

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.dropout = torch.nn.Dropout(p=0.3)

        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1)
        self.conv1_bn = torch.nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)    
        self.conv2_bn = torch.nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2,2)
  
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)
        #self.conv3_bn = torch.nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 16 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool1(self.conv1_bn(F.relu(self.conv1(x))))
        x = self.pool2(self.conv2_bn(F.relu(self.conv2(x))))
        x =F.relu(self.conv3(x))
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

def shuffle_data():
    global addrs
    global labels
    new_list = list(zip(addrs, labels))
    shuffle(new_list)
    addrs, labels = zip(*new_list)

cat_dog_train_path = '/Users/Artem/AppData/CatsDogs/train/*.jpg'
addrs = glob.glob(cat_dog_train_path)
labels = [ [1,0] if 'cat' in addr else [0,1] for addr in addrs ] 

shuffle_data()

train_addrs = addrs[0:int(0.95*len(addrs))][0:18000]
train_labels = labels[0:int(0.95*len(labels))][0:18000]

test_addrs = addrs[int(0.95*len(addrs)):][0:600]
test_labels = labels[int(0.95*len(labels)):][0:600]

train_data = []
for i in range(len(train_addrs)):
    addr = train_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    train_data.append([np.array(img), np.array(train_labels[i])])

test_data = []
for i in range(len(test_addrs)):
    addr = test_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_data.append([np.array(img), np.array(test_labels[i])])


train_batch = torch.FloatTensor(np.array([i[0] for i in train_data])).reshape(-1,64,64,3)
train_batch = train_batch.permute(0,3,1,2)

train_target = torch.LongTensor(np.array([i[1] for i in train_data]))

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0001)

minibatch_size = 100

test_batch = torch.FloatTensor(np.array([i[0] for i in test_data])).reshape(-1,64,64,3)
test_batch = test_batch.permute(0,3,1,2)

test_target = torch.LongTensor(np.array([i[1] for i in test_data]))

accuracy_history = []

train_batch /= 255.
test_batch /= 255.

for epoch in range(70):
    order = np.random.permutation(len(train_batch))
    for start_index in range(0, len(train_batch), minibatch_size):
        optimizer.zero_grad() 
        minibatch_indexes = order[start_index:start_index+minibatch_size]
        train_minibatch = train_batch[minibatch_indexes]
        train_minitarget = train_target[minibatch_indexes]
        preds = net(train_minibatch)
        loss_value = loss(preds, torch.max(train_minitarget, 1)[1])
        loss_value.backward()
        optimizer.step()
    print(epoch, ':', loss_value.item(), " - train_loss")
    accuracy = (preds.argmax(dim=1) == torch.max(train_minitarget, 1)[1]).float().mean().data.cpu()
    print(accuracy, " - train_accuracy")

    with torch.no_grad():
        net.eval()   
        test_preds = net(test_batch)
        print(loss(test_preds, torch.max(test_target, 1)[1]).item(), " - test_loss")  
        accuracy_history.append((test_preds.argmax(dim=1) == torch.max(test_target, 1)[1]).float().mean().data.cpu())
        print(accuracy_history[-1], "test_accuracy" + "\n")
        if epoch == 6:
           torch.save(net.state_dict(), '/Users/Artem/Documents/trained_cnn.pth')

    net.train()

plt.plot(accuracy_history) 





