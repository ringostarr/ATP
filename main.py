import DataUtils as du
import logging
import numpy as np
import torch as tr
SourceDir = "Full path of dataset directory"

parser = du.DataParser(SourceDir, True,1)
print(parser.display_stats(parser.train_conjectures))
print()


class CNNLSTMNet(tr.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = tr.nn.Conv1d(1, 20, 1)
        self.conv2 = tr.nn.Conv1d(1, 20, 1)
        self.pooling = tr.nn.MaxPool2d(2, 2)
        self.Lstm = tr.nn.LSTM(40, 32, 1,batch_first=True)
        self.fc1 = tr.nn.Linear(32, 1)
        #self.fc2 = tr.nn.Linear(256, 128)

        self.relu = tr.nn.LeakyReLU()
        self.sig = tr.nn.Sigmoid()

    def forward(self,input1,input2):
        x = tr.tensor(input1).float().unsqueeze(1)
        y = tr.tensor(input2).float().unsqueeze(1)

        x = self.conv1(x) # -- Conv for Steps
        y = self.conv2(y) # -- Conv for conjecture
        z = tr.cat((x,y),dim=1)  # - concat output
        z = tr.transpose(z,1,2)
        out , (h,c) = self.Lstm(z) # -- LSTM layer
        outputp = self.relu(self.fc1(h))
        #outputp = self.relu(self.fc2(outputp))

        outputp = self.sig(outputp)

        return outputp.flatten()

class CNNOnlyNet(tr.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = tr.nn.Conv1d(1,20,1)
        self.conv2 = tr.nn.Conv1d(1,20,1)
        self.conv3 = tr.nn.Conv1d(40,1,1)
        self.pooling = tr.nn.MaxPool1d(3)
        #self.Lstm = tr.nn.LSTM(154, 10, 1)
        self.fc1 = tr.nn.Linear(256, 1)
        self.norm = tr.nn.BatchNorm1d(1)
        #self.fc2 = tr.nn.Linear(256, 128)
        self.dropo = tr.nn.Dropout(0.5)
        self.relu = tr.nn.ReLU()
        self.sig = tr.nn.Sigmoid()

    def forward(self,input1,input2):


        x = tr.tensor(input1).float().unsqueeze(1)
        y = tr.tensor(input2).float().unsqueeze(1)

        x = self.conv1(x) # -- Conv for Steps
        y = self.conv2(y) # -- Conv for conjecture
        z= tr.cat((x,y),dim=1)
        z = self.norm(self.relu(self.conv3(z))) # -- Conv for conjecture
        z=z.reshape(128,-1)
        outputp = self.relu(self.fc1(z))
        outputp = self.sig(outputp)
        return outputp.flatten()


class LSTMNet(tr.nn.Module):
    def __init__(self):
        super().__init__()

        self.Lstm = tr.nn.LSTM(2,32,1,batch_first=True)

        self.relu = tr.nn.LeakyReLU()

        self.fc1 = tr.nn.Linear(32,16)
        self.fc2 = tr.nn.Linear(16,1)

        self.sig = tr.nn.Sigmoid()
    def forward(self,input1,input2=0):

        x= tr.tensor(input1).float() # Prediction info - +- features
        y= tr.tensor(input2).float() # Data info - Embeddings of conjecture name and dependencies
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        x = tr.cat((x,y),dim=2)
        #y = y.unsqueeze(0)
        out1,(ht,ct) = self.Lstm(x) # processes DataInfo in first RNN , discard output
        #ht = ht.flatten()
        x = self.relu(self.fc1(ht))

        x = self.sig(self.relu(self.fc2(x)))
        return x.flatten()

LSTNnet =LSTMNet()
CNNLstem = CNNLSTMNet()
CNNonly = CNNOnlyNet()

TrainSteps = 1000

import torch.optim as optim
import random
from torch.utils.data import DataLoader,TensorDataset
criterion = tr.nn.BCELoss()
optimizer = optim.Adam(CNNonly.parameters(),lr=0.0001)

val_acc_list = []
Test_acc_list = []
#Integer Encoding
TrainingLoss_List = []
for epoch in range(1000):
    running_loss = 0.0

    for i in range(20):

        train_generator, labels = parser.draw_random_batch_of_steps_and_conjectures(split='train', encoding='integer',max_len=256,batch_size=128)

        EncodedSteps = train_generator[0]
        EncodedConj = train_generator[1]
        Train_info = np.concatenate((EncodedSteps,EncodedConj))
        Labels = labels
        optimizer.zero_grad()
        outputs = CNNonly(EncodedSteps,EncodedSteps) # change
        label = tr.Tensor([Labels])
        loss = criterion(tr.Tensor(outputs),label.flatten())
        loss.backward()
        optimizer.step()
        running_loss  =running_loss+ loss.item()
    print(f'epoch [{epoch + 1},Trainting loss]: {running_loss:.3f}')
    TrainingLoss_List.append(running_loss)
    val_data, val_labels = parser.draw_random_batch_of_steps_and_conjectures(split='val', encoding='integer',max_len=256,batch_size=128)
    EncodedSteps = val_data[0]
    EncodedConj = val_data[1]
    Val_info = np.concatenate((EncodedSteps, EncodedConj))

    valOutputs = CNNonly(EncodedSteps,EncodedConj)
    #val_loss = criterion(tr.Tensor(valOutputs), tr.tensor([val_labels.flatten()]))
  # val_loss_list.append(val_loss)
    for i in range(len(valOutputs)):

        if valOutputs[i] > 0.5:

            valOutputs[i] = 1
        else:
            valOutputs[i] = 0
    valOutputs = tr.tensor(valOutputs.detach().numpy().flatten())
    Acc_list = tr.eq(valOutputs, tr.tensor(val_labels.flatten()))
    True_pred = len(Acc_list[Acc_list == True]) / len(Acc_list)
    val_acc_list.append((True_pred))

    print("Val Accuracy:", True_pred * 100)
   # print()

    print()
    Test_data, Test_labels = parser.draw_random_batch_of_steps_and_conjectures(split='test', encoding='integer',
                                                                                        max_len=256,
                                                                                        batch_size=128)
    EncodedSteps = Test_data[0]
    EncodedConj = Test_data[1]
    Val_info = np.concatenate((EncodedSteps,EncodedConj))

    TestOutputs = CNNonly(EncodedSteps,EncodedConj)
    for i in range(len(valOutputs)):

                if TestOutputs[i]>0.5:
                    # 0-45 - negative , 45-55 - unknown , 55 - +
                    TestOutputs[i] = 1
                else:
                    TestOutputs[i]=0
    TestOutputs = tr.tensor(TestOutputs.detach().numpy().flatten())

    Acc_list_test = tr.eq(TestOutputs,tr.tensor(Test_labels.flatten()))
    True_pred = len(Acc_list[Acc_list_test==True])/len(Acc_list)
    Test_acc_list.append((True_pred))
    print("Testing Accuracy:",True_pred*100)

import matplotlib.pyplot as plt

epochs = range(0,1000)
plt.plot(epochs,TrainingLoss_List,label='Training Loss')
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

plt.plot(epochs,Test_acc_list,Label='Test Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.show()
plt.plot(epochs,val_acc_list,Label='Test Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Val Accuracy")
plt.show()
print()
