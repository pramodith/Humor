import torch
import pandas as pd
import torch.nn as nn
import io
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from data_handler import *
import argparse
import sys


class LSTMModel(nn.Module):

    def __init__(self, train_file_path: str, dev_file_path: str, test_file_path: str, train_batch_size: int,
                 test_batch_size: int, lr: float, vocab_size : int):
        '''

        :param train_file_path: Path to the train file
        :param test_file_path: Path to the test file
        :param train_batch_size: Size of the batch during training
        :param test_batch_size: Size of the batch during testing
        :param lr: learning rate
        '''

        super(LSTMModel, self).__init__()
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_file_path = train_file_path
        self.dev_file_path = dev_file_path
        self.test_file_path = test_file_path
        self.vocab_size = vocab_size
        self.lr = lr
        # init Flair embeddings
        self.embedding = nn.Embedding(self.vocab_size,300)
        self.lstm = nn.LSTM(300,150,bidirectional=True)
        self.linear1 = nn.Linear(300,100)
        self.linear2 = nn.Linear(100,1)

    def forward(self, *input):
        input = input[1]
        out = self.embedding(input[1])
        out, (hn,cn) = self.lstm(out)
        out = self.linear2(self.linear1(torch.tanh(out[:,-1,:])))
        return out

    def train_loop(self,mode=True):
        if torch.cuda.is_available():
            self.cuda()
        optimizer = optim.Adam(self.parameters(), lr=self.lr,weight_decay=0.001)
        loss = nn.MSELoss()
        best_loss  = sys.maxsize
        train_dataloader,val_dataloader = get_dataloaders(self.train_file_path,"train",self.train_batch_size)
        for epoch in range(5):
            total_prev_loss = 0
            for (batch_num, batch) in enumerate(train_dataloader):
                # If gpu is available move to gpu.
                if torch.cuda.is_available():
                    input1 = batch[0].cuda()
                    input2 = batch[1].cuda()
                    locs = batch[2].cuda()
                    gt = batch[3].cuda()
                else:
                    input1 = batch[0]
                    input2 = batch[1]
                    locs = batch[2]
                    gt = batch[3]
                loss_val = 0
                self.train()
                # Clear gradients
                optimizer.zero_grad()
                final_scores = self.forward((input1,input2,locs))
                loss_val += loss(final_scores.squeeze(1), gt.float())
                # Compute gradients
                loss_val.backward()
                total_prev_loss += loss_val.item()
                print("Loss for batch" + str(batch_num) + ": " + str(loss_val.item()))
                # Update weights according to the gradients computed.
                optimizer.step()
            # Don't compute gradients in validation step
            with torch.no_grad():
                # Ensure that dropout behavior is correct.
                self.eval()
                mse_loss = 0
                for (val_batch_num, val_batch) in enumerate(val_dataloader):
                    if torch.cuda.is_available():
                        input1 = val_batch[0].cuda()
                        input2 = val_batch[1].cuda()
                        locs = val_batch[2].cuda()
                        gt = val_batch[3].cuda()
                    else:
                        input1 = val_batch[0]
                        input2 = val_batch[1]
                        locs = val_batch[2]
                        gt = val_batch[3]
                    final_scores = self.forward((input1,input2,locs))
                    mse_loss+=mean_squared_error(final_scores.cpu().detach().squeeze(1),gt.cpu().detach())
                if mse_loss<best_loss:
                    torch.save(self.state_dict(), "model_" + str(epoch) + ".pth")
                    best_loss = mse_loss
                print("Validation Loss is " + str(mse_loss /(val_batch_num+1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",action="store",type=int,default=4,required=False)
    parser.add_argument("--train_file_path",type=str,default="../data/task-1/train.csv",required=False)
    parser.add_argument("--dev_file_path", type=str, default="../data/task-1/dev.csv", required=False)
    parser.add_argument("--test_file_path", type=str, default="../data/task-1/dev.csv", required=False)
    parser.add_argument("--model_file_path", type=str, default="../models/model_4.pth", required=False)
    parser.add_argument("--lr",type=float,default=0.0001,required=False)
    args = parser.parse_args()
    obj = LSTMModel(args.train_file_path,args.dev_file_path,args.test_file_path,args.batch_size,64,args.lr,11728)
    obj.train_loop()
