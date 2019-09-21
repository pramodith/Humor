import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score,f1_score,mean_squared_error
import sys
from pytorch_transformers import BertForMaskedLM,BertTokenizer
from pytorch_pretrained_bert import BertAdam
import argparse
from data_handler import *
import torchsummary


class RBERT(nn.Module):

    def __init__(self,train_file_path : str, dev_file_path : str, test_file_path : str, lm_file_path : str, train_batch_size : int,test_batch_size : int,lr : float, lm_weights_file_path : str):
        '''

        :param train_file_path: Path to the train file
        :param test_file_path: Path to the test file
        :param train_batch_size: Size of the batch during training
        :param test_batch_size: Size of the batch during testing
        :param lr: learning rate
        '''

        super(RBERT, self).__init__()
        self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.load_joke_lm_weights(lm_weights_file_path)
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_file_path = train_file_path
        self.lm_file_path = lm_file_path
        self.dev_file_path = dev_file_path
        self.test_file_path = test_file_path
        self.lr = lr
        self.linear_reg1 = nn.Sequential(
                  nn.Dropout(0.3),
                  nn.Linear(768*3,100),
                  )
        self.final_linear = nn.Sequential(nn.Dropout(0.3),nn.Linear(100,1))

    def load_joke_lm_weights(self,lm_path : str):
        self.bert_model.load_state_dict(torch.load(lm_path))

    def pre_train_bert(self):
        optimizer = BertAdam(self.bert_model.parameters(),2e-5)
        train_dataloader = get_bert_lm_dataloader(self.lm_file_path)
        print("Training LM")
        if torch.cuda.is_available():
            self.bert_model.cuda()
        for epoch in range(1):
            print("Epoch : " +str(epoch))
            for ind,batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    inp = batch[0].cuda()
                else:
                    inp = batch[0]
                outputs = self.bert_model(inp,masked_lm_labels=inp)
                loss, prediction_scores = outputs[:2]
                loss.backward()
                print("Loss is :" + str(loss.item()))
                optimizer.step()
                if ind>2000:
                    break
        print("LM training done")
        torch.save(self.bert_model.state_dict(),"lm_joke_bert.pth")


    def forward(self, *input):
        '''
        :param input: input[0] is the sentence, input[1] are the entity locations , input[2] is the ground truth
        :return: Scores for each class
        '''

        input = input[0]
        #output_per_seq1, _ = self.bert_model(input[0].long())
        output_per_seq2, _ = self.bert_model(input[1].long())
        sent_emb = (torch.mean(output_per_seq2,1))
        final_scores = []
        '''
        Obtain the vectors that represent the entities and average them followed by a Tanh and a linear layer.
        '''
        for (i, loc) in enumerate(input[2]):
            # +1 is to ensure that the symbol token is not considered
            #entity1 = torch.mean(output_per_seq1[i, loc[0] + 1:loc[1]], 0)
            entity2 = torch.mean(output_per_seq2[i, loc[2] + 1:loc[3]], 0)
            entity2_max = torch.max(output_per_seq2[i, loc[2] + 1:loc[3]], 0)
            #diff = torch.sub(entity1,entity2)
            #prod = entity1*entity2
            sent_out = torch.tanh(self.linear_reg1(torch.cat((sent_emb[i],entity2,entity2_max),0)))
            final_out = self.final_linear(sent_out)
            final_scores.append(final_out)
        return torch.stack((final_scores))

    def train(self,mode=True):
        if torch.cuda.is_available():
            self.cuda()
        self.bert_model = self.bert_model.bert
        optimizer = optim.Adam(self.parameters(), lr=self.lr,weight_decay=0.001)
        loss = nn.MSELoss()
        best_loss  = sys.maxsize
        train_dataloader,val_dataloader = get_dataloaders_bert(self.train_file_path,"train",self.train_batch_size)
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
                self.linear_reg1.train()
                self.final_linear.train()
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
                self.bert_model.eval()
                self.linear_reg1.eval()
                self.final_linear.eval()
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

    def predict(self,model_path=None):

        '''
        This function predicts the classes on a test set and outputs a csv file containing the id and predicted class
        :param model_path: Path of the model to be loaded if not the current model is used.
        :return:

        '''
        if torch.cuda.is_available():
            self.cuda()
        if model_path:
            self.load_state_dict(torch.load(model_path))
        test_dataloader = get_dataloaders_bert(self.test_file_path,"test")
        self.bert_model.eval()
        self.linear_reg1.eval()
        self.final_linear.eval()
        with torch.no_grad():
            with open("task-1-output.csv","w+") as f:
                f.writelines("id,pred\n")
                for ind,batch in enumerate(test_dataloader):
                    if torch.cuda.is_available():
                        input1 = batch[0].cuda()
                        input2 = batch[1].cuda()
                        locs = batch[2].cuda()
                        id = batch[3].cuda()
                    else:
                        input1 = batch[0]
                        input2 = batch[1]
                        locs = batch[2]
                        id = batch[3]
                    final_scores = self.forward((input1,input2,locs)).view(-1)
                    for cnt,pred in enumerate(final_scores):
                        f.writelines(str(id[cnt].item())+","+str(pred.item())+"\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",action="store",type=int,default=4,required=False)
    parser.add_argument("--train_file_path",type=str,default="../data/task-1/train.csv",required=False)
    parser.add_argument("--dev_file_path", type=str, default="../data/task-1/dev.csv", required=False)
    parser.add_argument("--test_file_path", type=str, default="../data/task-1/dev.csv", required=False)
    parser.add_argument("--lm_file_path", type=str, default="../data/task-1/shortjokes.csv", required=False)
    parser.add_argument("--lm_weights_file_path", type=str, default="../models/lm_joke_bert.pth", required=False)
    parser.add_argument("--model_file_path", type=str, default="../models/model_4.pth", required=False)
    parser.add_argument("--predict", type=str, default=False,required=False)
    parser.add_argument("--lr",type=float,default=0.0001,required=False)
    args = parser.parse_args()
    obj = RBERT(args.train_file_path,args.dev_file_path,args.test_file_path,args.lm_file_path,args.batch_size,64,args.lr,args.lm_weights_file_path)
    #obj.pre_train_bert()
    if args.predict=='true':
        obj.predict(args.model_file_path)
    else:
        obj.train()
