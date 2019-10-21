import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score,f1_score,mean_squared_error
import sys
from pytorch_transformers import BertForMaskedLM,DistilBertForMaskedLM
from pytorch_pretrained_bert import BertAdam
import argparse
from data_handler import *
import torchnlp.nn as nn_nlp
import gensim
import numpy as np
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class RBERT(nn.Module):

    def __init__(self,train_file_path : str, dev_file_path : str, test_file_path : str, lm_file_path : str, train_batch_size : int,
                 test_batch_size : int,lr : float, lm_weights_file_path : str,epochs : int, lm_pretrain : str, task : int, train_scratch : str, model_path : str,
                 joke_classification_path : str, add_joke_model : str):
        '''

        :param train_file_path: Path to the train file
        :param test_file_path: Path to the test file
        :param train_batch_size: Size of the batch during training
        :param test_batch_size: Size of the batch during testing
        :param lr: learning rate
        '''

        super(RBERT, self).__init__()
        self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased',output_hidden_states=True)
        if lm_pretrain != 'true':
            pass
            #self.load_joke_lm_weights(lm_weights_file_path)
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_file_path = train_file_path
        self.lm_file_path = lm_file_path
        #self.lstm = nn.LSTM(768*3,768,bidirectional=False)
        self.attention = nn_nlp.Attention(768*2)
        self.dev_file_path = dev_file_path
        self.test_file_path = test_file_path
        self.joke_classification_path = joke_classification_path
        self.lr = lr
        self.task = task
        self.gensim_model = gensim.models.KeyedVectors.load_word2vec_format('../../../word2vec.bin.gz', binary=True)
        self.add_joke_model = add_joke_model
        self.epochs = epochs
        self.linear_joke = nn.Sequential(nn.Dropout(0.3),nn.Linear(768,2))
        self.linear_reg1 = nn.Sequential(
                  nn.Dropout(0.3),
                  nn.Linear(768*6+300,100),
                  )
        if self.task:
            self.final_linear = nn.Sequential(nn.Dropout(0.3),nn.Linear(100,1))
        else:
            self.final_linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(100, 2))

        if train_scratch == 'true':
            self.load_state_dict(torch.load(model_path))

    def init_embeddings(self, ind: np.ndarray):
        ind = np.unique(ind.reshape(-1))
        self.nn_embeddings = torch.nn.Embedding(len(ind),300)
        self.emb_key = {ind[i]:i for i in range(len(ind))}
        emb_values = np.take(self.gensim_model.vectors,ind,0)
        self.nn_embeddings.load_state_dict({'weight':torch.tensor(emb_values)})
        del self.gensim_model

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

    def train_joke_classification(self):
        self.bert_model = self.bert_model.bert
        optimizer = optim.Adam(list(self.bert_model.parameters())+list(self.linear_joke.parameters()), lr=self.lr,weight_decay=0.001)
        train_dataloader, val_dataloader = get_dataloaders_joke_classification(self.joke_classification_path)
        best_accuracy = 0
        loss = nn.CrossEntropyLoss()
        print("Training Joke Model")
        if torch.cuda.is_available():
            self.bert_model.cuda()
            self.linear_joke.cuda()
        for epoch in range(self.epochs):
            loss_val = 0
            print("Epoch : " +str(epoch))
            self.bert_model.train()
            self.linear_joke.train()
            for ind,batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    inp = batch[0].cuda()
                    gt = batch[1].cuda()
                else:
                    inp = batch[0]
                    gt = batch[1]
                outputs,_,_ = self.bert_model(inp)
                outputs = self.linear_joke(outputs[:,0,:])
                loss_val += loss(outputs.squeeze(0), gt.long())
                loss_val.backward()
                print("Loss is :" + str(loss_val.item()))
                optimizer.step()

            self.bert_model.bert.eval()
            self.linear_joke.eval()
            for ind,batch in enumerate(val_dataloader):
                predictions = []
                ground_truth = []
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    inp = batch[0].cuda()
                    gt = batch[1].cuda()
                else:
                    inp = batch[0]
                    gt = batch[1]
                outputs,_,_ = self.bert_model.bert(inp)
                outputs = self.linear_joke(outputs[:,0,:])
                predictions.extend(torch.argmax(outputs.squeeze(0), 1).tolist())
                ground_truth.extend(gt.tolist())
                print("Loss is :" + str(loss_val.item()))
            accuracy = accuracy_score(ground_truth,predictions)
            print(f'''Accuracy is {accuracy}''')
            if accuracy > best_accuracy :
                best_accuracy = accuracy
                torch.save(self.bert_model.state_dict(),"joke_classification_bert.pth")

    def hook_encoder_bert(self,input,output):
        return output

    def forward(self, *input):
        '''
        :param input: input[0] is the sentence, input[1] are the entity locations , input[2] is the ground truth
        :return: Scores for each class
        '''
        final_scores = []

        if self.task==1:
            input = input[0]
            #output_per_seq1,attention_layer_inps = self.bert_model(input[0].long())
            #output_per_seq1 = torch.cat((output_per_seq1, attention_layer_inps[3], attention_layer_inps[5]), 2)
            #output_per_seq1 = output_per_seq1.transpose(0, 1)
            #output_per_seq1, _ = self.lstm(output_per_seq1)
            #output_per_seq1 = output_per_seq1.transpose(0, 1)
            output_per_seq2, _ ,attention_layer_inps = self.bert_model(input[1].long())
            output_per_seq2 = torch.cat((output_per_seq2,attention_layer_inps[8]),2)
            #output_per_seq2 = output_per_seq2.transpose(0,1)
            #output_per_seq2,_ = self.lstm(output_per_seq2)
            #output_per_seq2 = output_per_seq2.transpose(0,1)
            '''
            Obtain the vectors that represent the entities and average them followed by a Tanh and a linear layer.
            '''
            for (i, loc) in enumerate(input[2]):
                # +1 is to ensure that the symbol token is not considered
                #entity1 = torch.mean(output_per_seq1[i, loc[0] + 1:loc[1]], 0)
                entity2 = torch.mean(output_per_seq2[i, loc[2] + 1:loc[3]], 0)
                entity2_max = torch.max(output_per_seq2[i, loc[2] + 1:loc[3]], 0)
                imp_seq = torch.cat((output_per_seq2[i,0:loc[2]+1],output_per_seq2[i,loc[3]:]),0)
                _,attention_score = self.attention(entity2.unsqueeze(0).unsqueeze(0),imp_seq.unsqueeze(0))
                sent_attn = torch.sum(attention_score.squeeze(0).expand(768*2,-1).t()*imp_seq,0)
                #diff = torch.sub(entity1,entity2)
                #prod = entity1*entity2

                '''
                if input[3][i,0] == -1:
                    word2vec_entity1 = torch.zeros(300)
                else:
                    word2vec_entity1 = torch.tensor(self.gensim_model.vectors[input[3][i, 0]])
                if input[3][i,1] == -1:
                    word2vec_entity2 = torch.zeros(300)
                else:
                    word2vec_entity2 = torch.tensor(self.gensim_model.vectors[input[3][i, 1]])
                '''
                word2vec_diff = self.nn_embeddings(torch.tensor(self.emb_key[input[3][i,0].item()]))-self.nn_embeddings(torch.tensor(self.emb_key[input[3][i,1].item()]))
                sent_out = torch.tanh(self.linear_reg1(torch.cat((sent_attn,output_per_seq2[i,0],entity2,word2vec_diff),0)))
                #sent_out = torch.tanh(self.linear_reg1(sent_attn))
                #sent_out = torch.tanh(self.linear_reg1(torch.cat((sent_attn, diff, prod), 0)))
                final_out = self.final_linear(sent_out)
                final_scores.append(final_out)

        if self.task==2:
            input = input[0]
            output_per_seq2,_,attention_layer_inps = self.bert_model(input[1].long())
            output_per_seq2 = torch.cat((output_per_seq2,attention_layer_inps[8]),2)

            '''
            Obtain the vectors that represent the entities and average them followed by a Tanh and a linear layer.
            '''
            for (i, loc) in enumerate(input[2]):
                # +1 is to ensure that the symbol token is not considered
                #entity1 = torch.mean(output_per_seq1[i, loc[0] + 1:loc[1]], 0)
                if input[3]==1:
                    ent_ind = 0
                else:
                    ent_ind = 2

                entity2 = torch.mean(output_per_seq2[i, loc[ent_ind] + 1:loc[ent_ind+1]], 0)
                entity2_max = torch.max(output_per_seq2[i, loc[ent_ind] + 1:loc[ent_ind+1]], 0)
                imp_seq = torch.cat((output_per_seq2[i,0:loc[ent_ind]+1],output_per_seq2[i,loc[ent_ind+1]:]),0)
                _,attention_score = self.attention(entity2.unsqueeze(0).unsqueeze(0),imp_seq.unsqueeze(0))
                sent_attn = torch.sum(attention_score.squeeze(0).expand(768*2,-1).t()*imp_seq,0)
                #diff = torch.sub(entity1,entity2)
                #prod = entity1*entity2

                sent_out = torch.tanh(self.linear_reg1(torch.cat((sent_attn,output_per_seq2[i,0],entity2,word2vec_diff),0)))
                #sent_out = torch.tanh(self.linear_reg1(sent_attn))
                #sent_out = torch.tanh(self.linear_reg1(torch.cat((sent_attn, diff, prod), 0)))
                final_out = self.final_linear(sent_out)
                final_scores.append(final_out)

        elif self.task == 3:
            input = input[0]
            output_per_seq1, _ = self.bert_model(input[0].long())
            output_per_seq2, _ = self.bert_model(input[1].long())
            prod = torch.mean(output_per_seq1 * output_per_seq2,1)
            diff = torch.mean(output_per_seq1 - output_per_seq2,1)
            out = torch.tanh(self.linear_reg1(torch.cat((prod,diff),1)))
            final_out = self.final_linear(out)
            final_scores.append(final_out)

        return torch.stack((final_scores))

    def train(self,mode=True):
        #if self.add_joke_model :
        #    self.load_joke_lm_weights("joke_classification_bert.pth")
        if torch.cuda.is_available():
            self.cuda()
        #self.bert_model = self.bert_model.bert
        self.bert_model = self.bert_model.bert
        #self.bert_model.requires_grad = False
        #optimizer = optim.Adam(list(self.linear_reg1.parameters())+list(self.final_linear.parameters())+list(self.lstm.parameters())+list(self.attention.parameters()), lr=self.lr,weight_decay=0.001)
        optimizer = optim.Adam(self.parameters(), lr=self.lr,
                               weight_decay=0.001)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[3,5],gamma=0.1)

        if self.task == 1:
            loss = nn.MSELoss()
            train_dataloader, val_dataloader, word2vec_ind = get_dataloaders_bert(self.train_file_path, self.gensim_model, "train",
                                                                    self.train_batch_size)

        else :
            loss = nn.CrossEntropyLoss()
            train_dataloader, val_dataloader = get_dataloaders_bert_task2(self.train_file_path, "train",
                                                                    self.train_batch_size)

        self.init_embeddings(word2vec_ind)
        best_loss  = sys.maxsize
        best_accuracy = -sys.maxsize

        for epoch in range(self.epochs):
            total_prev_loss = 0
            for (batch_num, batch) in enumerate(train_dataloader):
                # If gpu is available move to gpu.
                if torch.cuda.is_available():
                    input1 = batch[0].cuda()
                    input2 = batch[1].cuda()
                    locs = batch[2].cuda()
                    word2vec_locs = batch[3].cuda()
                    gt = batch[4].cuda()
                else:
                    input1 = batch[0]
                    input2 = batch[1]
                    locs = batch[2]
                    word2vec_locs = batch[3]
                    gt = batch[4]

                loss_val = 0
                self.bert_model.train()
                self.linear_reg1.train()
                self.final_linear.train()

                # Clear gradients
                optimizer.zero_grad()
                final_scores = self.forward((input1,input2,locs,word2vec_locs))
                if self.task==1:
                    loss_val += loss(final_scores.squeeze(1), gt.float())
                else :
                    loss_val += loss(final_scores.squeeze(0), gt.long())

                # Compute gradients
                loss_val.backward()
                total_prev_loss += loss_val.item()
                print("Loss for batch" + str(batch_num) + ": " + str(loss_val.item()))
                # Update weights according to the gradients computed.
                optimizer.step()

            # Don't compute gradients in validation step
            with torch.no_grad():
                # Ensure that dropout behavior is correct.
                predictions = []
                ground_truth = []
                self.bert_model.eval()
                self.linear_reg1.eval()
                self.final_linear.eval()
                mse_loss = 0
                for (val_batch_num, val_batch) in enumerate(val_dataloader):
                    if torch.cuda.is_available():
                        input1 = val_batch[0].cuda()
                        input2 = val_batch[1].cuda()
                        locs = val_batch[2].cuda()
                        word2vec_locs = val_batch[3].cuda()
                        gt = val_batch[4].cuda()
                    else:
                        input1 = val_batch[0]
                        input2 = val_batch[1]
                        locs = val_batch[2]
                        word2vec_locs = val_batch[3]
                        gt = val_batch[4]
                    final_scores = self.forward((input1,input2,locs,word2vec_locs))
                    if self.task == 1:
                        mse_loss += mean_squared_error(final_scores.cpu().detach().squeeze(1),gt.cpu().detach())

                    elif self.task == 2:
                        mse_loss += loss(final_scores.squeeze(0),gt.long())
                        predictions.extend(torch.argmax(final_scores.squeeze(0),1).tolist())
                        ground_truth.extend(gt.tolist())
                print("Validation Loss is " + str(mse_loss / (val_batch_num + 1)))

                if self.task == 1:
                    if mse_loss < best_loss:
                        torch.save(self.state_dict(), "model_" + str(self.task) + str(epoch) + ".pth")
                        best_loss = mse_loss
                elif self.task == 2:
                    accuracy = accuracy_score(ground_truth,predictions)
                    if accuracy > best_accuracy:
                        torch.save(self.state_dict(), "model_" + str(self.task) + str(epoch) + ".pth")
                        best_accuracy = accuracy
                    print ("Accuracy is " + str(accuracy_score(ground_truth,predictions)))
                scheduler.step()

    def predict(self,model_path=None):

        '''
        This function predicts the classes on a test set and outputs a csv file containing the id and predicted class
        :param model_path: Path of the model to be loaded if not the current model is used.
        :return:

        '''
        self.bert_model = self.bert_model.bert
        if torch.cuda.is_available():
            self.cuda()
        if model_path:
            self.load_state_dict(torch.load(model_path))
        if self.task == 1:
            test_dataloader = get_dataloaders_bert(self.test_file_path,"test")
        else:
            test_dataloader = get_dataloaders_bert_task2(self.test_file_path, "test")
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
                        #id = batch[3].cuda()
                    else:
                        input1 = batch[0]
                        input2 = batch[1]
                        locs = batch[2]
                    if self.task == 2:
                        final_scores_1 = self.forward((input1,input1,locs,torch.tensor(1)))
                        final_scores_2 = self.forward((input2,input2,locs,torch.tensor(2)))
                    #if self.task == 2:
                    #    final_scores = torch.argmax(final_scores.squeeze(0),1)
                    for cnt,pred in enumerate(final_scores_1):
                        if final_scores_1[cnt]>final_scores_2[cnt]:
                            #f.writelines(str(id[cnt].item())+","+str(pred.item())+"\n")
                            f.writelines(str(cnt) + "," + str(1) + "\n")
                        else:
                            f.writelines(str(cnt) + "," + str(2) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",action="store",type=int,default=4,required=False)
    parser.add_argument("--train_file_path",type=str,default="../data/task-1/train.csv",required=False)
    parser.add_argument("--dev_file_path", type=str, default="../data/task-1/dev.csv", required=False)
    parser.add_argument("--test_file_path", type=str, default="../data/task-1/dev.csv", required=False)
    parser.add_argument("--lm_file_path", type=str, default="../data/task-1/shortjokes1.csv", required=False)
    parser.add_argument("--lm_weights_file_path", type=str, default="../models/lm_joke_bert.pth", required=False)
    parser.add_argument("--model_file_path", type=str, default="../models/model_4.pth", required=False)
    parser.add_argument("--predict", type=str, default='re',required=False)
    parser.add_argument("--add_joke_train", type=str, default='true',required=False)
    parser.add_argument("--lm_pretrain", type=str, default='false',required=False)
    parser.add_argument("--joke_classification_path", type=str,default='../data/task-1/joke_classification.csv', required=False)
    parser.add_argument("--lr",type=float,default=0.0001,required=False)
    parser.add_argument("--train_scratch", type=str, default='false',required=False)
    parser.add_argument("--task", type=int, default=1, required=False)
    parser.add_argument("--epochs", type=int, default=5, required=False)
    args = parser.parse_args()

    obj = RBERT(args.train_file_path,args.dev_file_path,args.test_file_path,args.lm_file_path,args.batch_size,64,
                args.lr,args.lm_weights_file_path,args.epochs,args.lm_pretrain,args.task,args.train_scratch,args.model_file_path,
                args.joke_classification_path, args.add_joke_train)

    if args.lm_pretrain=='true':
        obj.pre_train_bert()

    if args.predict=='true':
        obj.predict(args.model_file_path)
    else:
        #obj.train_joke_classification()
        obj.train()
