import torch
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import torch.nn as nn
import io
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import accuracy_score,f1_score,mean_squared_error
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json
import argparse


def tokenize(X: list):
    '''
    This function tokenizes the input sentences and returns a vectorized representation of them and the location
    of each entity in the sentence.

    :param X: List of all input sentences
    :return: A vectorized list representation of the sentence and a numpy array containing the locations of each entity. First two
    values in  a row belong to entity1 and the next two values belong to entity2.
    '''

    # Add the SOS and EOS tokens.
    # TODO: Replace fullstops with [SEP]
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in X]

    # Load the tokenizer
    tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'tokenizer', 'bert-base-uncased',
                               do_basic_tokenize=True)

    # Tokenize and vectorize
    tokenized_text = [tokenizer.tokenize(sentence) for sentence in sentences]
    X = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_text]

    # MAX_SEQ_LEN
    MAX_LEN = max([len(x) for x in X])

    #Pad sequences to make them all eqally long
    X = pad_sequences(X, MAX_LEN, 'long', 'post', 'post')

    return X

def get_dataloaders(file_path : str ,mode="train",train_batch_size=64,test_batch_size = 64):
    '''

    This function creates pytorch dataloaders for fast and easy iteration over the dataset.

    :param file_path: Path of the file containing train/test data
    :param mode: Test mode or Train mode
    :param train_batch_size: Size of the batch during training
    :param test_batch_size: Size of the batch during testing
    :return: Dataloaders
    '''

    # Read the data,tokenize and vectorize
    df = pd.read_csv(file_path, sep=",")
    id = df['id']
    X = df['original'].values
    X = [sent.replace("\"","") for sent in X]
    replaced = df['original'].apply(lambda x: x[x.index("<"):x.index(">")+1])
    y = df['meanGrade'].values
    edit = df['edit']
    X = [sent.replace(replaced[i],edit[i]) for i,sent in enumerate(X)]
    X = tokenize(X)

    if mode == "train":
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(X, y,
                                                                                            random_state=2019,
                                                                                            test_size=0.2)

        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)
        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)

        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
        # with an iterator the entire dataset does not need to be loaded into memory

        train_data = TensorDataset(train_inputs, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

        validation_data = TensorDataset(validation_inputs, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=test_batch_size)
        return train_dataloader, validation_dataloader

    if mode == "test":
        test_data = TensorDataset(torch.tensor(X), torch.tensor(y))
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)

        return test_data_loader


class RBERT(nn.Module):

    def __init__(self,train_file_path : str, dev_file_path : str, test_file_path : str, train_batch_size : int,test_batch_size : int,lr : float):
        '''

        :param train_file_path: Path to the train file
        :param test_file_path: Path to the test file
        :param train_batch_size: Size of the batch during training
        :param test_batch_size: Size of the batch during testing
        :param lr: learning rate
        '''

        super(RBERT, self).__init__()
        self.bert_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'model', 'bert-base-uncased')
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_file_path = train_file_path
        self.dev_file_path = dev_file_path
        self.test_file_path = test_file_path
        self.lr = lr
        self.linear_reg1 = nn.Sequential(
                  nn.Dropout(0.1),
                  nn.Linear(768,100),
                  )
        self.final_linear = nn.Sequential(nn.Dropout(0.1),nn.Linear(100,1))

    def forward(self, *input):
        '''

        :param input: input[0] is the sentence, input[1] are the entity locations , input[2] is the ground truth
        :return: Scores for each class

        '''

        input = input[0]
        output_per_seq, _ = self.bert_model(input.long())
        output_per_seq = torch.mean(output_per_seq,1)
        final_scores = []
        '''
        Obtain the vectors that represent the entities and average them followed by a Tanh and a linear layer.
        '''
        sent_emb = torch.tanh(output_per_seq)
        sent_out = self.linear_reg1(sent_emb)
        final_out = self.final_linear(sent_out)
        return final_out

    def train(self,mode=True):
        if torch.cuda.is_available():
            self.cuda()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        loss = nn.MSELoss()
        train_dataloader,val_dataloader = get_dataloaders(self.train_file_path,"train",self.train_batch_size)
        for epoch in range(5):

            for (batch_num, batch) in enumerate(train_dataloader):

                # If gpu is available move to gpu.
                if torch.cuda.is_available():
                    input = batch[0].cuda()
                    gt = batch[1].cuda()
                else:
                    input = batch[0]
                    gt = batch[1]
                loss_val = 0
                self.linear_reg1.train()
                self.final_linear.train()
                # Clear gradients
                optimizer.zero_grad()
                final_scores = self.forward((input))
                loss_val += loss(final_scores.squeeze(1), gt.float())
                # Compute gradients
                loss_val.backward()
                print("Loss for batch" + str(batch_num) + ": " + str(loss_val.item()))
                # Update weights according to the gradients computed.
                optimizer.step()
            
            torch.save(self.state_dict(), "model_" + str(epoch) + ".pth")

            # Don't compute gradients in validation step
            with torch.no_grad():
                # Ensure that dropout behavior is correct.
                self.bert_model.eval()
                self.linear_reg1.eval()
                self.final_linear.eval()
                mse_loss = 0
                for (val_batch_num, val_batch) in enumerate(val_dataloader):
                    if torch.cuda.is_available():
                        input = val_batch[0].cuda()
                        gt = val_batch[1].cuda()
                    else:
                        input = val_batch[0]
                        gt = val_batch[1]
                    final_scores = self.forward((input))
                    mse_loss+=mean_squared_error(final_scores.cpu().detach().squeeze(1),gt.cpu().detach())
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
        test_dataloader = get_dataloaders(self.test_file_path,"test")
        self.bert_model.eval()
        self.linear_entities.eval()
        self.linear_cls.eval()
        self.final_linear.eval()
        with open("label2ind.json",'r') as f:
            label2ind = json.load(f)
        ind2label = {v:k for (k,v) in label2ind.items()}
        acc = 0
        f1 = 0
        with torch.no_grad():
            with open("predictions.csv","w+") as f, open("gt.csv","w+") as g:

                for ind,batch in enumerate(test_dataloader):
                    if torch.cuda.is_available():
                        input = batch[0].cuda()
                        locs = batch[1].cuda()
                        gt = batch[2].cuda()
                    else:
                        input = batch[0]
                        locs = batch[1]
                        gt = batch[2]
                    final_scores = self.forward((input,locs))
                    predictions = torch.argmax(final_scores,1).tolist()
                   # print(sentences[ind])
                    #print(ind2label[predictions[0]],ind2label[gt[0].item()])
                    acc += accuracy_score(gt.tolist(), predictions)
                    f1 += f1_score(gt.tolist(), predictions, average='weighted')
                    for cnt,pred in enumerate(zip(predictions,gt.tolist())):
                        f.writelines(str(8000+ind*64+cnt)+"\t"+str(ind2label[pred[0]])+"\n")
                        g.writelines(str(8000+ind*64+cnt)+"\t"+str(ind2label[pred[1]])+"\n")


        print("Accuracy is " + str(acc/(ind+1)))
        print("F1-score is " + str(f1/(ind+1)))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",action="store",type=int,default=4,required=False)
    parser.add_argument("--train_file_path",type=str,default="../data/task-1/train.csv",required=False)
    parser.add_argument("--dev_file_path", type=str, default="../data/task-1/dev.csv", required=False)
    parser.add_argument("--test_file_path", type=str, default="../data/test.csv", required=False)
    parser.add_argument("--model_file_path", type=str, default="model_4.pth", required=False)
    parser.add_argument("--lr",type=float,default=0.0001,required=False)
    args = parser.parse_args()
    obj = RBERT(args.train_file_path,args.dev_file_path,args.test_file_path,args.batch_size,64,args.lr)
    obj.train()
    #obj.predict(args.model_file_path)