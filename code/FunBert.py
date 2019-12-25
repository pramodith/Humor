import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import sys
from pytorch_transformers import BertForMaskedLM, DistilBertForMaskedLM, RobertaModel
from pytorch_pretrained_bert import BertAdam
import argparse
from data_handler import *
import torchnlp.nn as nn_nlp
import gensim
import numpy as np
import json

torch.manual_seed(2)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class RBERT(nn.Module):

    def __init__(self, train_file_path: str, dev_file_path: str, test_file_path: str, lm_file_path: str,
                 train_batch_size: int,
                 test_batch_size: int, lr: float, lm_weights_file_path: str, epochs: int, lm_pretrain: str, task: int,
                 train_scratch: str, model_path: str,
                 joke_classification_path: str, add_joke_model: str,word2vec : str):
        '''

        :param train_file_path: Path to the train file
        :param test_file_path: Path to the test file
        :param train_batch_size: Size of the batch during training
        :param test_batch_size: Size of the batch during testing
        :param lr: learning rate
        '''

        super(RBERT, self).__init__()
        self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states=True,
                                                       attention_probs_dropout_prob=0.1)
        #self.bert_model = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True,attention_probs_dropout_prob=0.1)
        if lm_pretrain != 'true':
            pass
            # self.load_joke_lm_weights(lm_weights_file_path)
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_file_path = train_file_path
        self.lm_file_path = lm_file_path
        self.nn_embeddings = torch.nn.Embedding(13, 300)
        self.word_embeddings = torch.nn.Embedding(7898,300)
        #self.multihead_attn = nn.MultiheadAttention(embed_dim=768+300,num_heads=1,dropout=0.1)
        #self.lstm = nn.LSTM(768*2,768*2,bidirectional=False)
        self.attention = nn_nlp.Attention(768*2)
        #self.entity_layer = nn.Sequential(nn.Dropout(0.3), nn.Linear(768 + 300, 768))
        self.word2vec = word2vec
        self.dev_file_path = dev_file_path
        self.test_file_path = test_file_path
        self.joke_classification_path = joke_classification_path
        self.lr = lr
        self.task = task

        if word2vec=='true':
            self.gensim_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',
                                                                                binary=True)
        else:
            self.gensim_model = None

        self.add_joke_model = add_joke_model
        self.epochs = epochs
        self.linear_joke = nn.Sequential(nn.Dropout(0.3), nn.Linear(768, 2))
        self.linear_reg1 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768 * 10, 1))

        if self.task:
            self.final_linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(768*8, 1))
        else:
            self.final_linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(100, 2))

        if train_scratch == 'true':
            self.bert_model = self.bert_model.bert
            print("loaded the model")
            self.load_state_dict(torch.load(model_path))

    def init_embeddings(self, ind: np.ndarray):
        ind = np.unique(ind.reshape(-1))
        self.nn_embeddings = torch.nn.Embedding(len(ind), 300).cuda()
        self.emb_key = {str(ind[i]): i for i in range(len(ind))}
        with open("word2vec_ind.json", 'w') as f:
            json.dump(self.emb_key, f)
        emb_values = np.take(self.gensim_model.vectors, ind, 0)
        self.nn_embeddings.load_state_dict({'weight': torch.tensor(emb_values)})
        del self.gensim_model

    def load_joke_lm_weights(self, lm_path: str):
        self.bert_model.load_state_dict(torch.load(lm_path))

    def freeze(self, epoch):
        num_params = 190
        for ind, param in enumerate(self.parameters()):
            print(param.requires_grad,param.shape)
            if epoch>4 or num_params - ind <= 10 * (epoch):

                param.requires_grad = True

            else:
                param.requires_grad = False

    def pre_train_bert(self):
        var1 = [{'params': self.bert_model.bert.encoder.layer[i].parameters(), 'lr': 2e-5 * (0.95 ** ((12 - i))),
                 'weight_decay': 0.001} for i in range(12)]
        #var2 = [{'params': x for x in list(set(self.bert_model.bert.parameters()).difference(self.bert_model.bert.encoder.parameters()))}]
        optimizer = optim.Adam(var1, lr=5e-5, weight_decay=1e-3)
        #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, total_steps=2000, epochs=1,
        #                                          steps_per_epoch=400, anneal_strategy='linear')

        train_dataloader = get_bert_lm_dataloader(self.lm_file_path,64)
        print("Training LM")
        if torch.cuda.is_available():
            self.bert_model.cuda()
        for epoch in range(1):
            print("Epoch : " + str(epoch))
            for ind, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    inp = batch[0].cuda()
                else:
                    inp = batch[0]
                pos = torch.randint(high=inp.shape[1],size=(int(inp.shape[1]*0.8),))
                inp[:,pos] = -1
                outputs = self.bert_model(inp, masked_lm_labels=inp.long())
                loss, prediction_scores = outputs[:2]
                loss.backward()
                print("Loss is :" + str(loss.item()))
                optimizer.step()
                #scheduler.step()
                #for param_group in optimizer.param_groups:
                # print(param_group['lr'])
                if ind > 2000:
                    break
        print("LM training done")
        torch.save(self.bert_model.state_dict(), "lm_joke_bert.pth")

    def train_joke_classification(self):
        self.bert_model = self.bert_model.bert
        optimizer = optim.Adam(list(self.bert_model.parameters()) + list(self.linear_joke.parameters()), lr=self.lr,
                               weight_decay=0.001)
        train_dataloader, val_dataloader = get_dataloaders_joke_classification(self.joke_classification_path)
        best_accuracy = 0
        loss = nn.CrossEntropyLoss()
        print("Training Joke Model")
        if torch.cuda.is_available():
            self.bert_model.cuda()
            self.linear_joke.cuda()
        for epoch in range(self.epochs):
            loss_val = 0
            print("Epoch : " + str(epoch))
            self.bert_model.train()
            self.linear_joke.train()
            for ind, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    inp = batch[0].cuda()
                    gt = batch[1].cuda()
                else:
                    inp = batch[0]
                    gt = batch[1]
                outputs, _, _ = self.bert_model(inp)
                outputs = self.linear_joke(outputs[:, 0, :])
                loss_val += loss(outputs.squeeze(0), gt.long())
                loss_val.backward()
                print("Loss is :" + str(loss_val.item()))
                optimizer.step()

            self.bert_model.bert.eval()
            self.linear_joke.eval()
            for ind, batch in enumerate(val_dataloader):
                predictions = []
                ground_truth = []
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    inp = batch[0].cuda()
                    gt = batch[1].cuda()
                else:
                    inp = batch[0]
                    gt = batch[1]
                outputs, _, _ = self.bert_model.bert(inp)
                outputs = self.linear_joke(outputs[:, 0, :])
                predictions.extend(torch.argmax(outputs.squeeze(0), 1).tolist())
                ground_truth.extend(gt.tolist())
                print("Loss is :" + str(loss_val.item()))
            accuracy = accuracy_score(ground_truth, predictions)
            print(f'''Accuracy is {accuracy}''')
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(self.bert_model.state_dict(), "joke_classification_bert.pth")

    def hook_encoder_bert(self, input, output):
        return output

    def forward(self, *input) :
        final_out = []
        input = input[0]
        out_per_seq, _,attention_layer_inps= self.bert_model(input[0].long(),input[1].long())
        out_per_seq = torch.cat((out_per_seq,attention_layer_inps[11]),2)
        pos = input[0].clone().detach().cpu()
        for (i, loc) in enumerate(input[2]):
            # +1 is to ensure that the symbol token is not considered
            entity1 = torch.mean(out_per_seq[i,loc[0]+1:loc[1]],0)
            entity2 = torch.mean(out_per_seq[i, loc[2] + 1:loc[3]], 0)
            entity_diff = torch.abs(entity2-entity1)
            imp_seq1 = torch.cat((out_per_seq[i, 0:loc[0] + 1], out_per_seq[i, loc[1]:]), 0)
            imp_seq2 = torch.cat((out_per_seq[i, np.where(pos[i].numpy()==102)[0][0]:loc[2] + 1], out_per_seq[i, loc[3]:]), 0)
            _, attention_score = self.attention(entity2.unsqueeze(0).unsqueeze(0), imp_seq2.unsqueeze(0))
            sent_attn2 = torch.sum(attention_score.squeeze(0).expand(768 * 2, -1).t() * imp_seq2, 0)
            _, attention_score = self.attention(entity1.unsqueeze(0).unsqueeze(0), imp_seq1.unsqueeze(0))
            sent_attn1 = torch.sum(attention_score.squeeze(0).expand(768 * 2, -1).t() * imp_seq1, 0)
            #attn_diff = torch.abs(sent_attn2-sent_attn1)
            out = self.final_linear(torch.cat((out_per_seq[i,0,:],sent_attn2,sent_attn1,entity_diff)))
            final_out.append(out)
        #out = self.final_linear(torch.cat((out_per_seq[:, 0, :],entity_diff), 1))

        return torch.stack(final_out)


    '''
    def forward(self, *input):
        
        #:param input: input[0] is the sentence, input[1] are the entity locations , input[2] is the ground truth
        #:return: Scores for each class
        
        final_scores = []

        if self.task == 1:
            input = input[0]
            output_per_seq1,_,attention_layer_inps = self.bert_model(input[0].long())
            #attention_layer_inps = torch.mean(attention_layer_inps,0)
            output_per_seq1 = torch.cat((output_per_seq1, attention_layer_inps[11]), 2)
            #output_per_seq1 = output_per_seq1.transpose(0, 1)
            #output_per_seq1, _ = self.lstm(output_per_seq1)
            #output_per_seq1 = output_per_seq1.transpose(0, 1)
            output_per_seq2, _, attention_layer_inps = self.bert_model(input[1].long())
            #attention_layer_inps = torch.mean(attention_layer_inps, 0)
            output_per_seq2 = torch.cat((output_per_seq2, attention_layer_inps[11]), 2)
            #output_per_seq2 = output_per_seq2.transpose(0,1)
            #output_per_seq2,_ = self.lstm(output_per_seq2)
            #output_per_seq2 = output_per_seq2.transpose(0,1)
            
            #Obtain the vectors that represent the entities and average them followed by a Tanh and a linear layer.
            for (i, loc) in enumerate(input[2]):
                # +1 is to ensure that the symbol token is not considered
                entity2 = torch.mean(output_per_seq2[i, loc[2] + 1:loc[3]], 0)
                #entity2_max = torch.max(output_per_seq2[i, loc[2] + 1:loc[3]], 0)
                #imp_seq1 = torch.cat((output_per_seq1[i, 0:loc[0] + 1], output_per_seq1[i, loc[1]:]), 0)
                imp_seq2 = torch.cat((output_per_seq2[i, 0:loc[2] + 1], output_per_seq2[i, loc[3]:]), 0)
                _, attention_score = self.attention(entity2.unsqueeze(0).unsqueeze(0), imp_seq2.unsqueeze(0))
                sent_attn = torch.sum(attention_score.squeeze(0).expand(768*2, -1).t() * imp_seq2, 0)
                #_, attention_score1 = self.attention(entity1.unsqueeze(0).unsqueeze(0), imp_seq1.unsqueeze(0))
                #sent_attn1 = torch.sum(attention_score1.squeeze(0).expand(768 * 2, -1).t() * imp_seq1, 0)
                #tag2vec_entity1 = self.nn_embeddings(loc[4])
                tag2vec_entity2 = self.nn_embeddings(loc[5])
                # diff = torch.sub(entity1,entity2)
                # prod = entity1*entity2
                #word2vec_diff = torch.abs(self.word_embeddings(loc[6]) - self.word_embeddings(loc[7]))
                #entity1 = torch.tanh(
                #    self.entity_layer(torch.cat((entity1, self.word_embeddings(loc[6]), tag2vec_entity1), 0)))
                #entity2 = torch.tanh(self.entity_layer(torch.cat((entity2,self.word_embeddings(loc[7])),0)))
                sent_out = torch.tanh(
                        self.linear_reg1(torch.cat((sent_attn,torch.abs(output_per_seq2[i, 0]-output_per_seq1[i, 0]),entity2,self.word_embeddings(loc[7])), 0)))
                final_out = self.final_linear(sent_out)
                final_scores.append(final_out)

        if self.task == 2:
            input = input[0]
            output_per_seq2, _, attention_layer_inps = self.bert_model(input[1].long())
            output_per_seq2 = torch.cat((output_per_seq2, attention_layer_inps[8]), 2)

            #Obtain the vectors that represent the entities and average them followed by a Tanh and a linear layer.
            for (i, loc) in enumerate(input[2]):
                # +1 is to ensure that the symbol token is not considered
                # entity1 = torch.mean(output_per_seq1[i, loc[0] + 1:loc[1]], 0)
                if input[3] == 1:
                    ent_ind = 0
                else:
                    ent_ind = 2

                entity2 = torch.mean(output_per_seq2[i, loc[ent_ind] + 1:loc[ent_ind + 1]], 0)
                entity2_max = torch.max(output_per_seq2[i, loc[ent_ind] + 1:loc[ent_ind + 1]], 0)
                imp_seq = torch.cat((output_per_seq2[i, 0:loc[ent_ind] + 1], output_per_seq2[i, loc[ent_ind + 1]:]), 0)
                _, attention_score = self.attention(entity2.unsqueeze(0).unsqueeze(0), imp_seq.unsqueeze(0))
                sent_attn = torch.sum(attention_score.squeeze(0).expand(768 * 2, -1).t() * imp_seq, 0)
                # diff = torch.sub(entity1,entity2)
                # prod = entity1*entity2

                sent_out = torch.tanh(
                    self.linear_reg1(torch.cat((sent_attn, output_per_seq2[i, 0], entity2, word2vec_diff), 0)))
                # sent_out = torch.tanh(self.linear_reg1(sent_attn))
                # sent_out = torch.tanh(self.linear_reg1(torch.cat((sent_attn, diff, prod), 0)))
                final_out = self.final_linear(sent_out)
                final_scores.append(final_out)

        elif self.task == 3:
            input = input[0]
            output_per_seq1, _ = self.bert_model(input[0].long())
            output_per_seq2, _ = self.bert_model(input[1].long())
            prod = torch.mean(output_per_seq1 * output_per_seq2, 1)
            diff = torch.mean(output_per_seq1 - output_per_seq2, 1)
            out = torch.tanh(self.linear_reg1(torch.cat((prod, diff), 1)))
            final_out = self.final_linear(out)
            final_scores.append(final_out)

        return torch.stack((final_scores))
    '''

    def multitask_train(self):

        if torch.cuda.is_available():
            self.cuda()
        best_loss = sys.maxsize
        best_accuracy = -sys.maxsize
        self.bert_model = self.bert_model.bert
        loss_1 = nn.MSELoss()
        loss_2 = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=4000, epochs=1,
                                                  steps_per_epoch=400, anneal_strategy='linear')
        train_dataloader_reg, val_dataloader_reg = get_dataloaders_bert(self.train_file_path,
                                                                              self.gensim_model, "train",
                                                                              self.train_batch_size)
        train_dataloader_cl, val_dataloader_cl = get_dataloaders_joke_classification(self.joke_classification_path)

        for epoch in range(self.epochs):
            for batch_num,batch in enumerate(zip(train_dataloader_reg,train_dataloader_cl)):
                # If gpu is available move to gpu.
                batch_reg = batch[0]
                batch_cl = batch[1]
                if torch.cuda.is_available():
                    input1 = batch_reg[0].cuda()
                    input2 = batch_reg[1].cuda()
                    locs = batch_reg[2].cuda()
                    gt = batch_reg[3].cuda()
                else:
                    input1 = batch_reg[0]
                    input2 = batch_reg[1]
                    locs = batch_reg[2]
                    gt = batch_reg[3]

                loss_val = 0
                self.bert_model.train()
                #self.attention.train()
                self.linear_reg1.train()
                self.final_linear.train()

                optimizer.zero_grad()
                final_scores = self.forward((input1, input2, locs))
                loss_val += loss_1(final_scores.squeeze(1), gt.float())

                self.linear_joke.train()
                if torch.cuda.is_available():
                    inp = batch_cl[0].cuda()
                    gt = batch_cl[1].cuda()
                else:
                    inp = batch_cl[0]
                    gt = batch_cl[1]

                outputs, _, _ = self.bert_model(inp)
                outputs = self.linear_joke(outputs[:, 0, :])
                loss_val += loss_2(outputs.squeeze(0), gt.long())
                loss_val.backward()
                print(f"Loss is {batch_num}:" + str(loss_val.item()))
                optimizer.step()
                scheduler.step()

            with torch.no_grad():
                # Ensure that dropout behavior is correct.
                predictions = []
                ground_truth = []
                self.bert_model.eval()
                #self.attention.eval()
                self.linear_reg1.eval()
                self.final_linear.eval()
                self.linear_joke.eval()
                mse_loss = 0
                for val_batch_num, val_batch in enumerate(zip(val_dataloader_reg,val_dataloader_cl)):

                    val_batch_reg = val_batch[0]
                    val_batch_cl = val_batch[1]

                    if torch.cuda.is_available():
                        input1 = val_batch_reg[0].cuda()
                        input2 = val_batch_reg[1].cuda()
                        locs = val_batch_reg[2].cuda()
                        #word2vec_locs = val_batch[3].cuda()
                        gt = val_batch_reg[3].cuda()
                    else:
                        input1 = val_batch_reg[0]
                        input2 = val_batch_reg[1]
                        locs = val_batch_reg[2]
                        #word2vec_locs = val_batch[3]
                        gt = val_batch_reg[3]

                    final_scores = self.forward((input1, input2, locs))
                    mse_loss += mean_squared_error(final_scores.cpu().detach().squeeze(1), gt.cpu().detach())

                    predictions = []
                    ground_truth = []
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        inp = batch_cl[0].cuda()
                        gt = batch_cl[1].cuda()
                    else:
                        inp = batch_cl[0]
                        gt = batch_cl[1]
                    outputs, _, _ = self.bert_model.bert(inp)
                    outputs = self.linear_joke(outputs[:, 0, :])
                    predictions.extend(torch.argmax(outputs.squeeze(0), 1).tolist())
                    ground_truth.extend(gt.tolist())

                print("Validation Loss is " + str(mse_loss / (val_batch_num + 1)))
                accuracy = accuracy_score(ground_truth, predictions)
                print(f'''Accuracy is {accuracy}''')
            torch.save(self.state_dict(), "model_" + str(self.task) + str(epoch) + ".pth")


    def train(self, mode=True):
        # if self.add_joke_model :
        #    self.load_joke_lm_weights("joke_classification_bert.pth")
        if torch.cuda.is_available():
            self.cuda()
        self.bert_model = self.bert_model.bert
        var1 = [{'params':self.bert_model.encoder.layer[i].parameters(), 'lr': 2e-5*(0.95**((12-i))),'weight_decay':0.001} for i in range(12)]
        var2 = [{'params': x for x in list(set(self.parameters()).difference(self.bert_model.encoder.parameters()))}]
        optimizer = optim.Adam(list(self.bert_model.parameters())+list(self.final_linear.parameters())+list(self.attention.parameters()),
                               lr = self.lr,weight_decay=1e-3)
        #optimizer = optim.Adam(var1+var2,lr=2e-5,weight_decay=1e-3)
        #optimizer = optim.Adam([filter(lambda x: not hasattr(x,'encoder'), self.parameters())]

        #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-5, total_steps=620, epochs=self.epochs,
        #                                          steps_per_epoch=60, anneal_strategy='linear')

        if self.task == 1:
            loss = nn.MSELoss()
            train_dataloader, val_dataloader = get_sent_emb_dataloaders_bert(self.train_file_path,
                                                                                  mode="train",
                                                                                  train_batch_size = self.train_batch_size)
            #self.word_embeddings.load_state_dict({'weight': vectors})


        else:
            loss = nn.CrossEntropyLoss()
            train_dataloader, val_dataloader = get_dataloaders_bert_task2(self.train_file_path, "train",
                                                                          self.train_batch_size)

        if self.word2vec=='true':
            self.init_embeddings(word2vec_ind)
        best_loss = sys.maxsize
        best_accuracy = -sys.maxsize

        for epoch in range(self.epochs):
            self.bert_model.train()
            #self.word_embeddings.train()
            #self.nn_embeddings.train()
            self.attention.train()
            #self.linear_reg1.train()
            self.final_linear.train()
            #if epoch == 5:
            #self.freeze(epoch)
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            total_prev_loss = 0
            for (batch_num, batch) in enumerate(train_dataloader):
                # If gpu is available move to gpu.
                if torch.cuda.is_available():
                    input1 = batch[0].cuda()
                    input2 = batch[1].cuda()
                    locs = batch[2].cuda()
                    #word2vec_locs = batch[3].cuda()
                    gt = batch[3].cuda()
                else:
                    input1 = batch[0]
                    input2 = batch[1]
                    locs = batch[2]
                    #word2vec_locs = batch[3]
                    gt = batch[3]

                loss_val = 0

                # Clear gradients
                optimizer.zero_grad()
                if self.word2vec=='true':
                    final_scores = self.forward((input1, input2, locs, word2vec_locs))
                else:
                    final_scores = self.forward((input1, input2, locs))
                if self.task == 1:
                    loss_val += loss(final_scores.squeeze(1), gt.float())
                else:
                    loss_val += loss(final_scores.squeeze(0), gt.long())

                # Compute gradients
                loss_val.backward()
                total_prev_loss += loss_val.item()
                print("Loss for batch" + str(batch_num) + ": " + str(loss_val.item()))
                # Update weights according to the gradients computed.
                optimizer.step()
                #scheduler.step()

            # Don't compute gradients in validation step
            with torch.no_grad():
                # Ensure that dropout behavior is correct.
                predictions = []
                ground_truth = []
                self.bert_model.eval()
                self.attention.eval()
                #self.word_embeddings.eval()
                #self.nn_embeddings.eval()
                #self.linear_reg1.eval()
                self.final_linear.eval()
                mse_loss = 0
                for (val_batch_num, val_batch) in enumerate(val_dataloader):
                    if torch.cuda.is_available():
                        input1 = val_batch[0].cuda()
                        input2 = val_batch[1].cuda()
                        locs = val_batch[2].cuda()
                        #word2vec_locs = val_batch[3].cuda()
                        gt = val_batch[3].cuda()
                    else:
                        input1 = val_batch[0]
                        input2 = val_batch[1]
                        locs = val_batch[2]
                        #word2vec_locs = val_batch[3]
                        gt = val_batch[3]

                    if self.word2vec=='true':
                        final_scores = self.forward((input1, input2, locs, word2vec_locs))
                    else:
                        final_scores = self.forward((input1, input2, locs))

                    if self.task == 1:
                        mse_loss += mean_squared_error(final_scores.cpu().detach().squeeze(1), gt.cpu().detach())

                    elif self.task == 2:
                        mse_loss += loss(final_scores.squeeze(0), gt.long())
                        predictions.extend(torch.argmax(final_scores.squeeze(0), 1).tolist())
                        ground_truth.extend(gt.tolist())

                print("Validation Loss is " + str(mse_loss / (val_batch_num + 1)))

                if self.task == 1:
                    if mse_loss < best_loss:
                        torch.save(self.state_dict(), "model_" + str(self.task) + str(epoch) + ".pth")
                        best_loss = mse_loss
                elif self.task == 2:
                    accuracy = accuracy_score(ground_truth, predictions)
                    if accuracy > best_accuracy:
                        torch.save(self.state_dict(), "model_" + str(self.task) + str(epoch) + ".pth")
                        best_accuracy = accuracy
                    print("Accuracy is " + str(accuracy_score(ground_truth, predictions)))
                #scheduler.step()

    def predict(self, model_path=None):

        '''
        This function predicts the classes on a test set and outputs a csv file containing the id and predicted class
        :param model_path: Path of the model to be loaded if not the current model is used.
        :return:
        '''

        self.bert_model = self.bert_model.bert
        if torch.cuda.is_available():
            self.cuda()
        if model_path:
            #pass
            self.load_state_dict(torch.load(model_path))
        if self.task == 1:
            test_dataloader = get_sent_emb_dataloaders_bert(self.test_file_path, mode="test")

        else:
            test_dataloader = get_dataloaders_bert_task2(self.test_file_path, "test")
        self.bert_model.eval()
        self.linear_reg1.eval()
        self.final_linear.eval()
        with torch.no_grad():
            with open("task-1-output.csv", "w+") as f:
                f.writelines("id,pred\n")
                for ind, batch in enumerate(test_dataloader):
                    if torch.cuda.is_available():
                        input1 = batch[0].cuda()
                        input2 = batch[1].cuda()
                        locs = batch[2].cuda()
                       # word2vec_indices = batch[3].cuda()
                        id = batch[3].cuda()
                    else:
                        input1 = batch[0]
                        input2 = batch[1]
                        locs = batch[2]
                    if self.task == 2:
                        final_scores_1 = self.forward((input1, input1, locs, torch.tensor(1)))
                        final_scores_2 = self.forward((input2, input2, locs, torch.tensor(2)))
                    else:
                        final_scores_1 = self.forward((input1, input2, locs))
                    # if self.task == 1:
                    #    final_scores = torch.argmax(final_scores.squeeze(0),1)
                    for cnt, pred in enumerate(final_scores_1):
                        # if final_scores_1[cnt]>final_scores_2[cnt]:
                        f.writelines(str(id[cnt].item()) + "," + str(pred.item()) + "\n")
                        #    f.writelines(str(cnt) + "," + str(1) + "\n")
                        # else:
                        #    f.writelines(str(cnt) + "," + str(2) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", action="store", type=int, default=32, required=False)
    parser.add_argument("--train_file_path", type=str, default="../data/task-1/train.csv", required=False)
    parser.add_argument("--dev_file_path", type=str, default="../data/task-1/dev.csv", required=False)
    parser.add_argument("--test_file_path", type=str, default="../data/task-1/dev.csv", required=False)
    parser.add_argument("--lm_file_path", type=str, default="../data/task-1/shortjokes1.csv", required=False)
    parser.add_argument("--lm_weights_file_path", type=str, default="../models/lm_joke_bert.pth", required=False)
    parser.add_argument("--model_file_path", type=str, default="../models/model_4.pth", required=False)
    parser.add_argument("--predict", type=str, default='false', required=False)
    parser.add_argument("--add_joke_train", type=str, default='true', required=False)
    parser.add_argument("--lm_pretrain", type=str, default='false', required=False)
    parser.add_argument("--word2vec", type=str, default='false', required=False)
    parser.add_argument("--joke_classification_path", type=str, default='../data/task-1/joke_classification.csv',
                        required=False)
    parser.add_argument("--lr", type=float, default=2e-5, required=False)
    parser.add_argument("--train_scratch", type=str, default='false', required=False)
    parser.add_argument("--task", type=int, default=1, required=False)
    parser.add_argument("--epochs", type=int, default=5, required=False)
    args = parser.parse_args()

    obj = RBERT(args.train_file_path, args.dev_file_path, args.test_file_path, args.lm_file_path, args.batch_size, 64,
                args.lr, args.lm_weights_file_path, args.epochs, args.lm_pretrain, args.task, args.train_scratch,
                args.model_file_path,
                args.joke_classification_path, args.add_joke_train,args.word2vec)

    if args.lm_pretrain == 'true':
        obj.pre_train_bert()

    if args.predict == 'true':
        obj.predict(args.model_file_path)
    else:
        # obj.train_joke_classification()
        obj.train()
        #obj.multitask_train()
