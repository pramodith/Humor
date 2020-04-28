import torch
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json
import re
from utils import get_glove_embeddings
import gensim
from transformers import BertTokenizer,DistilBertTokenizer,RobertaTokenizer

def convert_task2_to_task1():
    df2 = pd.read_csv("../data/task-2/train.csv")
    df1 = pd.read_csv("../data/task-1/train.csv")
    df2_1 = df2.drop(columns=['edit2','grades2','original2','meanGrade2','label'])
    df2_1.rename({"edit1":"edit","grades1":"grades","original1":"original","meanGrade1":"meanGrade"},axis='columns',inplace=True)
    df2_1['id'] = df2_1['id'].apply(lambda x : x.split("-")[0])
    df2_2 = df2.drop(columns=['edit1','grades1','original1','meanGrade1','label'])
    df2_2.rename({"edit2": "edit", "grades2": "grades", "original2": "original", "meanGrade2": "meanGrade"},axis='columns',inplace=True)
    df2_2['id'] = df2_2['id'].apply(lambda x : x.split("-")[1])
    combined = pd.concat([df1,df2_1,df2_2])
    combined.drop_duplicates(subset=['id'],keep='last',inplace=True)
    combined.to_csv("../data/task-1/combined.csv",index=False)

def create_vocab_dict(X : list):
    sentences = [X[i].split(" ") for i in range(len(X))]
    words = [word for sent in sentences for word in sent]
    words = set(words)
    words.add("<UNK>")
    words.add("<SOS>")
    words.add("<EOS>")
    word2ind = {}
    for ind, word in enumerate(words):
        word2ind[word] = ind + 1
    with open("word2ind.json","w+") as f:
        json.dump(word2ind, f, indent=5)


def tokenize(X :list):
    max_seq_len = max([len(X[i]) for i in range(len(X))])
    sentences = [X[i].split(" ") for i in range(len(X))]
    with open("word2ind.json", "r") as f:
        word2ind = json.load(f)
    X = [[word2ind[sent[ind]] if ind<len(sent) else word2ind["<EOS>"] for ind in range(max_seq_len) ] for sent in sentences]
    for sent in X:
        sent.insert(0,word2ind['<SOS>'])
    return X

def get_bert_lm_dataloader(file_path : str,batch_size = 16):
    jokes_df = pd.read_csv(file_path)
    jokes = jokes_df['Joke']
    jokes = "<s> " + jokes + " </s>"
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    X = [tokenizer.encode(sent,add_special_tokens=False) for sent in jokes]
    MAX_LEN = max([len(sent) for sent in X])
    X = pad_sequences(X, MAX_LEN, 'long', 'post', 'post',tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
    dataset = TensorDataset(torch.tensor(X))
    sampler = RandomSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, pin_memory=True)
    return data_loader

def get_glove_bert_dataloaders(file_path : str, mode='train',model=None,train_batch_size=64,test_batch_size = 64):
    df = pd.read_csv(file_path, sep=",")
    if mode == 'train':
        df1 = pd.read_csv(file_path[:-4] + "_funlines.csv", sep=",")
        df = pd.concat([df, df1], ignore_index=True)

    id = df['id']
    X = df['original'].values
    X = [sent.replace("\"", "") for sent in X]
    replaced = df['original'].apply(lambda x: x[x.index("<"):x.index(">") + 1])
    replaced_clean = [x.replace("<", "").replace("/>", "") for x in replaced]
    edit = df['edit']
    if mode!='test':
        y = df['meanGrade'].values
    X1 = [sent.replace(replaced[i], "< " + replaced[i].strip("<|/>") + " " + "<") for i, sent in enumerate(X)]
    X2 = [sent.replace(replaced[i], "<" + edit[i] + "/>") for i, sent in enumerate(X)]
    X2 = [sent.replace("<" + edit[i] + "/>", "^ " + edit[i] + " ^") for i, sent in enumerate(X2)]
    glove_tokens_X1, glove_vectors1,locs_x1 = get_glove_embeddings(X1)
    glove_tokens, glove_vectors, locs_x2 = get_glove_embeddings(X2)
    glove_vectors.update(glove_vectors1)

    if mode == 'train':
        word2ind = {word.lower(): i for i, word in enumerate(sorted(glove_vectors.keys()))}
        vectors = np.asarray([glove_vectors[key] for key in sorted(glove_vectors.keys())])
        with open('word2ind.json','w') as f:
            json.dump(word2ind,f)
    else:
        with open('word2ind.json','r') as f:
            word2ind =  json.load(f)
            #vectors = np.asarray([glove_vectors[key] for key in sorted(glove_vectors.keys())])
    X2_glove = np.asarray([np.array([word2ind[word] if word in word2ind else word2ind['<other>'] for word in glove_tokens[i]]) for i in range(len(glove_tokens))])
    X1_glove = np.asarray([np.array([word2ind[word] if word in word2ind else word2ind['<other>'] for word in glove_tokens[i]]) for i in range(len(glove_tokens_X1))])
    MAX_LEN_1 = max([len(x) for x in X1_glove]) + 1
    MAX_LEN_2 = max([len(x) for x in X2_glove]) + 1
    MAX_LEN = max(MAX_LEN_1, MAX_LEN_2)
    X2_glove = pad_sequences(X2_glove, MAX_LEN, 'long', 'post', 'post')
    X1_glove = pad_sequences(X1_glove,MAX_LEN, 'long', 'post', 'post')
    replacement_locs = np.concatenate((locs_x1,locs_x2),1)
    #X, entity_locs = tokenize_bert(X2,False)

    if mode == "train":

        train1_inputs = torch.tensor(X1_glove)
        train2_inputs = torch.tensor(X2_glove)
        train_labels = torch.tensor(y)
        train_entity_locs = torch.tensor(replacement_locs)
        #if model:
        #    train_word2vec_locs = torch.tensor(word2vec_indices)

        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
        # with an iterator the entire dataset does not need to be loaded into memory

        # train_data = TensorDataset(train1_inputs,train2_inputs, train_entity_locs, train_word2vec_locs, train_labels)
        train_data = TensorDataset(train1_inputs, train2_inputs, train_entity_locs,train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

        #if model:
        #    return train_dataloader, word2vec_indices
        #else:
        #    return train_dataloader
        return train_dataloader,vectors

    if mode == "val":
        test1_input = torch.tensor(X1_glove)
        test2_input = torch.tensor(X2_glove)

        val_entity_locs = torch.tensor(replacement_locs)
        # word2vec_locs = torch.tensor(word2vec_indices)
        y = torch.tensor(y)
        id = torch.tensor(id)
        test_data = TensorDataset(test1_input, test2_input, val_entity_locs,y, id)
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)
        return test_data_loader

    if mode == "test":
        test1_input = torch.tensor(X1)
        test2_input = torch.tensor(X2)

        train_entity_locs = torch.tensor(replacement_locs)
        # word2vec_locs = torch.tensor(word2vec_indices)
        id = torch.tensor(id)
        test_data = TensorDataset(test1_input, test2_input, train_entity_locs,id)
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)

        return test_data_loader

def get_dataloaders_joke_classification(file_path : str,train_batch_size=64,test_batch_size = 64,mode = 'train'):
    df = pd.read_csv(file_path,sep=',')
    X = df['Joke'].values
    X,_ = tokenize_bert(X,False)
    y = df['label'].values

    if mode == "train":
        train1_inputs, validation1_inputs, train_labels, validation_labels = train_test_split(X, y,
                                                                                              random_state=2019,
                                                                                              test_size=0.2)
        train1_inputs = torch.tensor(train1_inputs)
        train_labels = torch.tensor(train_labels)
        validation1_inputs = torch.tensor(validation1_inputs)
        validation_labels = torch.tensor(validation_labels)

        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
        # with an iterator the entire dataset does not need to be loaded into memory

        train_data = TensorDataset(train1_inputs,train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

        validation_data = TensorDataset(validation1_inputs, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=test_batch_size)
        return train_dataloader, validation_dataloader

    if mode == "test":
        test_data = TensorDataset(torch.tensor(X), torch.tensor(id))
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)
        return test_data_loader


def get_sent_emb_dataloaders_bert(file_path: str, mode='train', train_batch_size=64, test_batch_size=64, model=None):
    df = pd.read_csv(file_path, sep=",")
    if mode == 'train':
        df1 = pd.read_csv(file_path[:-4] + "_funlines.csv", sep=",")
        df = pd.concat([df, df1], ignore_index=True)
    id = df['id']
    X = df['original'].values
    X = [sent.replace("\"", "") for sent in X]
    for i in range(len(df)):
        if "<" not in df.loc[i, 'original']:
            print("here")
    replaced = df['original'].apply(lambda x: x[x.index("<"):x.index(">") + 1])
    replaced_clean = [x.replace("<", "").replace("/>", "") for x in replaced]
    if mode != 'test':
        y = df['meanGrade'].values
    edit = df['edit']
    X2 = [sent.replace(replaced[i], "^ " + edit[i] + " ^") for i, sent in enumerate(X)]
    X1 = [sent.replace("<", "< ").replace("/>", " <") for i, sent in enumerate(X)]
    X, _, entity_locs = tokenize_bert_sent(X1, X2)

    if mode == "train":

        train1_inputs = torch.tensor(X)
        train_labels = torch.tensor(y)
        train_entity_locs = torch.tensor(entity_locs)

        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
        # with an iterator the entire dataset does not need to be loaded into memory

        # train_data = TensorDataset(train1_inputs,train2_inputs, train_entity_locs, train_word2vec_locs, train_labels)
        train_data = TensorDataset(train1_inputs, train_entity_locs, train_labels)
        train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)

        # validation_data = TensorDataset(validation1_inputs,validation2_inputs, validation_entity_locs, validation_word2vec_locs, validation_labels)
        return train_dataloader

    if mode == "val":
        test1_input = torch.tensor(X)
        y = torch.tensor(y)
        train_entity_locs = torch.tensor(entity_locs)
        # word2vec_locs = torch.tensor(word2vec_indices)
        id = torch.tensor(id)
        test_data = TensorDataset(test1_input, train_entity_locs,y, id)
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)

        return test_data_loader

    if mode == "test":
        test1_input = torch.tensor(X)
        test2_input = torch.tensor(sent_emb)

        train_entity_locs = torch.tensor(entity_locs)
        # word2vec_locs = torch.tensor(word2vec_indices)
        id = torch.tensor(id)
        test_data = TensorDataset(test1_input, test2_input, train_entity_locs, id)
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)

        return test_data_loader

def get_dataloaders(file_path : str ,mode="train",train_batch_size=64,test_batch_size = 64):
    df = pd.read_csv(file_path, sep=",")
    id = df['id']
    X = df['original'].apply(lambda sent: sent.replace("\"","").replace("<","").replace("/>","").lower())
    replaced = df['original'].apply(lambda x: x[x.index("<")+1:x.index(">")-1].lower())
    if mode=='train':
        df1 = pd.read_csv(file_path[:-4]+"_funlines.csv",sep=",")
        df = pd.concat([df,df1],ignore_index=True)
    edit = df['edit']
    words = X + " " + df['edit']
    create_vocab_dict(words)
    X2 = [sent.replace(" "+replaced[i]+" "," "+edit[i]+" ") for i, sent in enumerate(X)]
    X1 = [sent.replace("<","").replace("/>", "") for i, sent in enumerate(X)]
    locs =[]

    for i in range(len(X1)):
        if replaced[i] in X[i].split(" "):
            locs.append(X1[i].split(" ").index(replaced[i]))
        else:
            locs.append(0)
    X2 = tokenize(X2)
    X1 = tokenize(X1)

    if mode == "train":

        train1_inputs = torch.tensor(X1)
        validation1_inputs = torch.tensor(validation1_inputs)
        train2_inputs = torch.tensor(X2)
        validation2_inputs = torch.tensor(validation2_inputs)
        train_labels = torch.tensor(y)
        validation_labels = torch.tensor(validation_labels)
        train_entity_locs = torch.tensor(locs)
        validation_entity_locs = torch.tensor(validation_entity_locs)

        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
        # with an iterator the entire dataset does not need to be loaded into memory

        train_data = TensorDataset(train1_inputs, train2_inputs, train_entity_locs, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

        validation_data = TensorDataset(validation1_inputs, validation2_inputs, validation_entity_locs,
                                        validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=test_batch_size)
        return train_dataloader, validation_dataloader

    if mode == "test":
        test_data = TensorDataset(torch.tensor(X), torch.tensor(id))
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)

        return test_data_loader

def tokenize_bert_sent(X1: list, X2 : list ):
    sentences = ["[CLS] " + X1[i] + " [SEP] " + X2[i] + " [SEP]" for i in range(len(X1))]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_text = [tokenizer.tokenize(sentence) for sentence in sentences]
    X = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_text]
    sent_emb = [[0 if i<sentence.index(102) else 1 for i  in range(len(sentence))] for sentence in X]
    MAX_LEN = max([len(x) for x in sent_emb])+1
    # Pad sequences to make them all equally long
    X = pad_sequences(X, MAX_LEN, 'long', 'post', 'post',0)
    sent_emb = pad_sequences(sent_emb,MAX_LEN,'long','post','post',1)
    # Find the locations of each entity and store them
    entity_locs1 = np.asarray(
            [[i for i, s in enumerate(sent) if '<' in s and len(s) == 1] for sent in tokenized_text])
    entity_locs2 = np.asarray([[i for i, s in enumerate(sent) if '^' in s] for sent in tokenized_text])

    return X, sent_emb,np.concatenate((entity_locs1, entity_locs2), 1)

def tokenize_bert(X: list, org : bool):
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
    #tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    #tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize and vectorize
    tokenized_text = [tokenizer.tokenize(sentence,add_special_tokens=False) for sentence in sentences]
    tokenized_text2 = [tokenizer.tokenize(sentence,add_special_tokens=True) for sentence in sentences]
    print(all([tokenized_text[i]==tokenized_text2[i] for i in range(len(tokenized_text))]))
    X = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_text]

    # MAX_SEQ_LEN
    #MAX_LEN = max([len(x) for x in X])+1
    MAX_LEN = 50
    #Pad sequences to make them all eqally long
    X = pad_sequences(X, MAX_LEN, 'long', 'post', 'post',value=tokenizer.pad_token)

    '''
    if org:
        entity_locs = np.asarray([[i for i, s in enumerate(sent) if '<' in s and len(s)<=2] for sent in tokenized_text])
    else:
        entity_locs = np.asarray([[i for i, s in enumerate(sent) if '^' in s] for sent in tokenized_text])
    '''

    # Find the locations of each entity and store them
    if org:
        entity_locs = np.asarray([[i for i, s in enumerate(sent) if '<' in s and len(s)==2] for sent in tokenized_text])
    else:
        entity_locs = np.asarray([[i for i, s in enumerate(sent) if '^' in s and len(s)==2] for sent in tokenized_text])

    return X,entity_locs


def get_dataloaders_bert(file_path : str, model ,mode="train",train_batch_size=64,test_batch_size = 64):

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
    if mode=='train':
        df1 = pd.read_csv(file_path[:-4]+"_funlines.csv",sep=",")
        df = pd.concat([df,df1],ignore_index=True)
    id = df['id']
    X = df['original'].values
    X = [sent.replace("\"","") for sent in X]
    for i in range(len(df)):
        if "<" not in df.loc[i,'original']:
            print("here")
    replaced = df['original'].apply(lambda x: x[x.index("<"):x.index(">")+1])
    replaced_clean = [x.replace("<","").replace("/>","") for x in replaced]
    if mode!='test':
        y = df['meanGrade'].values
    edit = df['edit']
    X2 = [sent.replace(replaced[i], "^ " + edit[i] + " ^") for i, sent in enumerate(X)]
    X1 = [sent.replace("<","< ").replace("/>"," <") for i,sent in enumerate(X)]
    X1,e1_locs = tokenize_bert(X1,True)
    X2,e2_locs = tokenize_bert(X2,False)

    replacement_locs = np.concatenate((e1_locs, e2_locs), 1)
    if model:
        word2vec_replaced = np.asarray([model.vocab[replaced_clean[i]].index if replaced_clean[i] in model else -1 for i in range(len(replaced))]).reshape(-1,1)
        word2vec_edited = np.asarray([model.vocab[edit[i]].index if edit[i] in model else -1 for i in range(len(edit))]).reshape(-1,1)
        word2vec_indices = np.concatenate((word2vec_replaced,word2vec_edited),1)

    if mode == "train":


        train1_inputs = torch.tensor(X1)
        train2_inputs = torch.tensor(X2)
        train_labels = torch.tensor(y)
        train_entity_locs = torch.tensor(replacement_locs)
        if model:
            train_word2vec_locs = torch.tensor(word2vec_indices)

        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
        # with an iterator the entire dataset does not need to be loaded into memory

        #train_data = TensorDataset(train1_inputs,train2_inputs, train_entity_locs, train_word2vec_locs, train_labels)
        train_data = TensorDataset(train1_inputs, train2_inputs, train_entity_locs, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

        if model:
            return train_dataloader, word2vec_indices
        else:
            return  train_dataloader

    if mode == "val":
        test1_input = torch.tensor(X1)
        test2_input = torch.tensor(X2)

        train_entity_locs = torch.tensor(replacement_locs)
        #word2vec_locs = torch.tensor(word2vec_indices)
        y = torch.tensor(y)
        id = torch.tensor(id)
        test_data = TensorDataset(test1_input, test2_input, train_entity_locs,y,id)
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)
        return test_data_loader

    if mode == "test":
        test1_input = torch.tensor(X1)
        test2_input = torch.tensor(X2)

        train_entity_locs = torch.tensor(replacement_locs)
        #word2vec_locs = torch.tensor(word2vec_indices)
        id = torch.tensor(id)
        test_data = TensorDataset(test1_input, test2_input, train_entity_locs,id)
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)

        return test_data_loader

def tokenize_roberta_sent(X1: list, X2 : list ):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    sentences = ["<s> " + X1[i] + " </s></s> " + X2[i] + " </s>" for i in range(len(X1))]
    tokenized_text = [tokenizer.tokenize(sentence) for sentence in sentences]
    X = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_text]
    #sent_emb = [[0 if i<sentence.index(102) else 1 for i  in range(len(sentence)) ] for sentence in X]
    MAX_LEN = max([len(x) for x in X])+1
    # Pad sequences to make them all eqally long
    X = pad_sequences(X, MAX_LEN, 'long', 'post', 'post',1)
    #sent_emb = pad_sequences(sent_emb,MAX_LEN,'long','post','post',1)
    # Find the locations of each entity and store them
    entity_locs1 = np.asarray(
            [[i for i, s in enumerate(sent) if '<' in s and len(s) == 2] for sent in tokenized_text])
    entity_locs2 = np.asarray([[i for i, s in enumerate(sent) if '^' in s and len(s) == 2] for sent in tokenized_text])

    return X,np.concatenate((entity_locs1, entity_locs2), 1)

def get_dataloaders_bert_task2(file_path : str ,mode="train",train_batch_size=64,test_batch_size = 64):
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

    def get_modified_sentence(orginal_key : str, edit_key : str):
        X = df[orginal_key].values
        X = [sent.replace("\"","") for sent in X]
        replaced = df[orginal_key].apply(lambda x: x[x.index("<"):x.index(">")+1])
        edit = df[edit_key]
        X = [sent.replace(replaced[i], "^ " + edit[i] + " ^") for i, sent in enumerate(X)]
        return X

    def get_original_sentence(original_key : str):
        X = df[original_key].values
        X = [sent.replace("<","< ").replace("/>"," <") for sent in X]
        return X

    X_org = get_original_sentence('original1')
    X1 = get_modified_sentence("original1","edit1")
    X2 = get_modified_sentence("original2","edit2")
    X_org,org_locs = tokenize_bert(X_org,True)
    X1,e1_locs = tokenize_bert(X1,False)
    X2,e2_locs = tokenize_bert(X2,False)
    if mode != 'test':
        y = df['meanGrade1'] > df['meanGrade2']
        y = [0 if label else 1 for label in y]

    replacement_locs = np.concatenate((org_locs,e1_locs, e2_locs), 1)

    if mode == "train":
        train_inputs = torch.tensor(X_org)
        train1_inputs = torch.tensor(X1)
        train2_inputs = torch.tensor(X2)

        train_labels = torch.tensor(y)
        train_entity_locs = torch.tensor(replacement_locs)

        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
        # with an iterator the entire dataset does not need to be loaded into memory

        train_data = TensorDataset(train_inputs,train1_inputs,train2_inputs, train_entity_locs, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

        return train_dataloader

    if mode == "val":
        test_input = torch.tensor(X_org)
        test1_input = torch.tensor(X1)
        test2_input = torch.tensor(X2)
        train_labels = torch.tensor(y)
        train_entity_locs = torch.tensor(replacement_locs)
        test_data = TensorDataset(test_input, test1_input, test2_input, train_entity_locs, train_labels)
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)

        return test_data_loader
    if mode == "test":
        test_org = torch.tensor(X_org)
        test1_input = torch.tensor(X1)
        test2_input = torch.tensor(X2)
        train_entity_locs = torch.tensor(replacement_locs)
        test_data = TensorDataset(test_org,test1_input, test2_input, train_entity_locs)
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)

        return test_data_loader


if __name__ == "__main__":
    convert_task2_to_task1()
    #get_dataloaders("../data/task-1/train.csv")