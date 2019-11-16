import torch
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json
import gensim
from pytorch_transformers import BertTokenizer,DistilBertTokenizer
from utils import pos_tag

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
    jokes = "[CLS] " + jokes + " [SEP]"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    X = [tokenizer.encode(sent) for sent in jokes]
    MAX_LEN = max([len(sent) for sent in jokes])
    X = pad_sequences(X, MAX_LEN, 'long', 'post', 'post')
    X = torch.tensor(X)[:,:150]
    dataset = TensorDataset(torch.tensor(X))
    sampler = RandomSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, pin_memory=True)
    return data_loader

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



def get_dataloaders(file_path : str ,mode="train",train_batch_size=64,test_batch_size = 64):
    df = pd.read_csv(file_path, sep=",")
    id = df['id']
    X = df['original'].apply(lambda sent: sent.replace("\"","").replace("<","").replace("/>","").lower())
    replaced = df['original'].apply(lambda x: x[x.index("<")+1:x.index(">")-1].lower())
    if mode == 'train':
        y = df['meanGrade'].values
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
        train1_inputs, validation1_inputs, train_labels, validation_labels = train_test_split(X1, y,
                                                                                              random_state=2019,
                                                                                              test_size=0.2)
        train2_inputs, validation2_inputs, _, _ = train_test_split(X2, y,
                                                                   random_state=2019,
                                                                   test_size=0.2)
        train_entity_locs, validation_entity_locs, _, _ = train_test_split(locs, y,
                                                                           random_state=2019, test_size=0.2)

        train1_inputs = torch.tensor(train1_inputs)
        validation1_inputs = torch.tensor(validation1_inputs)
        train2_inputs = torch.tensor(train2_inputs)
        validation2_inputs = torch.tensor(validation2_inputs)
        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)
        train_entity_locs = torch.tensor(train_entity_locs)
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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize and vectorize
    tokenized_text = [tokenizer.tokenize(sentence) for sentence in sentences]
    X = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_text]

    # MAX_SEQ_LEN
    #MAX_LEN = max([len(x) for x in X])+1
    MAX_LEN = 50
    #Pad sequences to make them all eqally long
    X = pad_sequences(X, MAX_LEN, 'long', 'post', 'post')

    # Find the locations of each entity and store them
    if org:
        entity_locs = np.asarray([[i for i, s in enumerate(sent) if s == '<'] for sent in tokenized_text])
    else:
        entity_locs = np.asarray([[i for i, s in enumerate(sent) if s == '^'] for sent in tokenized_text])

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
    id = df['id']
    X = df['original'].values
    X = [sent.replace("\"","") for sent in X]
    replaced = df['original'].apply(lambda x: x[x.index("<"):x.index(">")+1])
    replaced_clean = [x.replace("<","").replace("/>","") for x in replaced]
    if mode=='train':
        y = df['meanGrade'].values
    edit = df['edit']

    org_tag = pos_tag(X, replaced)
    X1 = [sent.replace(replaced[i], "< " + replaced[i].strip("<|/>") + " " + org_tag[i] + " <") for i, sent in enumerate(X)]
    X2 = [sent.replace(replaced[i], "<"+edit[i]+"/>") for i, sent in enumerate(X)]
    edited_tag = pos_tag(X2,edit)
    X2 = [sent.replace("<"+edit[i]+"/>", "^ " + edit[i] + " " + edited_tag[i] + " ^") for i, sent in enumerate(X2)]


    X1,e1_locs = tokenize_bert(X1,True)
    X2,e2_locs = tokenize_bert(X2,False)
    replacement_locs = np.concatenate((e1_locs, e2_locs), 1)
    if model:
        word2vec_replaced = np.asarray([model.vocab[replaced_clean[i]].index if replaced_clean[i] in model else -1 for i in range(len(replaced))]).reshape(-1,1)
        word2vec_edited = np.asarray([model.vocab[edit[i]].index if edit[i] in model else -1 for i in range(len(edit))]).reshape(-1,1)
        word2vec_indices = np.concatenate((word2vec_replaced,word2vec_edited),1)

    if mode == "train":
        train1_inputs, validation1_inputs, train_labels, validation_labels = train_test_split(X1, y,
                                                                                            random_state=2019,
                                                                                            test_size=0.2)
        train2_inputs, validation2_inputs, _, _ = train_test_split(X2, y,
                                                                                              random_state=2019,
                                                                                              test_size=0.2)
        train_entity_locs, validation_entity_locs, _, _ = train_test_split(replacement_locs, y,
                                                                           random_state=2019, test_size=0.2)

        if model:
            train_word2vec_locs, validation_word2vec_locs, _, _ = train_test_split(word2vec_indices, y,
                                                                          random_state=2019, test_size=0.2)

        train1_inputs = torch.tensor(train1_inputs)
        validation1_inputs = torch.tensor(validation1_inputs)
        train2_inputs = torch.tensor(train2_inputs)
        validation2_inputs = torch.tensor(validation2_inputs)
        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)
        train_entity_locs = torch.tensor(train_entity_locs)
        validation_entity_locs = torch.tensor(validation_entity_locs)
        if model:
            train_word2vec_locs = torch.tensor(train_word2vec_locs)
            validation_word2vec_locs = torch.tensor(validation_word2vec_locs)

        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
        # with an iterator the entire dataset does not need to be loaded into memory

        #train_data = TensorDataset(train1_inputs,train2_inputs, train_entity_locs, train_word2vec_locs, train_labels)
        train_data = TensorDataset(train1_inputs, train2_inputs, train_entity_locs, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

        #validation_data = TensorDataset(validation1_inputs,validation2_inputs, validation_entity_locs, validation_word2vec_locs, validation_labels)
        validation_data = TensorDataset(validation1_inputs, validation2_inputs, validation_entity_locs,validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=test_batch_size)
        if model:
            return train_dataloader, validation_dataloader, word2vec_indices
        else:
            return  train_dataloader, validation_dataloader

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


    X1 = get_modified_sentence("original1","edit1")
    X2 = get_modified_sentence("original2","edit2")
    X1,e1_locs = tokenize_bert(X1,False)
    X2,e2_locs = tokenize_bert(X2,False)
    if mode == 'train':
        y = df['meanGrade1'] > df['meanGrade2']
        y = [0 if label else 1 for label in y]

    replacement_locs = np.concatenate((e1_locs, e2_locs), 1)

    if mode == "train":
        train1_inputs, validation1_inputs, train_labels, validation_labels = train_test_split(X1, y,
                                                                                            random_state=2019,
                                                                                            test_size=0.2)
        train2_inputs, validation2_inputs, _, _ = train_test_split(X2, y,
                                                                                              random_state=2019,
                                                                                              test_size=0.2)
        train_entity_locs, validation_entity_locs, _, _ = train_test_split(replacement_locs, y,
                                                                           random_state=2019, test_size=0.2)

        train1_inputs = torch.tensor(train1_inputs)
        validation1_inputs = torch.tensor(validation1_inputs)
        train2_inputs = torch.tensor(train2_inputs)
        validation2_inputs = torch.tensor(validation2_inputs)
        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)
        train_entity_locs = torch.tensor(train_entity_locs)
        validation_entity_locs = torch.tensor(validation_entity_locs)

        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
        # with an iterator the entire dataset does not need to be loaded into memory

        train_data = TensorDataset(train1_inputs,train2_inputs, train_entity_locs, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

        validation_data = TensorDataset(validation1_inputs,validation2_inputs, validation_entity_locs, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=test_batch_size)
        return train_dataloader, validation_dataloader

    if mode == "test":
        test1_input = torch.tensor(X1)
        test2_input = torch.tensor(X2)
        train_entity_locs = torch.tensor(replacement_locs)
        test_data = TensorDataset(test1_input, test2_input, train_entity_locs)
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)

        return test_data_loader


if __name__ == "__main__":
    convert_task2_to_task1()
    #get_dataloaders("../data/task-1/train.csv")