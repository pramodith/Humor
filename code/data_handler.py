import torch
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json

def create_vocab_dict(X : list):
    max_seq_len = max([len(X[i]) for i in range(len(X))])
    sentences = [X[i].split(" ") for i in range(len(X))]
    words = [word for sent in sentences for word in sent]
    words = set(words)
    words.add("<UNK>")
    words.add("<SOS>")
    words.add("<EOS>")
    word2ind = {}
    for ind, word in enumerate(words):
        word2ind[word] = ind + 1
    json.dump(word2ind, "word2ind.json", indent=5)


def tokenize(X :list):
    X = [[word2ind[sent[ind]] if ind<max_seq_len else word2ind["<EOS>"] for ind in range(max_seq_len) ] for sent in sentences]
    X = [sent.insert(0,word2ind["<SOS>"]) for sent in sentences]
    return X


def get_dataloaders(file_path : str ,mode="train",train_batch_size=64,test_batch_size = 64):
    df = pd.read_csv(file_path, sep=",")
    id = df['id']
    X = df['original'].apply(lambda sent: sent.replace("\"","").replace())
    replaced = df['original'].apply(lambda x: x[x.index("<"):x.index(">") + 1])
    if mode == 'train':
        y = df['meanGrade'].values
    edit = df['edit']
    words = X + df['edit']
    X2 = [sent.replace(replaced[i],edit[i]) for i, sent in enumerate(X)]
    X1 = [sent.replace("<","").replace("/>", "") for i, sent in enumerate(X)]
    locs = [X1[i].split(" ").index(replaced[i]) for i in range(len(X1))]

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
    tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'tokenizer', 'bert-base-uncased',
                               do_basic_tokenize=True)

    # Tokenize and vectorize
    tokenized_text = [tokenizer.tokenize(sentence) for sentence in sentences]
    X = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_text]

    # MAX_SEQ_LEN
    MAX_LEN = max([len(x) for x in X])

    #Pad sequences to make them all eqally long
    X = pad_sequences(X, MAX_LEN, 'long', 'post', 'post')

    # Find the locations of each entity and store them
    if org:
        entity_locs = np.asarray([[i for i, s in enumerate(sent) if s == '<'] for sent in tokenized_text])
    else:
        entity_locs = np.asarray([[i for i, s in enumerate(sent) if s == '^'] for sent in tokenized_text])

    return X,entity_locs

def get_dataloaders_bert(file_path : str ,mode="train",train_batch_size=64,test_batch_size = 64):
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
    if mode=='train':
        y = df['meanGrade'].values
    edit = df['edit']
    X2 = [sent.replace(replaced[i], "^ " + edit[i] + " ^") for i, sent in enumerate(X)]
    X1 = [sent.replace("<","< ").replace("/>"," <") for i,sent in enumerate(X)]
    X1,e1_locs = tokenize_bert(X1,True)
    X2,e2_locs = tokenize_bert(X2,False)
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
        test_data = TensorDataset(torch.tensor(X), torch.tensor(id))
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)

        return test_data_loader


if __name__ == "__main__":
    get_dataloaders("../data/task-1/train.csv")