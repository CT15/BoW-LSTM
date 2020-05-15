import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from nltk.corpus import stopwords

def get_word2idx():
    if not os.path.exists('./data/bow_mapping.csv'):
        data = pd.read_csv('./data/fixed_result_final_m.csv', comment='#')
        
        temp_dict = dict()

        for _, row in data.iterrows():
            string = row['posts']
            for word in string.split():
                current_val = temp_dict.get(word, 0)
                temp_dict[word] = (current_val + 1)

        words = []
        sw = set(stopwords.words('english'))
        for word, count in sorted(temp_dict.items(), key=lambda kv: kv[1], reverse=True):
            if word in sw or len(word) < 2:
                continue

            words.append(word)
            
            if len(words) == 1000:
            #if len(words) == 300:
                break

        assert len(words) == len(set(words))
        
        df = pd.DataFrame(words, columns=['word'])
        df.to_csv('./data/bow_mapping.csv', index=False)

    data = pd.read_csv('./data/bow_mapping.csv', comment='#')

    word2idx = dict()
    for index, row in data.iterrows():
        word2idx[row['word']] = index

    next_index = len(word2idx)
    word2idx['<unk>'] = next_index
    word2idx['<ins>'] = next_index + 1

    assert len(word2idx) == 1002
    assert len(set(word2idx.values())) == len(word2idx)

    return word2idx


def load_test_train_val(number):
    # number is int from 1 to 5
    base_dir = f'./separated_data/data{number}'
    test = pd.read_csv(f'{base_dir}/test.csv', comment='#')
    train = pd.read_csv(f'{base_dir}/train.csv', comment='#')
    val = pd.read_csv(f'{base_dir}/val.csv', comment='#')
    return test, train, val


def to_bow(s, ins, word2idx):
    bow = np.zeros(len(word2idx))

    for word in s.split():
        if word in word2idx:
            bow[word2idx[word]] += 1
        else:
            bow[word2idx['<unk>']] += 1

    # normalize
    bow = bow / np.linalg.norm(bow)

    bow[word2idx['<ins>']] = ins

    return bow.tolist()


def get_vectors_labels(data_df, word2idx):
    labels = list(data_df.new_labels)
    vectors = []
    for index, row in data_df.iterrows():
        vectors.append(to_bow(row['posts'], row['labels'], word2idx))
        assert index + 1 == len(vectors)

    labels = np.array(labels)
    vectors = np.array(vectors)

    assert labels.shape == (len(data_df),)
    assert vectors.shape == (len(data_df), len(word2idx))

    return vectors.tolist(), labels.tolist()


def get_padding(word2idx):
    pad = np.zeros(len(word2idx))
    return pad.tolist()


def get_padded_data(df, vectors, labels, seq_len, word2idx):
    data_indices = []
    thread_ids = df.thread_id.unique()
    
    for thread_id in thread_ids:
        idx = list(df[df.thread_id == thread_id].index)
        data_indices.append(idx)
    
    lengths = [len(indices) for indices in data_indices]
    
    grouped_vectors = [] # vectors grouped by thread
    grouped_labels = []

    for indices in data_indices:
        temp_vectors = [vectors[index] for index in indices]
        temp_labels = [labels[index] for index in indices]
        
        assert len(temp_vectors) == len(temp_labels)
        assert len(indices) <= seq_len
            
        if len(indices) < seq_len:
            temp_vectors = temp_vectors + ([get_padding(word2idx)] * (seq_len - len(indices)))
            temp_labels = temp_labels + [0] * (seq_len - len(indices))

        assert len(temp_vectors) == seq_len
        assert len(temp_labels) == seq_len

        grouped_vectors.append(temp_vectors)
        grouped_labels.append(temp_labels)

    lengths = np.array(lengths)
    grouped_vectors = np.array(grouped_vectors)
    grouped_labels = np.array(grouped_labels)
    
    assert lengths.shape == (len(thread_ids),)
    assert grouped_vectors.shape == (len(thread_ids), seq_len, len(vectors[0]))
    assert grouped_labels.shape == (len(thread_ids), seq_len,)

    return grouped_vectors, grouped_labels, lengths

def to_data_loader(vectors, labels, lengths, batch_size):
    inputs = np.array(vectors)
    labels = np.array(labels)
    lengths = np.array(lengths)

    data = TensorDataset(torch.from_numpy(inputs).type('torch.FloatTensor'), 
                         torch.from_numpy(labels),
                         torch.from_numpy(lengths))

    return DataLoader(data, shuffle=False, batch_size=batch_size, drop_last=True)


def get_train_test_val_loaders(number, seq_len, batch_size, word2idx):
    test, train, val = load_test_train_val(number)

    train_vectors, train_labels = get_vectors_labels(train, word2idx)
    test_vectors, test_labels = get_vectors_labels(test, word2idx)
    val_vectors, val_labels = get_vectors_labels(val, word2idx)

    train_vectors, train_labels, train_lengths = get_padded_data(train, train_vectors, train_labels, seq_len, word2idx)
    test_vectors, test_labels, test_lengths = get_padded_data(test, test_vectors, test_labels, seq_len, word2idx)
    val_vectors, val_labels, val_lengths = get_padded_data(val, val_vectors, val_labels, seq_len, word2idx)

    train_loader = to_data_loader(train_vectors, train_labels, train_lengths, batch_size)
    test_loader = to_data_loader(test_vectors, test_labels, test_lengths, batch_size)
    val_loader = to_data_loader(val_vectors, val_labels, val_lengths, batch_size)

    return train_loader, test_loader, val_loader


def evaluate(model, data_loader, batch_size):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    a = []
    b = []

    model.eval()

    for inputs, labels, lengths in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        predictions, _ = model(inputs, lengths)
        _, predictions, truths = model.loss(predictions, labels, lengths)
        
        predictions = predictions.tolist()
        truths = truths.tolist()

        a.append(predictions)
        b.append(truths)

    a = [int(pred) for predlist in a for pred in predlist]
    b = [int(truth) for truthlist in b for truth in truthlist]

    model.train()
    
    f1 = f1_score(b, a)
    precision = precision_score(b, a)
    recall = recall_score(b, a)
    accuracy = accuracy_score(b, a)
    conf_matrix = confusion_matrix(b, a)

    return f1, precision, recall, accuracy, conf_matrix
