import gensim
import pickle
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from pathlib import Path
import numpy as np

def open_pickle(path):
    with open(path, 'rb') as file:
        database = pickle.load(file)

        train_data, train_labels, val_data, val_labels, test_data, test_labels = database['train_data'], database['train_labels'],\
            database['val_data'], database['val_labels'], database['test_data'], database['test_labels']

        return train_data, train_labels, val_data, val_labels, test_data, test_labels

if __name__ == '__main__':
    train_path = '../database/train_filtered.pkl'
    val_path = '../database/val_filtered.pkl'
    test_path = '../database/test_filtered.pkl'

    dim = 512
    word2vec_model_path = f'../database/word2vec_twibot20_{dim}'
    word2vec_wv_path = f'../database/word2vec_twibot20_wv_{dim}'
    word2vec_features_path = f'../database/word2vec_{dim}.pkl'

    if Path(word2vec_features_path).exists() == False:
        with open(train_path, 'rb') as file:
            subject_dict_train = pickle.load(file) # 7401

        with open(val_path, 'rb') as file:
            subject_dict_val = pickle.load(file) # 822

        with open(test_path, 'rb') as file:
            subject_dict_test = pickle.load(file) # 1173

        subject_dict_train = {k:v for k,v in subject_dict_train.items() if v[1] != []}
        subject_dict_val = {k:v for k,v in subject_dict_val.items() if v[1] != []}
        subject_dict_test = {k:v for k,v in subject_dict_test.items() if v[1] != []}

        if Path(word2vec_model_path).exists() == False:
            train_data = [x.split() for _,v in subject_dict_train.items() for x in v[1]]
            model = Word2Vec(sentences=train_data, vector_size=dim, window=5, min_count=1, workers=4)
            model.save(word2vec_model_path)
            model.wv.save(word2vec_wv_path)
            wv = model.wv
        else:
            # model = Word2Vec.load(word2vec_model_path)
            wv = KeyedVectors.load(word2vec_wv_path)

        train_data = [np.mean(np.asarray([np.mean(np.asarray([wv[x] if x in wv else np.zeros((dim,)) for x in tweet.split()]),\
            axis=0, dtype=np.float64) for tweet in v[1]]), axis=0, dtype=np.float64) for _,v in subject_dict_train.items()]

        val_data = [np.mean(np.asarray([np.mean(np.asarray([wv[x] if x in wv else np.zeros((dim,)) for x in tweet.split()]),\
            axis=0, dtype=np.float64) for tweet in v[1]]), axis=0, dtype=np.float64) for _,v in subject_dict_val.items()]

        test_data = [np.mean(np.asarray([np.mean(np.asarray([wv[x] if x in wv else np.zeros((dim,)) for x in tweet.split()]),\
            axis=0, dtype=np.float64) for tweet in v[1]]), axis=0, dtype=np.float64) for _,v in subject_dict_test.items()]

        train_labels = [v[0] for _,v in subject_dict_train.items()]
        val_labels = [v[0] for _,v in subject_dict_val.items()]
        test_labels = [v[0] for _,v in subject_dict_test.items()]

        data = {
                'train_data':train_data,
                'train_labels':train_labels,
                'val_data':val_data,
                'val_labels':val_labels,
                'test_data':test_data,
                'test_labels':test_labels
            }

        with open(word2vec_features_path, 'wb') as file:
            pickle.dump(data, file)
    else:
        train_data, train_labels, val_data, val_labels, test_data, test_labels = open_pickle(word2vec_features_path)