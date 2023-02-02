import pickle
import numpy as np
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from pathlib import Path

def open_pickle(path):
    with open(path, 'rb') as file:
        database = pickle.load(file)

        train_data, train_labels, val_data, val_labels, test_data, test_labels = database['train_data'], database['train_labels'],\
            database['val_data'], database['val_labels'], database['test_data'], database['test_labels']

        return train_data, train_labels, val_data, val_labels, test_data, test_labels

if __name__ == '__main__':
    tfidf_path = '../database/tfidf.pkl'
    if Path(tfidf_path).exists() == False:
        train_path = '../database/train_filtered.pkl'
        val_path = '../database/val_filtered.pkl'
        test_path = '../database/test_filtered.pkl'

        # DB: 9396 subjects in total    
        with open(train_path, 'rb') as file:
            subject_dict_train = pickle.load(file) # 7401

        with open(val_path, 'rb') as file:
            subject_dict_val = pickle.load(file) # 822

        with open(test_path, 'rb') as file:
            subject_dict_test = pickle.load(file) # 1173

        subject_dict_train = {k:v for k,v in subject_dict_train.items() if v[1] != []}
        subject_dict_val = {k:v for k,v in subject_dict_val.items() if v[1] != []}
        subject_dict_test = {k:v for k,v in subject_dict_test.items() if v[1] != []}

        # Vectorization
        vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1,1))
        vectorizer.fit([x for _,v in subject_dict_train.items() for x in v[1]])
        train_data, train_labels = [vectorizer.transform(v[1]) for _,v in subject_dict_train.items()], [v[0] for _,v in subject_dict_train.items()]
        val_data, val_labels = [vectorizer.transform(v[1]) for _,v in subject_dict_val.items()], [v[0] for _,v in subject_dict_val.items()]
        test_data, test_labels = [vectorizer.transform(v[1]) for _,v in subject_dict_test.items()], [v[0] for _,v in subject_dict_test.items()]

        data = {
            'train_data':train_data,
            'train_labels':train_labels,
            'val_data':val_data,
            'val_labels':val_labels,
            'test_data':test_data,
            'test_labels':test_labels
        }

        with open(tfidf_path, 'wb') as file:
            pickle.dump(data, file)
    else:
        train_data, train_labels, val_data, val_labels, test_data, test_labels = open_pickle(tfidf_path)

    pca_path = '../database/tfidf_pca.pkl'
    if Path(pca_path).exists() == False:
        # Dimensionality Reduction by PCA
        indims = 1024
        pca = TruncatedSVD(n_components=indims, n_iter=10, random_state=69420)
        pca.fit(scipy.sparse.vstack(train_data))
        train_data = [pca.transform(v) for v in train_data]
        val_data = [pca.transform(v) for v in val_data]
        test_data = [pca.transform(v) for v in test_data]

        data = {
            'train_data':train_data,
            'train_labels':train_labels,
            'val_data':val_data,
            'val_labels':val_labels,
            'test_data':test_data,
            'test_labels':test_labels
        }

        with open(pca_path, 'wb') as file:
            pickle.dump(data, file)
    else:
        train_data, train_labels, val_data, val_labels, test_data, test_labels = open_pickle(pca_path)