import gensim
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pathlib import Path

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

    dim = 64
    doc2vec_model_path = f'../database/doc2vec_twibot20_{dim}'
    doc2vec_features_path = f'../database/doc2vec_{dim}.pkl'

    if Path(doc2vec_features_path).exists() == False:
        with open(train_path, 'rb') as file:
            subject_dict_train = pickle.load(file) # 7401

        with open(val_path, 'rb') as file:
            subject_dict_val = pickle.load(file) # 822

        with open(test_path, 'rb') as file:
            subject_dict_test = pickle.load(file) # 1173

        subject_dict_train = {k:v for k,v in subject_dict_train.items() if v[1] != []}
        subject_dict_val = {k:v for k,v in subject_dict_val.items() if v[1] != []}
        subject_dict_test = {k:v for k,v in subject_dict_test.items() if v[1] != []}

        if Path(doc2vec_model_path).exists() == False:
            train_data = [TaggedDocument(v[1], tags=v[0]) for _,v in subject_dict_train.items()]
            model = Doc2Vec(train_data, vector_size=dim, min_count=2, epochs=50)
            model.save(doc2vec_model_path)
        else:
            model = Doc2Vec.load(doc2vec_model_path) 

        train_data, train_labels = [model.infer_vector(v[1]) for _,v in subject_dict_train.items()], [v[0] for _,v in subject_dict_train.items()]
        val_data, val_labels = [model.infer_vector(v[1]) for _,v in subject_dict_val.items()], [v[0] for _,v in subject_dict_val.items()]
        test_data, test_labels = [model.infer_vector(v[1]) for _,v in subject_dict_test.items()], [v[0] for _,v in subject_dict_test.items()]

        data = {
                'train_data':train_data,
                'train_labels':train_labels,
                'val_data':val_data,
                'val_labels':val_labels,
                'test_data':test_data,
                'test_labels':test_labels
            }

        with open(doc2vec_features_path, 'wb') as file:
            pickle.dump(data, file)
    else:
        train_data, train_labels, val_data, val_labels, test_data, test_labels = open_pickle(doc2vec_features_path)