import pickle
import numpy as np

if __name__ == '__main__':
    train_path = '../database/train_filtered.pkl'
    val_path = '../database/val_filtered.pkl'
    test_path = '../database/test_filtered.pkl'

    # DB: 9396 subjects in total    
    with open(train_path, 'rb') as file:
        train_data = pickle.load(file) # 7401

    with open(val_path, 'rb') as file:
        val_data = pickle.load(file) # 822

    with open(test_path, 'rb') as file:
        test_data = pickle.load(file) # 1173


    
    

    

    
    