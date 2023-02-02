import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn

def random_forest(data):
    train_features = data['train_features']
    val_features = data['val_features']
    test_features = data['test_features']

    train_labels = data['train_labels']
    val_labels = data['val_labels']
    test_labels = data['test_labels']
    
    n = [100, 250, 500]
    criteria = ['gini', 'entropy', 'log_loss']
    max_depths = [None,1,3,5,10,20]
    min_impurity_decreases = [0.0, 1e-8, 1e-4, 0.001]

    results = {}
    for i in np.ndindex(len(n),len(criteria),len(max_depths),len(min_impurity_decreases)):
        rfc = RandomForestClassifier(n_estimators=n[i[0]], criterion=criteria[i[1]],max_depth=max_depths[i[2]],\
            min_impurity_decrease=min_impurity_decreases[i[3]], class_weight='balanced')

        rfc.fit(train_features, train_labels)
        results[i] = rfc.score(val_features, val_labels)

    # Best Model
    results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    best_config = next(iter(results.keys()))
    rfc = RandomForestClassifier(n_estimators=n[best_config[0]], criterion=criteria[best_config[1]],\
        max_depth=max_depths[best_config[2]],min_impurity_decrease=min_impurity_decreases[best_config[3]],\
        class_weight='balanced')
    rfc.fit(train_features, train_labels)
    test_acc = rfc.score(test_features, test_labels)

    print(f'RFC: n_estimators={n[best_config[0]]}, criterion={criteria[best_config[1]]}, max_depth={max_depths[best_config[2]]},\
            min_impurity_decrease={min_impurity_decreases[best_config[3]]}, class_weight="balanced"')
    print(f'Accuracy: {test_acc:2.4f}')

def svm(data):
    train_features = data['train_features']
    val_features = data['val_features']
    test_features = data['test_features']

    train_labels = data['train_labels']
    val_labels = data['val_labels']
    test_labels = data['test_labels']
    
    c = [0.1, 0.5, 1, 2, 5, 10]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    degree = [3,5,7,10]

    results = {}
    for i in np.ndindex(len(c),len(kernel),len(degree)):
        svc = SVC(C=c[i[0]], kernel=kernel[i[1]],degree=degree[i[2]],class_weight='balanced')
        svc.fit(train_features, train_labels)
        results[i] = svc.score(val_features, val_labels)

    # Best Model
    results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    best_config = next(iter(results.keys()))
    svc = SVC(C=c[best_config[0]], kernel=kernel[best_config[1]],degree=degree[best_config[2]],class_weight='balanced')
    svc.fit(train_features, train_labels)
    test_acc = svc.score(test_features, test_labels)

    print(f'SVM: C={c[best_config[0]]}, kernel={kernel[best_config[1]]}, degree={degree[best_config[2]]}')
    print(f'Accuracy: {test_acc:2.4f}')

if __name__ == '__main__':
    with open('features.pkl', 'rb') as file:
        data = pickle.load(file)

    svm(data)