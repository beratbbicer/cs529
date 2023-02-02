import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

class BasicNN(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(dim,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

            nn.Linear(1024,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(256,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(64,16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.Linear(16,2),
        )

    def forward(self, x):
        return self.layers(x)

def open_pickle(path):
    with open(path, 'rb') as file:
        database = pickle.load(file)

        train_data, train_labels, val_data, val_labels, test_data, test_labels = database['train_data'], database['train_labels'],\
            database['val_data'], database['val_labels'], database['test_data'], database['test_labels']

        return np.asarray(train_data,dtype=np.float64), np.asarray(train_labels,dtype=np.int64),\
            np.asarray(val_data,dtype=np.float64), np.asarray(val_labels,dtype=np.int64),\
            np.asarray(test_data,dtype=np.float64), np.asarray(test_labels,dtype=np.int64)

if __name__ == '__main__':
    dim = 512
    # datapath = f'../database/doc2vec_{dim}.pkl'
    datapath = f'../database/word2vec_{dim}.pkl'

    train_data, train_labels, val_data, val_labels, test_data, test_labels = open_pickle(datapath)
    batch_size = 256

    train_dataset = TensorDataset(torch.from_numpy(train_data).double(), torch.from_numpy(train_labels).long())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(torch.from_numpy(val_data).double(), torch.from_numpy(val_labels).long())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(torch.from_numpy(test_data).double(), torch.from_numpy(test_labels).long())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # ====================================================================
    # Train Loop
    lr = 1e-3
    max_epochs = 50
    device = torch.device('cuda:0')
    model = BasicNN(dim).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-5, cooldown=2, verbose=True)
    loss_function = nn.CrossEntropyLoss()
    epoch_train_losses, epoch_val_losses, epoch_test_losses = [], [], []

    for epoch in range(max_epochs):
        # ************************************************************
        # Training
        model.train()
        train_loss, train_loss = 0, 0

        for minibatch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            loss = loss_function(output, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / float(train_loader.dataset.__len__())
        print(f'[{epoch:2d}] T.L.: {train_loss:.4f}')
        epoch_train_losses.append(train_loss)

        # ************************************************************
        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for minibatch_idx, (data, labels) in enumerate(val_loader):
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                loss = loss_function(output, labels)
                val_loss += loss.item()

        val_loss = val_loss / float(val_loader.dataset.__len__())
        print(f'[{epoch:2d}] V.L.: {val_loss:.4f}')
        epoch_val_losses.append(val_loss)

        # ************************************************************
        # Test
        test_loss = 0
        corrects = 0.0
        cw_corrects,cw_counts = [0.0,0.0],[1e-8,1e-8]

        with torch.no_grad():
            for minibatch_idx, (data, labels) in enumerate(test_loader):
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                loss = loss_function(output, labels)
                test_loss += loss.item()

                (_, predictions) = torch.max(F.softmax(output, dim=1), dim=1)

                for i in range(len(predictions)):
                    if predictions[i] == labels[i]:
                        cw_corrects[labels[i].item()] += 1
                    cw_counts[labels[i].item()] += 1

                max_index = output.max(dim = 1)[1]
                corrects += (max_index == labels).sum()

        test_loss = test_loss / float(test_loader.dataset.__len__())
        cwas = [cw_corrects[i] / cw_counts[i] for i in range(len(cw_corrects))]
        print(f'[{epoch:2d}] T.L.: {test_loss:.4f}, Acc: {corrects / float(test_loader.dataset.__len__()):0.4f}')
        print(f'[{epoch:2d}] CWA --> Real: {cwas[0]:.4f}, Bot: {cwas[1]:.4f}\n')

        epoch_test_losses.append(test_loss)

    plt.figure(figsize=(12,8))
    plt.plot([i for i in range(max_epochs)], epoch_train_losses, 'r', label='Train')
    plt.plot([i for i in range(max_epochs)], epoch_val_losses, 'b', label='Val')
    plt.plot([i for i in range(max_epochs)], epoch_test_losses, 'g', label='Test')
    plt.legend(loc="upper right")
    plt.xlabel('Epoch')
    plt.ylabel('Standardized Loss')
    plt.title('Loss vs. Epochs')
    plt.savefig(f'experiment_doc2vec_{dim}.png',bbox_inches='tight')


    '''
    # =================================================
    # Word2Vec Avg
    64 ->
    [ 4] T.L.: 0.0019
    [ 4] V.L.: 0.0027
    [ 4] T.L.: 0.0024, Acc: 0.7158
    [ 4] CWA --> Real: 0.6387, Bot: 0.7813

    128 ->
    [ 2] T.L.: 0.0021
    [ 2] V.L.: 0.0030
    [ 2] T.L.: 0.0026, Acc: 0.6901
    [ 2] CWA --> Real: 0.6853, Bot: 0.6941

    256 ->
    [ 5] T.L.: 0.0019
    [ 5] V.L.: 0.0028
    [ 5] T.L.: 0.0025, Acc: 0.7166
    [ 5] CWA --> Real: 0.7281, Bot: 0.7068

    512 ->
    [ 8] T.L.: 0.0018
    [ 8] V.L.: 0.0029
    [ 8] T.L.: 0.0026, Acc: 0.7080
    [ 8] CWA --> Real: 0.7244, Bot: 0.6941

    # =================================================
    # Doc2Vec
    64 ->
    [ 7] T.L.: 0.0006
    [ 7] V.L.: 0.0040
    [ 7] T.L.: 0.0039, Acc: 0.5822
    [ 7] CWA --> Real: 0.5214, Bot: 0.6339

    128 ->
    [13] T.L.: 0.0006
    [13] V.L.: 0.0044
    [13] T.L.: 0.0043, Acc: 0.5916
    [13] CWA --> Real: 0.5121, Bot: 0.6593

    256 -> 
    [ 7] T.L.: 0.0006
    [ 7] V.L.: 0.0041
    [ 7] T.L.: 0.0039, Acc: 0.6036
    [ 7] CWA --> Real: 0.4525, Bot: 0.7322

    512 -> 
    [13] T.L.: 0.0006
    [13] V.L.: 0.0048
    [13] T.L.: 0.0048, Acc: 0.6122
    [13] CWA --> Real: 0.5102, Bot: 0.6989

    1024 ->
    [12] T.L.: 0.0006
    [12] V.L.: 0.0048
    [12] T.L.: 0.0044, Acc: 0.5959
    [12] CWA --> Real: 0.4581, Bot: 0.7132

    '''