import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

class BasicNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(28,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(64,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Linear(32,16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.Linear(16,2),
        )

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    # ====================================================================
    # Data Preparation

    with open('features.pkl', 'rb') as file:
        data = pickle.load(file)

    train_features = data['train_features']
    val_features = data['val_features']
    test_features = data['test_features']

    train_labels = data['train_labels']
    val_labels = data['val_labels']
    test_labels = data['test_labels']
    
    batch_size = 64

    train_dataset = TensorDataset(torch.from_numpy(train_features).double(), torch.from_numpy(train_labels).long())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(torch.from_numpy(val_features).double(), torch.from_numpy(val_labels).long())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(torch.from_numpy(test_features).double(), torch.from_numpy(test_labels).long())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # ====================================================================
    # Train Loop

    lr = 1e-3
    max_epochs = 50
    device = torch.device('cuda:0')
    model = BasicNN().double().to(device)
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
    plt.savefig('experiment.png',bbox_inches='tight')