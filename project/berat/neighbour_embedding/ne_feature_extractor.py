import networkx as nx
import pandas as pd
import numpy as np
import pickle

def compute_graph(edges_path, nodes_path):
    edges = pd.read_csv(edges_path, sep=",")[["Source","Target","Weight"]]
    # values = list(edges.columns.values)
    # graph = nx.from_pandas_edgelist(edges, 'Source', 'Target', ['Weight'])
    graph = nx.from_pandas_edgelist(edges, 'Source', 'Target')

    nodes = pd.read_csv(nodes_path, sep=',').drop(['timeset','follower count','friends count'], axis=1)
    # values = list(nodes.columns.values)
    data = nodes.set_index('Id').to_dict('index').items()
    graph.add_nodes_from(data)
    return graph

def train_val_test_split(features, labels, ratio):
    # Round 1: Split All into train and test
    test_idx = np.random.choice(len(features), size=int(len(features) * ratio), replace=False)
    train_idx = np.asarray([i for i in range(len(features)) if i not in test_idx])
    train_features, train_labels = features[train_idx], labels[train_idx]
    test_features, test_labels = features[test_idx], labels[test_idx]

    # Round 2: Split Test into val and test
    val_idx = np.random.choice(len(test_features), size=int(len(test_features) / 2), replace=False)
    test_idx = [i for i in range(len(test_features)) if i not in val_idx]

    val_features, val_labels = test_features[val_idx], test_labels[val_idx]
    test_features, test_labels = test_features[test_idx], test_labels[test_idx]

    return train_features, train_labels, val_features, val_labels, test_features, test_labels

def dump_data(edges_path, nodes_path):
    graph = compute_graph(edges_path, nodes_path)

    # Compute metric
    max_length = 0
    vertex_data = []
    for vertex, data in graph.nodes(data=True):
        neighbours = [graph.nodes[n]['Label'] for n in graph.neighbors(vertex)]
        vertex_data += [[vertex, data["Label"], neighbours]]

        if len(neighbours) > max_length:
            max_length = len(neighbours)
    
    # pad the features
    for i in range(len(vertex_data)):
        vertex_data[i][2] = [0 for i in range(max_length - len(vertex_data[i][2]))] + vertex_data[i][2]

    # train and test split
    all_features, all_labels = np.asarray([v[2] for v in vertex_data]).reshape((-1,max_length)),\
        np.asarray([v[1] for v in vertex_data]).reshape((-1))
    # Centralize to 0
    all_features = all_features.astype(np.float64) + (np.zeros((1,max_length)) - 0.5)
        
    ratio = 0.2
    real_idx = np.where(all_labels == 0)
    real_features, real_labels = all_features[real_idx], all_labels[real_idx]
    train_real_features, train_real_labels, val_real_features, val_real_labels, test_real_features, test_real_labels = \
        train_val_test_split(real_features, real_labels, ratio)

    bot_idx = np.where(all_labels == 1)
    bot_features, bot_labels = all_features[bot_idx], all_labels[bot_idx]
    train_bot_features, train_bot_labels, val_bot_features, val_bot_labels, test_bot_features, test_bot_labels = \
        train_val_test_split(bot_features, bot_labels, ratio)

    train_features = np.vstack((train_real_features,train_bot_features))
    val_features = np.vstack((val_real_features, val_bot_features))
    test_features = np.vstack((test_real_features, test_bot_features))
    train_labels = np.concatenate((train_real_labels, train_bot_labels), axis=0)
    val_labels = np.concatenate((val_real_labels, val_bot_labels), axis=0)
    test_labels = np.concatenate((test_real_labels, test_bot_labels), axis=0)

    data = {
        'train_features':train_features, 
        'train_labels':train_labels,
        'val_features':val_features, 
        'val_labels':val_labels, 
        'test_features':test_features,
        'test_labels':test_labels,
    }

    with open('features.pkl', 'wb') as file:
        pickle.dump(data, file)

if __name__ == '__main__':
    edges_path = "/home/bb/Bilkent/phd/courses/cs529/project/Twitter/TwiBot/edges_filtered.csv"
    nodes_path = "/home/bb/Bilkent/phd/courses/cs529/project/Twitter/TwiBot/nodes_filtered.csv"
    dump_data(edges_path, nodes_path)