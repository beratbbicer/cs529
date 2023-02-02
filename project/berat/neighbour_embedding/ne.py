import networkx as nx
import pandas as pd
import numpy as np

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

def standardize_scores(data):
    scores = np.asarray([data[k][1] for k in data], dtype=np.float64)
    mean, std = scores.mean(), scores.std()
    
    for k in data:
        data[k][1] = (data[k][1] - mean) / std

    minv = np.asarray([data[k][1] for k in data], dtype=np.float64).min()
    for k in data:
        data[k][1] -= minv

def normalize_scores(data):
    scores = np.asarray([data[k][1] for k in data], dtype=np.float64)
    minv, maxv = scores.min(), scores.max()
    
    for k in data:
        data[k][1] = (data[k][1] - minv) / (maxv - minv)

def compute_accuracy(data, threshold):
    # 1 bot, 0 real
    corrects, counts = [0.0, 0.0], [1e-8, 1e-8]

    for k,v in data.items():
        if v[1] > threshold:
            prediction = 1
        else:
            prediction = 0

        if prediction == v[0]:
            corrects[v[0]] += 1
        
        counts[v[0]] += 1

    if threshold > 0.40033:
        _ = 1

    return sum(corrects) / sum(counts), [corrects[i] / counts[i] for i in range(len(corrects))]

if __name__ == '__main__':
    edges_path = "/home/bb/Bilkent/phd/courses/cs529/project/Twitter/TwiBot/edges_filtered.csv"
    nodes_path = "/home/bb/Bilkent/phd/courses/cs529/project/Twitter/TwiBot/nodes_filtered.csv"
    graph = compute_graph(edges_path, nodes_path)

    vertex_data = {}
    for vertex, data in graph.nodes(data=True):
        neighbours = [str(graph.nodes[n]['Label']) for n in graph.neighbors(vertex)]
        vertex_data[vertex] = [data["Label"], int(''.join(neighbours), base=2)]

    # Standardize the scores
    normalize_scores(vertex_data)
    scores = [v[1] for k,v in vertex_data.items()]
    min_score, max_score, mean_score = min(scores), max(scores), np.mean(scores)
    threshold_search_space = np.hstack((np.linspace(min_score,mean_score,1000,endpoint=False),\
        np.linspace(mean_score,max_score,100,endpoint=False)))

    best_thresholds, best_accuracies, best_pairwise_accuracies = [], [], []
    for threshold in threshold_search_space:
        accuracy, pairwise_accuracy = compute_accuracy(vertex_data, threshold)

        if len(best_thresholds) < 10:
            best_thresholds.append(threshold)
            best_accuracies.append(accuracy)
            best_pairwise_accuracies.append(pairwise_accuracy)
        else:
            worst_threshold = min(best_thresholds)
            if accuracy > worst_threshold:
                idx = best_thresholds.index(worst_threshold)
                best_thresholds[idx] = threshold
                best_accuracies[idx] = accuracy
                best_pairwise_accuracies[idx] = pairwise_accuracy

    for row in sorted(zip(best_thresholds, best_accuracies, best_pairwise_accuracies), key=lambda item: item[1]):
        threshold, accuracy, pairwise_accuracy = row
        print(f"Threshold: {threshold:1.8f}, Acc: {accuracy:2.3f}")
        print(f"Class acc --> real(0): {pairwise_accuracy[0]:1.3f}, bot(1): {pairwise_accuracy[1]:2.3f}\n")