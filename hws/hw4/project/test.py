import numpy as np

num_nodes, num_clusters = 77, 11
nodes, clusters = [], []
nodes_clusters = np.zeros((num_nodes, num_clusters)).astype(np.int64)

with open('bipartite_nodes-clusters.csv', 'r') as file:
    file.readline()
    for line in file:
        node, cluster = line.strip().split(',')

        if node not in nodes:
            nodes += [node]

        if cluster not in clusters:
            clusters += [cluster]

with open('bipartite_nodes-clusters.csv', 'r') as file:
    file.readline()
    for line in file:
        node, cluster = line.strip().split(',')
        nodes_clusters[nodes.index(node), clusters.index(cluster)] = 1

cluster_cluster = np.matmul(nodes_clusters.transpose(), nodes_clusters)

with open('./bipartite_cluster-cluster_projection.csv', 'w') as file:
    file.write('Source,Target,Weight\n')
    for idx in np.ndindex(cluster_cluster.shape):
        src, target = clusters[idx[0]], clusters[idx[1]]
        file.write(f'{src},{target},{cluster_cluster[idx]}\n')