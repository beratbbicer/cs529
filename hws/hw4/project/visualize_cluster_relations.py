import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt


if __name__ == '__main__':
    graph = nx.read_gexf('../LesMiserables.gexf', node_type=float)
    gn_communities = list(list(nx.community.girvan_newman(graph))[4])
    lou_communities = nx.community.louvain_communities(graph)
    
    orig_nodes, all_clusters = list(graph.nodes), [f'gn{i}' for i in range(len(gn_communities))] + [f'lou{i}' for i in range(len(lou_communities))]

    bipartite_graph = nx.Graph()
    bipartite_graph.add_nodes_from(graph, bipartite=0)
    bipartite_graph.add_nodes_from([f'gn{i}' for i in range(len(gn_communities))] + [f'lou{i}' for i in range(len(lou_communities))], bipartite=1)

    for community in range(len(gn_communities)):
        for node in list(gn_communities[community]):
            bipartite_graph.add_edge(node, f'gn{community}')

    for community in range(len(lou_communities)):
        for node in list(lou_communities[community]):
            bipartite_graph.add_edge(node, f'lou{community}')

    biadjmat = bipartite.biadjacency_matrix(bipartite_graph, row_order=orig_nodes, column_order=all_clusters)
    nx.write_edgelist(bipartite_graph, 'bipartite_nodes-clusters.csv', delimiter=',', data=False)

    cluster_relations = bipartite.weighted_projected_graph(bipartite_graph, all_clusters)
    edge_labels=dict([((u,v,),d['weight']) for u,v,d in cluster_relations.edges(data=True)])
    node_labels = {node:node for node in cluster_relations.nodes()}
    
    plt.figure(figsize=(8,8))
    prog = 'neato'
    pos=nx.nx_agraph.graphviz_layout(cluster_relations, prog=prog)
    nx.draw_networkx_labels(cluster_relations, pos, labels=node_labels)
    nx.draw_networkx_edge_labels(cluster_relations, pos, edge_labels=edge_labels)
    nx.draw(cluster_relations, pos, node_size=1500, edge_cmap=plt.cm.Reds)
    plt.savefig(f'cluster_relations-{prog}.png', bbox_inches='tight')
    _= 1