from graph import DiGraph
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

g = nx.bfs_tree(nx.balanced_tree(2, 2), 0)

nx.relabel_nodes(g,
                 {i: 'h%d' % i for i in range(len(g.nodes))},
                 copy=False
                 )

h_nodes_list = g.nodes()
b_nodes_list = ['b%d' % i for i in range(len(h_nodes_list))]
bh_edge_list = [(a, b) for a, b in zip(b_nodes_list, h_nodes_list)]

g.add_edges_from(bh_edge_list)

pos = graphviz_layout(g, prog='dot')
nx.draw(g, pos, with_labels=True)
