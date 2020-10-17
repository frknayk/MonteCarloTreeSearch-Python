#!/usr/bin/env python
# btree.py - http://www.graphviz.org/pdf/dotguide.pdf Figure 13

from graphviz import Digraph

g = Digraph('g')

def make_graph_layer(root_name,layer_num,num_nodes):
    for node_num in range(num_nodes):
        node_name = 'layer_' + str(layer_num) + '_' + str(node_num)
        g.edge(root_name, node_name)


# Root node
root_name = 'root'
g.node(root_name)


make_graph_layer(root_name=root_name,
    layer_num=0,num_nodes=4)

make_graph_layer(root_name='layer_0_1',
    layer_num=1,num_nodes=4)

make_graph_layer(root_name='layer_1_3',
    layer_num=2,num_nodes=4)


g.view(cleanup=True)