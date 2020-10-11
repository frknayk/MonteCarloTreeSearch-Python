def visualize_tree(node):
    if node is None:
        node = []

    g = Digraph('g', filename='btree.gv')

    # Root node
    root_name = 'root'
    g.node(root_name)

    if node.children is None:
        return g.view(cleanup=True)

    for child_idx, child in enumerate(node.children):
        num_children = get_num_children(node = child)
        print(child_idx,num_children)

def vis(root):
    while node.get_num_children() != 0:
        pass

def get_num_children(node):
    num_of_child = 0
    if node.children is not None:
        num_of_child = node.children.__len__()
    return num_of_child

def make_graph_layer(g,root_name,layer_num,num_nodes):
    for node_num in range(num_nodes):
        node_name = 'layer_' + str(layer_num) + '_' + str(node_num)
        g.edge(root_name, node_name)

# visualize_tree(node=mcts)