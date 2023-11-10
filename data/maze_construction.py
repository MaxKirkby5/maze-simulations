import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle

# generate a 7x7 grid graph with randomly connected edges
G = nx.Graph()
grid_size = 7
generated_graphs = 100

def generate_random_maze(size) :
    for i in range(grid_size):
        for j in range (grid_size):
            G.add_node((i, j))

    for node in G.nodes():
        num_edges = random.randint(0,4)

        potential_neighbors = [
            (node[0] - 1, node[1]), # left
            (node[0] + 1, node[1]), # right
            (node[0], node[1] - 1), # down
            (node[0], node[1] + 1)  # up
        ]

        valid_neighbors = [nbr for nbr in potential_neighbors if nbr in G.nodes() and not G.has_edge(node, nbr)]

        random.shuffle(valid_neighbors)

        for nbr in valid_neighbors[:num_edges]:
            if len(G.edges(nbr)) < 4:
                G.add_edge(node, nbr)

def store_graph(graph, file_name) :
    with open(file_name, 'wb') as f:
        pickle.dump(graph, f)

def load_graph(file_name) :
    with open(file_name, 'rb') as f:
        return pickle.load(f)

for i in range(generated_graphs): 
    maze = generate_random_maze(grid_size)
    store_graph(maze, f'maze_{i}.pkl')

pos = dict((n, n) for n in G.nodes())

nx.draw(G, pos, with_labels=True, node_size=800, font_size=10)
plt.show()
