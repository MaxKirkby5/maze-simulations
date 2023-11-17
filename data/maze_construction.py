import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle

# set parameters of generation and graph storage capacities
num_graphs = 100

def store_graph(graph, file_name) :
    with open(file_name, 'wb') as f:
        pickle.dump(graph, f)

def load_graph(file_name) :
    with open(file_name, 'rb') as f:
        return pickle.load(f)

# generate a two-dimensional grid graph (7x7)
G = nx.grid_2d_graph(7, 7)

# remove random edges from the 7x7 grid graph
def random_maze(G):
    edges_to_remove = [edge for edge in G.edges if random.random() < 0.5]
    for edge in edges_to_remove:
        G.remove_edge(edge[0], edge[1])
        
# generate defined number of random mazes
def generate_random_maze() :
    for i in range(num_graphs): 
        maze = random_maze(G)
        store_graph(maze, f'maze_{i}.pkl')
generate_random_maze()

# draw the mazes
nx.draw(G, with_labels=True, node_size=800, font_size=10)
plt.show()

