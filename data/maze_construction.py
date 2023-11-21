import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle
import math
import numpy as np

# generate a two-dimensional grid graph (7x7)
G = nx.grid_2d_graph(7, 7)

# set parameters of graph generation and grid size/position
num_graphs = 100
grid_size = 7
pos = dict( (n, n) for n in G.nodes() )

# function to store and load graphs
def store_graph(graph, file_name) :
    with open(file_name, 'wb') as f:
        pickle.dump(graph, f)

def load_graph(file_name) :
    with open(file_name, 'rb') as f:
        return pickle.load(f)

# function to remove random edges from the 7x7 grid graph + maintain connectivity
def random_maze(G):
    edges_removed = 0
    limit = 24 # come back to = can we use minimum of n-1 edges (to maintain connectivity)?
    while edges_removed < limit:
        node = random.choice(list(G.nodes))
        get_neighbors = list(G.neighbors(node))
        if get_neighbors:  
            rand_neighbor = random.choice(get_neighbors)
            if G.has_edge(node, rand_neighbor):
                G.remove_edge(node, rand_neighbor)
                if nx.is_connected(G):
                    edges_removed += 1
                else:
                    G.add_edge(node, rand_neighbor)
random_maze(G)

# function for assessing outer edge counts 
def outer_edge_count(G):
    outer_edge_count = 0
    for ((x1, y1), (x2, y2)) in G.edges():
        # Check if both nodes of the edge are on the perimeter
        if (x1 == 0 or x1 == grid_size - 1 or y1 == 0 or y1 == grid_size - 1) and \
           (x2 == 0 or x2 == grid_size - 1 or y2 == 0 or y2 == grid_size - 1):
            outer_edge_count += 1
    return outer_edge_count

print(outer_edge_count(G))

# create coordinate system for overlayed graph 
def merge(list1, list2):
    merged_list = []
    for i in list1 :
        for j in list2:
            merged_list.append((i, j))
    return merged_list

m_nodes = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
n_nodes = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
merged_nodes = merge(m_nodes, n_nodes)

G2 = nx.Graph()

# create conditional graph creation function
def conditional_G2_graph() :
    for (x1, y1) in G.nodes():
        get_neigh = list(G.neighbors((x1, y1)))
        for neigh_node in get_neigh:
            if G.has_edge((x1, y1), neigh_node):
                G2.add_edge((min(m_nodes, key=lambda x:abs(x-x1)), min(n_nodes, key=lambda x:abs(x-y1))), (min(m_nodes, key=lambda x:abs(x-neigh_node[0])), min(n_nodes, key=lambda x:abs(x-neigh_node[1]))))
conditional_G2_graph()

#G2.add_nodes_from(nodes_for_adding=merged_nodes)
pos2 = {(x, y): (x, y) for x, y in merged_nodes}

# merge the two graphs
pos = {node: node for node in G.nodes()}
merged_graph = nx.compose(G, G2)

# Combine positions for the merged graph
combined_pos = pos.copy()
combined_pos.update(pos2)

# Define different colours and sizes for the two graphs
colors_g = ['blue' for node in G.nodes()]
sizes_g = [300 for node in G.nodes()]
colors_g2 = ['red' for node in G2.nodes()]
sizes_g2 = [100 for node in G2.nodes()]
combined_colors = colors_g + colors_g2
combined_sizes = sizes_g + sizes_g2

# Draw the merged graph
nx.draw(merged_graph, combined_pos, with_labels=True, node_color=combined_colors, node_size=combined_sizes, font_size=10)
plt.show()








#closed_component_count(G)

# function for assessing open connected component
def open_component_count(G) :
    print(list(nx.find_cliques(G)))

# get euclidean distance
def euc_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

euc_dist = [euc_distance(p1, p2) for p1 in G.nodes() for p2 in G.nodes() if p1 < p2]

# get shortest path distance
sp_dist = [nx.shortest_path_length(G, source=p1, target=p2, weight=None) 
    for p1 in G.nodes() for p2 in G.nodes() if p1 < p2]

# get correlation coefficient
corr_coefficient = np.corrcoef(euc_dist, sp_dist)[0, 1]

# calculate maze fitness
maze_fitness = 1 - abs(corr_coefficient)
print(maze_fitness)

#def maze_criteria(G): 
 #   if nx.is_connected(G) and nx.number_of_edges(G) < 46 and outer_edge_count(G) < 19:
  #      return True

# generate defined number of random mazes
#def generate_random_maze() :
 #   for i in range(num_graphs): 
  #      maze = random_maze(G)
   #     store_graph(maze, f'maze_{i}.pkl')
#generate_random_maze()

# draw the mazes
nx.draw(G, pos, with_labels=True, node_size=800, font_size=10)
plt.show()

