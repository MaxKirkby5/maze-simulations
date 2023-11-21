import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle
import math
import numpy as np
import itertools

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

def create_G2():
 for i in m_nodes:
    for j in n_nodes:
        G2.add_node((i, j))
create_G2()

def conditional_G2_graph(G, G2):
    for x1 in range(0, 6):
        for y1 in range(0, 6):
            node_G2 = (x1 + 0.5, y1 + 0.5)
            right_neighbor = (x1 + 1.5, y1 + 0.5)
            upper_neighbor = (x1 + 0.5, y1 + 1.5)
            
            # Corresponding nodes in G that would intersect with an edge in G2
            if x1 < 5:  # Check to the right, but not for the last column
                if not G.has_edge((x1 + 1, y1), (x1 + 1, y1 + 1)):
                    G2.add_edge(node_G2, right_neighbor)
            
            if y1 < 5:  # Check above, but not for the top row
                if not G.has_edge((x1, y1 + 1), (x1 + 1, y1 + 1)):
                    G2.add_edge(node_G2, upper_neighbor)

# Call the function with the two graphs
conditional_G2_graph(G, G2)

open_clusters = []
closed_clusters = []

def checking_paths(graph):
    clusters = list(nx.connected_components(G2))
    filtered_clusters = sorted([cluster for cluster in clusters if len(cluster) > 2], key=len, reverse=True)
    for cluster in filtered_clusters:
        for (x1, y1) in cluster:
            if x1 == 0.5 :
                if not G.has_edge((x1 - 0.5, y1+0.5), (x1-0.5, y1-0.5)):
                    open_clusters.append(cluster)
            if x1 == 5.5 :
                if not G.has_edge((x1 + 0.5, y1+0.5), (x1+0.5, y1-0.5)):
                    open_clusters.append(cluster)
            if y1 == 5.5 :
                if not G.has_edge((x1 - 0.5, y1+0.5), (x1+0.5, y1+0.5)):
                    open_clusters.append(cluster)
            if y1 == 0.5 :
                if not G.has_edge((x1 - 0.5, y1-0.5), (x1+0.5, y1-0.5)):
                    open_clusters.append(cluster)
    for element in filtered_clusters :
        if element not in open_clusters :
            closed_clusters.append(element)
    return open_clusters, closed_clusters

def open_component_check(graph) :
    checking_paths(graph)
    if any(len(cluster) > 2 for cluster in open_clusters) : 
        return False
    else : 
        return True
print(open_component_check(G))

def closed_component_check(graph) :
    checking_paths(graph)
    if any(len(cluster) > 7 for cluster in closed_clusters) : 
        return False
    else : 
        return True
print(closed_component_check(G))


def maze_criteria(graph):
    if nx.is_connected(graph) and nx.number_of_edges(graph) < 46 and outer_edge_count(graph) < 19 and open_component_check(graph) == True and closed_component_check(graph) == True:
        return True
    else :
        return False

#def maze_criteria(G): 
 #   if nx.is_connected(G) and nx.number_of_edges(G) < 46 and outer_edge_count(G) < 19:
  #      return True











pos2 = {(x, y): (x, y) for x, y in G2.nodes()}

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

