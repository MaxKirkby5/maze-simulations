import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle

# generate a two-dimensional grid graph (7x7)
G = nx.grid_2d_graph(7, 7)

# set parameters of generation, position and graph storage capacities
num_graphs = 100
grid_size = 7
pos = dict( (n, n) for n in G.nodes() )

def store_graph(graph, file_name) :
    with open(file_name, 'wb') as f:
        pickle.dump(graph, f)

def load_graph(file_name) :
    with open(file_name, 'rb') as f:
        return pickle.load(f)

# function to remove random edges from the 7x7 grid graph and maintain connectivity
def random_maze(G):
    edges_removed = 0
    limit = 24 # minimum of n-1 edges to maintain connectivity
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

# function for assessing closed connected component
def closed_component_count(G) :
    print(list(nx.find_cliques(G)))

closed_component_count(G)




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

