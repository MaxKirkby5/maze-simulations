import networkx as nx
import matplotlib.pyplot as plt
import random
import itertools
import numpy as np
from scipy.spatial.distance import euclidean

# function to remove random edges from the 7x7 grid graph (removed edges threshold is 28 by default)
def random_maze(removed_edges=60):
    graph = nx.grid_2d_graph(7, 7)
    edges = list(graph.edges)
    num_edges_to_remove = random.randint(0, removed_edges)
    for _ in range(num_edges_to_remove):
        edge = random.choice(edges)
        if graph.has_edge(*edge):
            graph.remove_edge(*edge)
        else:
            continue
    return graph
    
# function for assessing outer edge counts 
def outer_edge_count(graph):
    outer_edge_count = 0
    for ((x1, y1), (x2, y2)) in graph.edges():
        # check if both nodes of the edge are on the perimeter
        if (x1 == 0 or x1 == 7 - 1 or y1 == 0 or y1 == 7 - 1) and \
           (x2 == 0 or x2 == 7 - 1 or y2 == 0 or y2 == 7 - 1):
            outer_edge_count += 1
    return outer_edge_count

# create coordinate system for inverse graph using cartesian product of two lists
def create_inverse_graph(graph):
    inverse_graph = nx.Graph()
    nodes = list(itertools.product(np.arange(0.5, 6.0, 1.0), repeat=2))
    inverse_graph.add_nodes_from(nodes)

    # run edge algorithm on inverse graph to search for loops
    for x1 in range(0, 6):
        for y1 in range(0, 6):
            node_G2 = (x1 + 0.5, y1 + 0.5)
            right_neighbor = (x1 + 1.5, y1 + 0.5)
            upper_neighbor = (x1 + 0.5, y1 + 1.5)
            
            if x1 < 5:  
                if not graph.has_edge((x1 + 1, y1), (x1 + 1, y1 + 1)):
                    inverse_graph.add_edge(node_G2, right_neighbor)
            
            if y1 < 5: 
                if not graph.has_edge((x1, y1 + 1), (x1 + 1, y1 + 1)):
                    inverse_graph.add_edge(node_G2, upper_neighbor)

    # plot graph and inverse graph (DISABLED)
    '''
    pos2 = {(x, y): (x, y) for x, y in inverse_graph.nodes()}
    pos = {node: node for node in graph.nodes()}
    merged_graph = nx.compose(graph, inverse_graph)
    combined_pos = pos.copy()
    combined_pos.update(pos2)

    colors_g = ['blue' for node in graph.nodes()]
    sizes_g = [300 for node in graph.nodes()]
    colors_g2 = ['red' for node in inverse_graph.nodes()]
    sizes_g2 = [100 for node in inverse_graph.nodes()]
    combined_colors = colors_g + colors_g2
    combined_sizes = sizes_g + sizes_g2

    nx.draw(merged_graph, combined_pos, with_labels=True, node_color=combined_colors, node_size=combined_sizes, font_size=10)
    plt.show()
    '''
    return inverse_graph


# ADD parameters for everything
# Super high level definitions
def checking_paths(graph, inverse_graph):
    open_clusters = []
    closed_clusters = []
    clusters = list(nx.connected_components(inverse_graph))
    # parameterise clusters by length
    filtered_clusters = sorted([cluster for cluster in clusters if len(cluster) > 2], key=len, reverse=True)
    
    for cluster in filtered_clusters:
        open_flag = False
        for (x1, y1) in cluster:
            if x1 == 0.5 and not graph.has_edge((x1 - 0.5, y1 + 0.5), (x1 - 0.5, y1 - 0.5)):
                open_flag = True
            if x1 == 5.5 and not graph.has_edge((x1 + 0.5, y1 + 0.5), (x1 + 0.5, y1 - 0.5)):
                open_flag = True
            if y1 == 5.5 and not graph.has_edge((x1 - 0.5, y1 + 0.5), (x1 + 0.5, y1 + 0.5)):
                open_flag = True
            if y1 == 0.5 and not graph.has_edge((x1 - 0.5, y1 - 0.5), (x1 + 0.5, y1 - 0.5)):
                open_flag = True
        if open_flag:
            open_clusters.append(cluster)
        else:
            closed_clusters.append(cluster)
    
    return open_clusters, closed_clusters

# check open components of graph
def open_component_check(graph):
    inverse = create_inverse_graph(graph)
    open_clusters, _ = checking_paths(graph, inverse)
    return not any(len(cluster) > 2 for cluster in open_clusters)

# check closed components of graph
def closed_component_check(graph):
    inverse = create_inverse_graph(graph)
    _, closed_clusters = checking_paths(graph, inverse)
    return not any(len(cluster) > 7 for cluster in closed_clusters)

# check if given maze passes criteria for a successful maze
def maze_criteria(graph, max_edges=56, max_outer_edges=19):
    if nx.is_connected(graph) and nx.number_of_edges(graph) > max_edges and outer_edge_count(graph) < max_outer_edges and open_component_check(graph) == True and closed_component_check(graph) == True:
        return True
    else :
        return False

# calculate maze fitness
def get_maze_fitness(graph):
    euclidean_distances = [euclidean(p1, p2) for p1 in graph.nodes() for p2 in graph.nodes() if p1 < p2]
    geodesic_distances = [nx.shortest_path_length(graph, source=p1, target=p2, weight=None)
                    for p1 in graph.nodes() for p2 in graph.nodes() if p1 < p2]
    correlation_coefficient = np.corrcoef(euclidean_distances, geodesic_distances)[0, 1]
    return 1 - abs(correlation_coefficient)

# generate 100 mazes that pass criteria
def generate_mazes(num_mazes=100):
    successful_mazes = []
    while len(successful_mazes) < num_mazes:
        maze = random_maze()
        inverse_maze = create_inverse_graph(maze)
        checking_paths(maze, inverse_maze)
        if maze_criteria(maze) == True:
            successful_mazes.append(maze)
        else:
            continue
    print("Finished generating mazes.")
    print(f"Good mazes: [{len(successful_mazes)}]")
    return successful_mazes


# randomise order of edges in maze when passing through


def best_maze(num_init_mazes=100):
    graph = nx.grid_2d_graph(7, 7)
    good_mazes = generate_mazes(num_init_mazes)
    base_maze = max(good_mazes, key=get_maze_fitness)
    best_maze_fitness = get_maze_fitness(base_maze)
    best_maze_config = base_maze.copy()  # copy best maze config
    print(f"Initial best maze fitness: {best_maze_fitness}")

    all_possible_edges = list(graph.edges)
    current_edges = set(base_maze.edges)

    for edge in all_possible_edges:
        if edge in current_edges:
            base_maze.remove_edge(*edge)
        else:
            base_maze.add_edge(*edge)

        if maze_criteria(base_maze):
            current_fitness = get_maze_fitness(base_maze)
            if current_fitness > best_maze_fitness:
                best_maze_fitness = current_fitness
                best_maze_config = base_maze.copy()  
        else:
            if edge in current_edges:
                base_maze.add_edge(*edge)
            else:
                base_maze.remove_edge(*edge)

    print("Finished optimizing maze.")
    print(f"Best maze fitness: {best_maze_fitness}")
    #pos = {node: node for node in graph.nodes()}
    #nx.draw(best_maze_config, pos, with_labels=False, font_size=10)
    #plt.show()

    return best_maze_config  # return best maze config 

    