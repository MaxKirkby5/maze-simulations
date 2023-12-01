import networkx as nx
import matplotlib.pyplot as plt
import random
import itertools
import numpy as np
from scipy.spatial.distance import euclidean
from GridMaze.maze import maze_plotting as mp

def plot_maze(graph):
    '''Function that plots a maze on cartesian coordinates.'''
    pos = {node: node for node in graph.nodes()}
    nx.draw(graph, pos, with_labels=True, font_size=10, edge_color='black')

    unidirectional_edges = [(u, v) for u, v in graph.edges() if not graph.has_edge(v, u)]
    nx.draw_networkx_edges(graph, pos, edgelist=unidirectional_edges, edge_color='pink', arrows=True)
    plt.show()

def check_outer_edge_count(graph):
    '''Function that checks the number of edges on the perimeter of the maze. 
    Method: Checks if both nodes of the edge are on the perimeter.'''
    outer_edge_count = 0
    for ((x1, y1), (x2, y2)) in graph.edges():
        if (x1 == 0 or x1 == 7 - 1 or y1 == 0 or y1 == 7 - 1) and \
           (x2 == 0 or x2 == 7 - 1 or y2 == 0 or y2 == 7 - 1):
            outer_edge_count += 1
    return outer_edge_count

def has_1x1_square(graph):
    '''Function that checks if there is any 1x1 square in the graph.'''
    for x in range(6):  
        for y in range(6):
            if all([graph.has_edge((x, y), (x + 1, y)),
                    graph.has_edge((x + 1, y), (x + 1, y + 1)),
                    graph.has_edge((x + 1, y + 1), (x, y + 1)),
                    graph.has_edge((x, y + 1), (x, y))]):
                return True
    return False

def generate_minimally_connected_grid(width=7, height=7):
    '''Function that generates a minimally connected grid graph with a random spanning tree'''
    grid_graph = nx.grid_2d_graph(width, height)
    spanning_tree = nx.tree.random_spanning_tree(grid_graph)
    edges = list(grid_graph.edges)

    for edge in random.sample(edges, len(edges)):
        spanning_tree.add_edge(*edge)
        if has_1x1_square(spanning_tree):
            spanning_tree.remove_edge(*edge)
    return spanning_tree

def check_open_and_closed_components(graph, 
                                             grid_size=7, 
                                             cluster_length=2,
                                             open_components_allowed=2,
                                             closed_components_allowed=7):
    '''Create inverse graph'''
    inverse_graph = nx.Graph()
    nodes = list(itertools.product(np.arange(0.5, 6.0, 1.0), repeat=2))
    inverse_graph.add_nodes_from(nodes)
    for x1 in range(0, grid_size-1):
        for y1 in range(0, grid_size-1):
            node_G2 = (x1 + 0.5, y1 + 0.5)
            right_neighbor = (x1 + 1.5, y1 + 0.5)
            upper_neighbor = (x1 + 0.5, y1 + 1.5)
            
            if x1 < 5:  
                if not graph.has_edge((x1 + 1, y1), (x1 + 1, y1 + 1)):
                    inverse_graph.add_edge(node_G2, right_neighbor)
            
            if y1 < 5:  
                if not graph.has_edge((x1, y1 + 1), (x1 + 1, y1 + 1)):
                    inverse_graph.add_edge(node_G2, upper_neighbor)
    '''Search for open and closed components in original graph'''
    open_clusters = []
    closed_clusters = []
    clusters = list(nx.connected_components(inverse_graph))
    filtered_clusters = sorted([cluster for cluster in clusters if len(cluster) > cluster_length], key=len, reverse=True)
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

    if any(len(cluster) > open_components_allowed for cluster in open_clusters) or any(len(cluster) > closed_components_allowed for cluster in closed_clusters):
        return False
    else:
        return True

def check_if_successful_maze(graph, max_outer_edges=19):
    '''Function that checks if the maze passes the criteria for a successful maze.'''
    if check_open_and_closed_components(graph) == True and check_outer_edge_count(graph) < max_outer_edges and nx.is_connected(graph):
        return True
    else :
        return False

def calculate_maze_fitness(graph):
    '''Function that calculates the fitness of a maze by summating the euclidean distance and the geodesic distance between all points in the maze.'''
    euclidean_distances = [euclidean(p1, p2) for p1 in graph.nodes() for p2 in graph.nodes() if p1 < p2]
    geodesic_distances = [nx.shortest_path_length(graph, source=p1, target=p2, weight=None)
                    for p1 in graph.nodes() for p2 in graph.nodes() if p1 < p2]
    correlation_coefficient = np.corrcoef(euclidean_distances, geodesic_distances)[0, 1]
    return 1 - abs(correlation_coefficient)

def generate_list_of_successful_mazes(num_mazes=100):
    '''Function that generates n successful mazes and stores them.'''
    successful_mazes = []
    unsuccessful_mazes = []
    while len(successful_mazes) < num_mazes:
        maze = generate_minimally_connected_grid()
        if check_if_successful_maze(maze) == True:
            successful_mazes.append(maze)
        else:
            unsuccessful_mazes.append(maze)
            continue
    print("Finished generating mazes.")
    return successful_mazes

def optimise_the_maze():
    generated_mazes = generate_list_of_successful_mazes(100)
    best_maze = max(generated_mazes, key=calculate_maze_fitness)
    best_fitness = calculate_maze_fitness(best_maze)

    converged = False
    while not converged:
        variant_mazes = []
        all_edges = list(nx.grid_2d_graph(7, 7).edges())

        for edge in all_edges:
            updating_maze = best_maze.copy()
            if best_maze.has_edge(*edge):
                updating_maze.remove_edge(*edge)
            else:
                updating_maze.add_edge(*edge)
            
            if check_if_successful_maze(updating_maze) and not has_1x1_square(updating_maze):
                variant_mazes.append(updating_maze)

        new_best_maze = max(variant_mazes, key=calculate_maze_fitness, default=best_maze)
        new_best_fitness = calculate_maze_fitness(new_best_maze)

        if new_best_fitness > best_fitness:
            best_maze = new_best_maze
            best_fitness = new_best_fitness
            print(best_fitness)
        else:
            converged = True

    print(f"Optimized maze fitness: {best_fitness}")
    plot_maze(best_maze)
    return best_maze



