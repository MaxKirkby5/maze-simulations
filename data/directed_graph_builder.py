import networkx as nx
import matplotlib.pyplot as plt
import random
import itertools
import numpy as np
from scipy.spatial.distance import euclidean
from GridMaze.maze import builder as mb
from GridMaze.maze import maze_plotting as mp

def count_number_of_one_way_edges(maze):
    unidirectional_edges = [(u, v) for u, v in maze.edges() if not maze.has_edge(v, u)]
    return len(unidirectional_edges)

def count_number_of_outer_edges(maze):
    '''Function that checks the number of edges on the perimeter of the maze for a directed graph. 
    Method: Checks if both nodes of the edge are on the perimeter and ensures each edge is counted only once.'''
    outer_edge_count = 0
    counted_edges = set()

    for ((x1, y1), (x2, y2)) in maze.edges():
        if (x1 == 0 or x1 == 7 - 1 or y1 == 0 or y1 == 7 - 1) and \
           (x2 == 0 or x2 == 7 - 1 or y2 == 0 or y2 == 7 - 1):
            edge = ((x1, y1), (x2, y2))
            reverse_edge = ((x2, y2), (x1, y1))

            if edge not in counted_edges and reverse_edge not in counted_edges:
                outer_edge_count += 1
                counted_edges.add(edge)
                counted_edges.add(reverse_edge)

    return outer_edge_count

def generate_random_directed_maze(one_way_edges=5, edge_variations=20):
    '''Function that removes random edges from a 7x7 grid graph and returns a directed graph.
    Adds defined goals to defined nodes. 
    '''
    base_graph = mb.generate_minimally_connected_grid() 
    directed_graph = nx.MultiDiGraph(incoming_graph_data=base_graph)
    directed_edges = list(directed_graph.edges())

    for _ in range(edge_variations):
        while count_number_of_one_way_edges(directed_graph) < one_way_edges:
            edge = random.choice(directed_edges)
            if directed_graph.has_edge(*edge):
                directed_graph.remove_edge(*edge)
                if nx.is_strongly_connected(directed_graph) and not mb.has_1x1_square(directed_graph):
                    continue
                else:
                    directed_graph.add_edge(*edge)
            else :
                directed_graph.add_edge(*edge)
                if nx.is_strongly_connected(directed_graph) and not mb.has_1x1_square(directed_graph):
                    continue
                else:
                    directed_graph.remove_edge(*edge)
    return directed_graph

def check_open_and_closed_components(graph, grid_size=7, cluster_length=2, open_components=2, closed_components_max=7):
    inverse_graph = nx.Graph()
    nodes = list(itertools.product(np.arange(0.5, 6.0, 1.0), repeat=2))
    inverse_graph.add_nodes_from(nodes)

    for x1 in range(0, grid_size-1):
        for y1 in range(0, grid_size-1):
            node_G2 = (x1 + 0.5, y1 + 0.5)
            right_neighbor = (x1 + 1.5, y1 + 0.5)
            upper_neighbor = (x1 + 0.5, y1 + 1.5)
            
            if x1 < 5:  
                if not graph.has_edge((x1 + 1, y1), (x1 + 1, y1 + 1)) and not graph.has_edge((x1 + 1, y1 + 1), (x1 + 1, y1)):
                    inverse_graph.add_edge(node_G2, right_neighbor)
            
            if y1 < 5:  
                if not graph.has_edge((x1, y1 + 1), (x1 + 1, y1 + 1)) and not graph.has_edge((x1 + 1, y1 + 1), (x1, y1 + 1)):
                    inverse_graph.add_edge(node_G2, upper_neighbor)
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
    return not any(len(cluster) > open_components for cluster in open_clusters) and not any(len(cluster) > closed_components_max for cluster in closed_clusters)

def is_directed_maze_successful(graph, max_outer_edge_count=19):
    if check_open_and_closed_components(graph) and count_number_of_outer_edges(graph) < max_outer_edge_count and nx.is_strongly_connected(graph):
        return True
    else:
        return False

def calculate_maze_fitness(graph):
    '''Function that calculates the fitness of a maze by summating the euclidean distance and the geodesic distance between all points in the maze.'''
    euclidean_distances = [euclidean(p1, p2) for p1 in graph.nodes() for p2 in graph.nodes() if p1 < p2]
    geodesic_distances = [nx.shortest_path_length(graph, source=p1, target=p2, weight="weight")
                    for p1 in graph.nodes() for p2 in graph.nodes() if p1 < p2]
    correlation_coefficient = np.corrcoef(euclidean_distances, geodesic_distances)[0, 1]
    return 1 - abs(correlation_coefficient)

def generate_list_of_successful_mazes(num_mazes=10):
    '''Function that generates n successful mazes and stores them.'''
    successful_mazes = []
    unsuccessful_mazes = []
    while len(successful_mazes) < num_mazes:
        maze = generate_random_directed_maze()
        if is_directed_maze_successful(maze) == True:
            successful_mazes.append(maze)
            print(f"Number of one-way edges: {count_number_of_one_way_edges(maze)}")
        else:
            unsuccessful_mazes.append(maze)
            continue
    print("Finished generating mazes.")
    mp.plot_maze(successful_mazes[0])
    return successful_mazes

def optimise_the_maze(number_of_one_way_edges=5):
    generated_mazes = generate_list_of_successful_mazes(100)
    best_maze = max(generated_mazes, key=calculate_maze_fitness)
    best_fitness = calculate_maze_fitness(best_maze)

    converged = False
    while not converged:
        variant_mazes = []
        base_graph = nx.grid_2d_graph(7, 7)
        base_directed_graph = nx.MultiDiGraph(incoming_graph_data=base_graph)
        all_edges = list(base_directed_graph.edges())

        for edge in all_edges:
            updating_maze = best_maze.copy()
            if best_maze.has_edge(*edge):
                updating_maze.remove_edge(*edge)
            else:
                updating_maze.add_edge(*edge)
            
            if is_directed_maze_successful(updating_maze) and count_number_of_one_way_edges(updating_maze) < number_of_one_way_edges and not mb.has_1x1_square(updating_maze):
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
    mb.plot_maze(best_maze)
    return best_maze


def optimise_the_maze_v2(number_of_one_way_edges=5):
    generated_mazes = generate_list_of_successful_mazes(100)
    best_maze = max(generated_mazes, key=calculate_maze_fitness)
    best_fitness = calculate_maze_fitness(best_maze)

    converged = False
    while not converged:
        variant_mazes = []
        all_edges = list(best_maze.edges())  

        for edge in all_edges:
            updating_maze = best_maze.copy()
            if updating_maze.has_edge(*edge):
                updating_maze.remove_edge(*edge)
                reverse_edge = (edge[1], edge[0])
                if not updating_maze.has_edge(*reverse_edge):
                    updating_maze.add_edge(*reverse_edge)
            else:
                updating_maze.add_edge(*edge)
                reverse_edge = (edge[1], edge[0])
                updating_maze.add_edge(*reverse_edge)
            
            if is_directed_maze_successful(updating_maze) and count_number_of_one_way_edges(updating_maze) <= number_of_one_way_edges and not mb.has_1x1_square(updating_maze):
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
    mb.plot_maze(best_maze)
    return best_maze

