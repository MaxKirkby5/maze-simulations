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
    """
    Function that checks the number of edges on the perimeter of the maze. 
    Method: Checks if both nodes of the edge are on the perimeter.
    """
    outer_edge_count = 0
    for ((x1, y1), (x2, y2)) in graph.edges():
        if (x1 == 0 or x1 == 7 - 1 or y1 == 0 or y1 == 7 - 1) and \
           (x2 == 0 or x2 == 7 - 1 or y2 == 0 or y2 == 7 - 1):
            outer_edge_count += 1
    return outer_edge_count



def generate_random_maze_v2(removed_edges=36):
    """
    Function that removes random edges from a 7x7 grid graph to prevent 1x1 squares.
    """
    graph = nx.grid_2d_graph(7, 7)
    edges = list(graph.edges)
    num_edges_to_remove = random.randint(25, removed_edges)
    print(num_edges_to_remove)

    for _ in range(num_edges_to_remove):
        edge = random.choice(edges)
        if graph.has_edge(*edge):
            graph.remove_edge(*edge)
            if has_1x1_square(graph) or not nx.is_connected(graph):
                graph.add_edge(*edge)
        else:
            continue
    return graph

def generate_minimally_connected_grid(width=7, height=7):
    # Create a 7x7 grid graph
    grid_graph = nx.grid_2d_graph(width, height)
    spanning_tree = nx.tree.random_spanning_tree(grid_graph)
    edges = list(grid_graph.edges)

    while not has_1x1_square(spanning_tree):
        edge = random.choice(edges)
        if not spanning_tree.has_edge(*edge):
            spanning_tree.add_edge(*edge)
        else:
            continue
    print(len(list(spanning_tree.edges)))
    return spanning_tree

def has_1x1_square(graph):
    """
    Check if there is any 1x1 square in the graph.
    """
    for x in range(6):  # Since it's a 7x7 grid, we check up to 6
        for y in range(6):
            if all([graph.has_edge((x, y), (x + 1, y)),
                    graph.has_edge((x + 1, y), (x + 1, y + 1)),
                    graph.has_edge((x + 1, y + 1), (x, y + 1)),
                    graph.has_edge((x, y + 1), (x, y))]):
                return True
    return False

def generate_random_maze(removed_edges=36):
    """
    Function that removes random edges from a 7x7 grid graph. 
    Method: The number of edges to remove is randomly chosen between 0 and the removed_edges parameter (set at 36, the maximum 
    number of edges that can be removed while still maintaining a connected graph).
    """
    graph = nx.grid_2d_graph(7, 7)
    edges = list(graph.edges)
    num_edges_to_remove = random.randint(0, removed_edges)
    for _ in range(num_edges_to_remove):
        edge = random.choice(edges)
        if graph.has_edge(*edge):
            graph.remove_edge(*edge)
            if has_1x1_square(graph):
                graph.add_edge(*edge)
        else:
            continue
    return graph


def check_outer_edge_count(graph):
    """
    Function that checks the number of edges on the perimeter of the maze. 
    Method: Checks if both nodes of the edge are on the perimeter.
    """
    outer_edge_count = 0
    for ((x1, y1), (x2, y2)) in graph.edges():
        if (x1 == 0 or x1 == 7 - 1 or y1 == 0 or y1 == 7 - 1) and \
           (x2 == 0 or x2 == 7 - 1 or y2 == 0 or y2 == 7 - 1):
            outer_edge_count += 1
    return outer_edge_count

def check_squares_open_and_closed_components(graph, 
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
    '''Search for squares in original graph'''
    open_clusters = []
    closed_clusters = []
    squares = 0
    clusters = list(nx.connected_components(inverse_graph))
    inverse_graph_nodes = list(inverse_graph.nodes())
    nodes_not_in_connected_components = [node for node in inverse_graph_nodes if node not in clusters]
    for (x,y) in nodes_not_in_connected_components:
        if graph.has_edge((x-0.5, y+0.5), (x+0.5, y+0.5)) \
            and graph.has_edge((x+0.5, y-0.5), (x-0.5, y-0.5))\
            and graph.has_edge((x-0.5, y+0.5), (x-0.5, y-0.5)) \
            and graph.has_edge((x+0.5, y+0.5), (x+0.5, y-0.5)):
            squares += 1
    
    '''Search for open and closed components in original graph'''
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

    if any(len(cluster) > open_components_allowed for cluster in open_clusters) or any(len(cluster) > closed_components_allowed for cluster in closed_clusters) or squares > 0:
        return False
    else:
        return True

def check_if_successful_maze(graph, max_edges=48, max_outer_edges=19):
    '''Function that checks if the maze passes the criteria for a successful maze.'''
    if check_squares_open_and_closed_components(graph) == True:
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

def generate_list_of_successful_mazes(num_mazes=1):
    '''Function that generates n successful mazes and stores them.'''
    successful_mazes = []
    unsuccessful_mazes = []
    while len(successful_mazes) < num_mazes:
        maze = generate_minimally_connected_grid()
        if check_if_successful_maze(maze) == True:
            print("Found successful maze")
            successful_mazes.append(maze)
        else:
            unsuccessful_mazes.append(maze)
            print("Found unsuccessful maze" + str(len(unsuccessful_mazes)))
            continue
    print("Finished generating mazes.")
    plot_maze(successful_mazes[0])
    return successful_mazes

def generate_optimal_maze(num_init_mazes=100):
    ''' Function that generates an optimal maze by removing and adding edges to maze with the highest fitness.'''
    graph = nx.grid_2d_graph(7, 7)
    good_mazes = generate_list_of_successful_mazes(num_init_mazes)
    base_maze = max(good_mazes, key=calculate_maze_fitness)
    best_maze_fitness = calculate_maze_fitness(base_maze)
    best_maze_config = base_maze.copy() 
    print(f"Initial best maze fitness: {best_maze_fitness}")

    all_possible_edges = list(graph.edges)
    current_edges = set(base_maze.edges)

    for edge in all_possible_edges:
        if edge in current_edges:
            base_maze.remove_edge(*edge)
        else:
            base_maze.add_edge(*edge)

        if check_if_successful_maze(base_maze):
            current_fitness = calculate_maze_fitness(base_maze)
            if current_fitness > best_maze_fitness:
                best_maze_fitness = current_fitness
                best_maze_config = base_maze.copy()  
        else:
            if edge in current_edges:
                base_maze.add_edge(*edge)
            else:
                base_maze.remove_edge(*edge)

    print(f"Best maze fitness: {best_maze_fitness}")
    plot_maze(best_maze_config)

    return best_maze_config 

def generate_optimised_directed_maze_v2(num_init_mazes=10, batch_size=5, num_iterations=10000):
    base_graph = nx.grid_2d_graph(7, 7)

    best_maze = max(generate_list_of_successful_mazes(num_init_mazes), key=calculate_maze_fitness)
    best_maze_fitness = calculate_maze_fitness(best_maze)
    configuration_of_best_maze = best_maze.copy()
    print(f"Initial best maze fitness is: {best_maze_fitness}")

    fitness_over_time = []
    all_possible_edges = list(base_graph.edges)

    for _ in range(num_iterations):
        current_edges = set(best_maze.edges)
        modified_edges = random.sample(all_possible_edges, batch_size)

        # Apply batch modifications
        for edge in modified_edges:
            if edge in current_edges:
                best_maze.remove_edge(*edge)
            else:
                best_maze.add_edge(*edge)

        # Check if the new configuration is successful and if it improves the fitness
        if check_if_successful_maze(best_maze):
            current_fitness = calculate_maze_fitness(best_maze)
            if current_fitness > best_maze_fitness:
                best_maze_fitness = current_fitness
                fitness_over_time.append(best_maze_fitness)
                configuration_of_best_maze = best_maze.copy()
            else:
                # Revert the changes if there's no improvement
                for edge in modified_edges:
                    if edge in current_edges:
                        best_maze.add_edge(*edge)
                    else:
                        best_maze.remove_edge(*edge)
        else:
            # Revert the changes if the maze is not successful
            for edge in modified_edges:
                if edge in current_edges:
                    best_maze.add_edge(*edge)
                else:
                    best_maze.remove_edge(*edge)

    print(f"Best maze fitness: {best_maze_fitness}")
    plt.plot(fitness_over_time)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Fitness of Maze Over Iterations')
    plt.show()
    mp.plot_maze(configuration_of_best_maze)



def acceptance_probability(old_fitness, new_fitness, temperature):
    if new_fitness > old_fitness:
        return 1.0
    return math.exp((new_fitness - old_fitness) / temperature)

def hybrid_optimization(num_iterations=10000, tabu_size=5, initial_temp=90, cooling_rate=0.5, num_init_mazes=100):
    best_maze = max(generate_list_of_successful_mazes(num_init_mazes), key=calculate_maze_fitness)
    best_maze_fitness = calculate_maze_fitness(best_maze)
    print(f"Initial best maze fitness is: {best_maze_fitness}")
    tabu_list = collections.deque(maxlen=tabu_size)
    temperature = initial_temp

    all_possible_edges = list(nx.grid_2d_graph(7, 7).edges())

    for _ in range(num_iterations):
        current_edges = set(best_maze.edges)
        edge = random.choice(all_possible_edges)

        if edge not in tabu_list:
            modified_maze = best_maze.copy()
            if edge in current_edges:
                modified_maze.remove_edge(*edge)
            else:
                modified_maze.add_edge(*edge)

            if check_if_successful_maze(modified_maze) and not has_1x1_square(modified_maze):
                current_fitness = calculate_maze_fitness(modified_maze)
                if acceptance_probability(best_maze_fitness, current_fitness, temperature) > random.random():
                    best_maze = modified_maze
                    best_maze_fitness = current_fitness
                    tabu_list.append(edge)

        temperature *= cooling_rate

    print(f"Best maze fitness: {best_maze_fitness}")
    plot_maze(best_maze)
    return best_maze


def select_parents(population, k=3):
    """Selects two parents from the population using tournament selection."""
    best1 = max(random.sample(population, k), key=calculate_maze_fitness)
    best2 = max(random.sample(population, k), key=calculate_maze_fitness)
    return best1, best2

def crossover(parent1, parent2):
    """Performs single-point crossover on the edge list."""
    child = nx.Graph()
    child.add_nodes_from(parent1.nodes())

    # Combine edges from both parents
    edges1 = list(parent1.edges())
    edges2 = list(parent2.edges())
    crossover_point = random.randint(0, min(len(edges1), len(edges2)))
    child.add_edges_from(edges1[:crossover_point] + edges2[crossover_point:])

    return child

def mutate(maze, mutation_rate=0.01):
    """Randomly adds or removes edges in the maze."""
    for edge in nx.grid_2d_graph(7, 7).edges():
        if random.random() < mutation_rate:
            if maze.has_edge(*edge):
                maze.remove_edge(*edge)
            else:
                maze.add_edge(*edge)

def genetic_optimization(rounds_of_selection=1000000, num_init_mazes=10000):
    population = generate_list_of_successful_mazes(num_init_mazes)
    initial_maze = max(population, key=calculate_maze_fitness)
    print(f"Best maze fitness: {calculate_maze_fitness(initial_maze)}")

    for _ in range(rounds_of_selection):
        parent1, parent2 = select_parents(population)

        # Crossover
        child = crossover(parent1, parent2)

        # Mutation
        mutate(child)

        # Replace parent with child if child is better
        if check_if_successful_maze(child) and not has_1x1_square(child):
            child_fitness = calculate_maze_fitness(child)
            parent_fitness = calculate_maze_fitness(parent1)

            if child_fitness > parent_fitness:
                population.remove(parent1)
                population.append(child)

    best_maze = max(population, key=calculate_maze_fitness)
    print(f"Best maze fitness: {calculate_maze_fitness(best_maze)}")
    return best_maze

def optimising_the_optimisation():
    best_parameters = None
    best_fitness = float('-inf')

    for _ in range(100):
        # Randomly vary parameters
        num_iterations = random.randint(1000, 10000)
        tabu_size = random.randint(1, 10)
        initial_temp = random.randint(10, 100)
        cooling_rate = random.uniform(0.9, 0.99)
        num_initial_mazes = random.randint(1, 100)

        # Calculate average fitness over 10 runs
        total_fitness = 0
        for _ in range(10):
            maze = hybrid_optimization(num_iterations, tabu_size, initial_temp, cooling_rate, num_initial_mazes)
            total_fitness += calculate_maze_fitness(maze)
        average_fitness = total_fitness / 10

        # Update best parameters if current set is better
        if average_fitness > best_fitness:
            best_fitness = average_fitness
            best_parameters = [num_iterations, tabu_size, initial_temp, cooling_rate, num_initial_mazes]

    return best_parameters


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
