import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from GridMaze.maze import builder as mb
from GridMaze.maze import directed_graph_builder as dgb

def plot_maze(
    maze,
    highlight_nodes=False,
    highlight_color="lime",
    special_location2color=None,
    node_size=300,
    edge_size=10,
    font_size=10,
    ):
    '''Function that plots a maze on cartesian coordinates.'''
    node_positions = {node: node for node in maze.nodes()}
    
    nx.draw_networkx(maze,pos=node_positions, with_labels=False, font_size=font_size, edge_color='black')
    unidirectional_edges = [(u, v) for u, v in maze.edges() if not maze.has_edge(v, u)]
    nx.draw_networkx_edges(maze,pos=node_positions, edgelist=unidirectional_edges, edge_color='#FC0FC0', arrows=True)

    plt.show()


