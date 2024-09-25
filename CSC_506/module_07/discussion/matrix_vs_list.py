import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
"""----------------------------------------------------------------------------------
Adjacency Matrix VS Adjacency List

Builds and compares an adjacency matrix against an adjacency list. Plots both 
next to each other.

Usage:
    $ python matrix_vs_list.py
----------------------------------------------------------------------------------"""
# -----------------------------------------------------------------------------------
# Get Adjacency Matrix
#
# Sets up an adjacency matrix in order to build and return its graphing information.
#
# Returns: Matrix graph data.
# -----------------------------------------------------------------------------------
def get_adjacency_matrix():
    # Adjacency matrix.
    # 0: Not connected to vetex. 1: Connected to vertex.
    adj_matrix = [
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0]
    ]
    graph_matrix = nx.Graph() # Graph object for adjacency matrix(for display).
    # Add edges based on the adjacency matrix. Connect vertices.
    for i in range(len(adj_matrix)):
        for j in range(i, len(adj_matrix)):
            if adj_matrix[i][j] == 1:
                graph_matrix.add_edge(i, j)
    return graph_matrix

# -----------------------------------------------------------------------------------
# Get Adjacency List
#
# Sets up an adjacency list in order to build and return its graphing information.
#
# Returns: List graph data.
# -----------------------------------------------------------------------------------
def get_adjacency_list():
    # Adjacency list. 
    # vertex: [vertex or vertices connected to]
    adj_list = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1, 3],
        3: [2]
    }
    graph_list = nx.Graph(adj_list) # Graph object for adjacency list(for display).
    return graph_list

# -----------------------------------------------------------------------------------
# Main driver function.
#
# Handles processing and plotting sets.
#
# Returns: none
# -----------------------------------------------------------------------------------
def main():
    # Get graph data for matrix and list.
    g_matrix = get_adjacency_matrix() # Matrix graph data.
    g_list = get_adjacency_list() # List graph data.
    # Figure setup.
    plt.figure(figsize=(12, 6))
    # Figure dimensions.
    n_rows = 1 # Number of rows.
    n_cols = 2 # Number of columns.
    index = 1 # Plot index begins at 1, being at the uppper left corner.
    # Display matrix and list graphs.
    plot_matrix(n_rows, n_cols, index, g_matrix) # Plot adjacency matrix.
    plot_list(n_rows, n_cols, index, g_list) # Plot adjacency list.
    plt.show() # Display the plots.

# -----------------------------------------------------------------------------------
# Plot the matrix representation.
#
# Processes the matrix data and sets up the graph for it.
#
# Returns: none
# -----------------------------------------------------------------------------------
def plot_matrix(n_rows, n_cols, index, graph_option):
    # Draw the graph from the adjacency matrix at index 1.
    ax1 = plt.subplot(n_rows, n_cols, index)
    # Color the background.
    rect1 = patches.Rectangle((0, 0), 1, 1, transform=ax1.transAxes,
                color='#cd5c5c', zorder=-1)
    ax1.add_patch(rect1)
    # Plot the final graph.
    nx.draw(graph_option, with_labels=True, node_color='#6495ed',
        edge_color='blue', node_size=2000, font_size=16, 
        font_weight='bold', ax=ax1)
    plt.title("Adjacency Matrix", fontsize=16, fontweight="bold", color="green")

# -----------------------------------------------------------------------------------
# Plot the list representation.
#
# Processes list data and sets up the graph for it.
#
# Returns: none
# -----------------------------------------------------------------------------------
def plot_list(n_rows, n_cols, index, graph_option):
    # Draw the graph from the adjacency list at index 2.
    ax2 = plt.subplot(n_rows, n_cols, index+1)
    # Color the background.
    rect2 = patches.Rectangle((0, 0), 1, 1, transform=ax2.transAxes,
                color='#6495ed', zorder=-1)
    ax2.add_patch(rect2)
    # Plot the final graph.
    nx.draw(graph_option, with_labels=True, node_color='#cd5c5c',
        edge_color='gray', node_size=2000, font_size=16, 
        font_weight='bold', ax=ax2)
    plt.title("Adjacency List", fontsize=16, fontweight="bold", color="green")

# Big red button.
if __name__ == '__main__':
    main()