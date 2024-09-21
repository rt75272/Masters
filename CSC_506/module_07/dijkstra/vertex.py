import pygame
# ---------------------------------------------------------------------------------------- 
# Vertex
#
# Vertex object for the grid animation. 
#
# Parameters:
#   i: The x-coordinate (column) of the vertex.
#   j: The y-coordinate (row) of the vertex.
# ----------------------------------------------------------------------------------------
class Vertex:
    # ---------------------------------------------------------------------------------------- 
    # Initialize a vertex with its position and properties.
    #
    # Parameters:
    #   i: The x-coordinate (column) of the vertex.
    #   j: The y-coordinate (row) of the vertex.
    # ----------------------------------------------------------------------------------------
    def __init__(self, i, j):
        self.x, self.y = i, j  # Store the position of the vertex.
        self.neighbors = []  # Initialize an empty list for neighboring vertices.
        self.prev = None  # Previous vertex in the path (for pathfinding).
        self.wall = False  # Indicates if this vertex is a wall.
        self.visited = False  # Indicates if this vertex has been visited during pathfinding.
        self.cols = 64  # Number of columns in the grid/animation.
        self.rows = 48  # Number of rows in the grid/animation.

    # ----------------------------------------------------------------------------------------
    # Render the vertex on the window.
    #
    # Parameters:# If it's a wall, color it black.
    #   window: The Pygame window surface to draw on.
    #   color: The color to use for rendering the vertex.
    #   shape: The shape of the vertex; 1 for rectangle, others for circle.
    # ----------------------------------------------------------------------------------------
    def show(self, window, color, shape=1):
        screen_width, screen_height = 1920, 1080  # Set the dimensions of the animation screen.
        cell_width = screen_width // self.cols  # Calculate the width of each cell in the grid.
        cell_height = screen_height // self.rows  # Calculate the height of each cell in the grid.
        # Check if the vertex is a wall and set color accordingly.
        if self.wall: # If it's a wall, color it black.
            color = (0, 0, 0) 
        # Draw a rectangle for the vertex.
        if shape == 1:
            pygame.draw.rect(window, color, (self.x * cell_width, self.y * cell_height, cell_width - 1, cell_height - 1))
        # Draw a circle for the vertex (if needed).
        else:
            pygame.draw.circle(window, color, (self.x * cell_width + cell_width // 2, self.y * cell_height + cell_height // 2), cell_width // 3)

    # ----------------------------------------------------------------------------------------
    # Add neighboring vertices to the current vertex (up, down, left, right).
    #
    # Parameters:
    #   grid: The 2D grid of vertices from which to add neighbors.
    # ----------------------------------------------------------------------------------------
    def add_neighbors(self, grid):
        # Check if the right neighbor exists and add colit.
        if self.x < self.cols - 1:
            self.neighbors.append(grid[self.x + 1][self.y])
        # Check if the left neighbor exists and add it.
        if self.x > 0:
            self.neighbors.append(grid[self.x - 1][self.y])
        # Check if the bottom neighbor exists and add it.
        if self.y < self.rows - 1:
            self.neighbors.append(grid[self.x][self.y + 1])
        # Check if the top neighbor exists and add it.
        if self.y > 0:
            self.neighbors.append(grid[self.x][self.y - 1])
