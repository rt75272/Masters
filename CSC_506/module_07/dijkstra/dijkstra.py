from collections import deque
import random
# -------------------------------------------------------------------------------------
# Dijkstra's Algorithm
#
# Dijkstra's algorithm implementation in order to locate the shortest path.
#
# Usage:
#   from dijkstra import Dijkstra, get_random_target
#   dijkstra = Dijkstra(grid, start, end)
#   end = get_random_target(grid, cols, rows)
# -------------------------------------------------------------------------------------
class Dijkstra:
    # -------------------------------------------------------------------------------------
    # Initialize the Dijkstra class with a grid, a starting point, and an ending point.
    #
    # Parameters:
    #   grid: The grid containing the spots to navigate.
    #   start: The starting spot for pathfinding.
    #   end: The ending spot for pathfinding.
    # -------------------------------------------------------------------------------------
    def __init__(self, grid, start, end):
        self.grid = grid  # Store the grid.
        self.start = start  # Store the starting spot.
        self.end = end  # Store the ending spot.
        self.queue = deque([start])  # Initialize the queue with the starting spot.
        self.path = []  # List to store the final path.
        self.start.visited = True  # Mark the starting spot as visited.

    # -------------------------------------------------------------------------------------
    # Execute the Dijkstra algorithm to find the shortest path from start to end.
    #
    # Returns:
    #   True if a path is found, False otherwise.
    # -------------------------------------------------------------------------------------
    def run(self):
        while self.queue:  # Continue until there are no more spots to explore.
            current = self.queue.popleft()  # Get the next spot from the queue.
            if current == self.end:  # If the current spot is the end spot.
                self.retrace_path(current)  # Retrace the path from end to start.
                return True  # Path found.
            # Explore the neighbors of the current spot.
            for neighbor in current.neighbors:
                # Check if the neighbor has not been visited and is not a wall.
                if not neighbor.visited and not neighbor.wall:
                    neighbor.visited = True  # Mark the neighbor as visited.
                    neighbor.prev = current  # Set the current spot as the previous spot.
                    self.queue.append(neighbor)  # Add the neighbor to the queue for exploration.
        return False  # No path found if the queue is empty.

    # -------------------------------------------------------------------------------------
    # Retrace the path from the end spot to the start spot by following the 'prev' pointers
    #
    # Parameters:
    #   current: The current spot from which to retrace the path.
    # -------------------------------------------------------------------------------------
    def retrace_path(self, current):
        while current.prev is not None:  # While there is a previous spot.
            self.path.append(current.prev)  # Add the previous spot to the path.
            current = current.prev  # Move to the previous spot.
        self.path.reverse()  # Reverse the path to show it from start to end.

# -------------------------------------------------------------------------------------
# Generate a random target point within the grid that is not a wall.
#
# Parameters:
#   grid: The grid containing the spots.
#   cols: Number of columns in the grid.
#   rows: Number of rows in the grid.
#
# Returns:
#   end: A valid end spot that is not a wall.
# -------------------------------------------------------------------------------------
def get_random_target(grid, cols, rows):
    while True:
        end_x = random.randint(0, cols-1)  # Random x-coordinate.
        end_y = random.randint(0, rows-1)  # Random y-coordinate.
        end = grid[end_x][end_y]  # Get the corresponding spot from the grid.
        if not end.wall:  # Ensure the end point is not a wall.
            return end  # Return the valid end point.
