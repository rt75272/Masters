import sys
import vertex
import pygame
from collections import deque
from tkinter import messagebox, Tk
from dijkstra import Dijkstra, get_random_target
# ----------------------------------------------------------------------------------------
# Main Driver
#
# Runs an animation of Dijkstra's algorithm searching for the shortest path to a randomly
# chosen vertex.
#
# Parameters:
#   pos: The position of the mouse click.
#   state: Boolean indicating whether to set the spot as a wall (True) or not (False).
# ----------------------------------------------------------------------------------------
size = (width, height) = 1920, 1080  # Define the window dimensions.
pygame.init()  # Initialize all Pygame modules.
window = pygame.display.set_mode(size)  # Create a window with the specified size.
pygame.display.set_caption("Dijkstra's Path Finding")  # Set the window title.
clock = pygame.time.Clock()  # Create a clock to control the frame rate.
cols, rows = 64, 48  # Set the number of columns and rows in the grid.
cell_width = width // cols  # Calculate the width of each cell in the grid.
cell_height = height // rows  # Calculate the height of each cell in the grid.
grid = []  # Initialize the grid as an empty list.
queue, visited = deque(), []  # Initialize a queue for BFS and a list for visited nodes.
path = []  # List to store the final path from start to end.

# ----------------------------------------------------------------------------------------
# Toggle the wall state of a spot based on mouse click position.
#
# Parameters:
# - pos: The position of the mouse click.
# - state: Boolean indicating whether to set the spot as a wall (True) or not (False).
# ----------------------------------------------------------------------------------------
def clickWall(pos, state):
    i = pos[0] // cell_width  # Calculate the column index.
    j = pos[1] // cell_height  # Calculate the row index.
    grid[i][j].wall = state  # Set the wall state of the spot.

# Create the grid with Vertex objects.
for i in range(cols):
    arr = []  # Initialize an array for each column.
    for j in range(rows):
        arr.append(vertex.Vertex(i, j))  # Create a new Vertex for each cell.
    grid.append(arr)  # Add the column array to the grid.

# Add neighbors for each vertex in the grid.
for i in range(cols):
    for j in range(rows):
        grid[i][j].add_neighbors(grid)  # Populate each vertex's neighbors.

# Set the start point for pathfinding.
start = grid[cols // 2][rows // 2]  # Choose the center of the grid as the start.
start.wall = False  # Ensure the start point is not a wall.

# Set a random end point for pathfinding.
end = get_random_target(grid, cols, rows)  # Get a random target point.
end.wall = False  # Ensure the end point is not a wall.

# Initialize Dijkstra's algorithm with the grid, start, and end points.
dijkstra = Dijkstra(grid, start, end)

# ----------------------------------------------------------------------------------------
# Main loop for handling events and updating the pathfinding visualization.
# ----------------------------------------------------------------------------------------
def main():
    flag = False  # Indicates if the pathfinding is complete.
    noflag = True  # Indicates if a no-solution message has been shown.
    startflag = True  # Automatically start pathfinding.
    while True:
        for event in pygame.event.get():  # Handle events from the Pygame event queue.
            if event.type == pygame.QUIT:
                pygame.quit()  # Close the Pygame window.
                sys.exit()  # Exit the program.
            if event.type == pygame.MOUSEBUTTONUP:  # Handle mouse clicks.
                if pygame.mouse.get_pressed(0):  # Left click to add walls.
                    clickWall(pygame.mouse.get_pos(), True)
                if pygame.mouse.get_pressed(2):  # Right click to remove walls.
                    clickWall(pygame.mouse.get_pos(), False)
            if event.type == pygame.MOUSEMOTION:  # Keep adding walls while dragging.
                if pygame.mouse.get_pressed()[0]:  # If left mouse button is pressed.
                    clickWall(pygame.mouse.get_pos(), True)
        if startflag:
            if len(dijkstra.queue) > 0:  # While there are spots to explore.
                current = dijkstra.queue.popleft()  # Get the next spot from the queue.
                if current == dijkstra.end:  # If the end is reached.
                    temp = current
                    while temp.prev:  # Backtrack to get the path.
                        path.append(temp.prev)
                        temp = temp.prev
                    flag = True  # Indicate that pathfinding is complete.
                    print("Done")  # Log that pathfinding is finished.
                if not flag:  # If pathfinding is still ongoing.
                    for neighbor in current.neighbors:  # Explore neighbors.
                        if not neighbor.visited and not neighbor.wall:  # Only if not visited and not a wall.
                            neighbor.visited = True  # Mark the neighbor as visited.
                            neighbor.prev = current  # Set the previous node for backtracking.
                            dijkstra.queue.append(neighbor)  # Add the neighbor to the queue.
            else:
                if noflag and not flag:  # If no solution was found.
                    Tk().wm_withdraw()  # Hide the Tkinter root window.
                    messagebox.showinfo("No Solution", "There was no solution")  # Show message.
                    noflag = False  # Prevent further messages.
        # Render the grid.
        window.fill((0, 20, 20))  # Clear the window with a background color.
        for i in range(cols):
            for j in range(rows):
                spot = grid[i][j]  # Get the current spot.
                spot.show(window, (44, 62, 80))  # Show the spot in default color.
                if spot in path:  # If part of the path, highlight it.
                    spot.show(window, (192, 57, 43))  # Path color.
                elif spot.visited:  # If the spot has been visited.
                    spot.show(window, (39, 174, 96))  # Visited color.
                if spot == start:  # Highlight the start spot.
                    spot.show(window, (0, 255, 200))  # Start color.
                if spot == end:  # Highlight the end spot.
                    spot.show(window, (0, 120, 255))  # End color.
        pygame.display.flip()  # Update the display with the rendered grid.
        clock.tick(100)  # Control the animation speed.

# Big red button to start the main loop.
if __name__ == "__main__":
    main()
