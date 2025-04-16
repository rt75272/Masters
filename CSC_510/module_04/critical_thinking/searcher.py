from simpleai.search import SearchProblem, astar
from simpleai.search.viewers import BaseViewer

class PathfindingProblem(SearchProblem):
    """This class represents the pathfinding problem on a grid where we aim to 
    find the shortest path from a start point to a goal point while avoiding obstacles.
    """

    def __init__(self, grid, start, goal):
        """Initialize the problem with a grid, start position, and goal position.
        
        :param grid: The 2D grid representing the map with obstacles.
        :param start: The starting point (x, y).
        :param goal: The goal point (x, y).
        """
        self.grid = grid
        self.start = start
        self.goal = goal
        super().__init__(initial_state=start)

    def actions(self, state):
        """Define the possible actions (valid moves) from the current state.
        The valid moves are up, down, left, and right, as long as they are within
        the grid and do not hit an obstacle.
        
        :param state: The current position on the grid (x, y).
        :return: A list of valid neighboring positions.
        """
        actions = []
        x, y = state
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right.
        # Check all possible directions.
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(self.grid) and 0 <= ny < len(self.grid[0]) and self.grid[nx][ny] != 1:
                actions.append((nx, ny))  # Add valid moves.
        return actions

    def result(self, state, action):
        """The result of an action is simply the new state (new position on the grid).
        
        :param state: The current position (x, y).
        :param action: The action taken, which results in the next position.
        :return: The new position after the action.
        """
        return action

    def is_goal(self, state):
        """Check if the current state is the goal state (destination).
        
        :param state: The current position (x, y).
        :return: True if the current state is the goal, False otherwise.
        """
        return state == self.goal

    def cost(self, state, action, result):
        """Each move costs 1 step (unit cost for all moves).
        
        :param state: The current position (x, y).
        :param action: The action taken (next position).
        :return: The cost of the move (1 step).
        """
        return 1

    def heuristic(self, state):
        """Heuristic function: Manhattan distance between current state and the goal state.
        
        :param state: The current position (x, y).
        :return: The Manhattan distance from the current position to the goal position.
        """
        x1, y1 = state
        x2, y2 = self.goal
        return abs(x1 - x2) + abs(y1 - y2)

def main():
    """Main driver function."""
    # Example grid: 0 = open space, 1 = obstacle.
    grid = [
        [0, 0, 0, 0, 0],  # Row 1.
        [0, 1, 1, 0, 0],  # Row 2 (obstacles at (1,1) and (1,2)).
        [0, 1, 0, 1, 0],  # Row 3 (obstacles at (2,1) and (2,3)).
        [0, 0, 0, 0, 0],  # Row 4.
        [0, 0, 0, 1, 0]   # Row 5 (obstacle at (4,3)).
    ]
    start = (0, 0)  # Starting position at the top-left corner.
    goal = (4, 4)   # Goal position at the bottom-right corner.
    # Create an instance of the pathfinding problem.
    problem = PathfindingProblem(grid, start, goal)
    # Run A* search to find the solution.
    result = astar(problem, viewer=BaseViewer())
    # Display the result.
    if result:
        print("Path found:", result.path())
    else:
        print("No path found.")

# Big red activation button.
if __name__ == "__main__":
    main()