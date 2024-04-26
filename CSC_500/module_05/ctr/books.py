from termcolor import colored
#-----------------------------------------------------------------------
# Bookstore Points
# 
# Module 5 Critical Thinking Assignment - Part 2 
#
# Calculates and outputs the points a user earned at the book store 
# this month.
#
# Usage:
#   $ python books.py
#   [follow prompt to complete the program]
#-----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Asks the user for the amount of books they purchased this month.
#
# Returns the amount of books purchased.
# ----------------------------------------------------------------------
def get_num_books():
    num_books = int(input("How many books did you buy this month? "))
    return num_books

# ----------------------------------------------------------------------
# Calculates the amount of points the user earned.
#
# Returns the amount of points.
# ----------------------------------------------------------------------

def get_points(num_books):
    points = 0
    if(num_books >= 2 and num_books < 4):
        points = 5
    elif(num_books >= 4 and num_books < 6):
        points = 15
    elif(num_books >= 6 and num_books < 8):
        points = 30
    elif(num_books >= 8):
        points = 60
    else:
        print("Something broke.")
    return points

# ----------------------------------------------------------------------
# Main driver function.
#
# Returns none.
# ----------------------------------------------------------------------
def main():
    num_books = get_num_books()
    points = get_points(num_books)
    print(colored(f"You earned {points} points this month!", "cyan", attrs=["bold"]))

# Pushing the big red button.
main()