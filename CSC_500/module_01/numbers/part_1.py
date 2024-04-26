#####################################################
# Part 1 - Add/Subtract
#
# Asks the user for two numbers then outputs the 
# result of the numbers being added together and
# the numbers being subtracted.
#
# Usage:
#   $ python part_1.py
#####################################################

# Asks the user for two numbers.
# Returns the two numbers as num1 and num2.
def get_numbers():
    # Convert num1 and num2 to floats, incase the user enters decimals.
    num1 = float(input("Enter a number: "))
    num2 = float(input("Enter another number: "))
    return num1, num2

# Takes in two numbers and adds them together.
# Returns the addition output.
def add(num1, num2):
    num_sum = num1 + num2
    return num_sum

# Takes in two numbers and subtracts them. 
# Returns the subtraction output.
def subtract(num1, num2):
    difference = num1 - num2
    return difference

# Main driver function.
# Makes use of all the other functions.
def main():
    num1, num2 = get_numbers() # Grabbing the numbers the user entered.
    add_output = add(num1, num2) # Process the numbers added together.
    subtract_output = subtract(num1, num2) # Process the numbers being subtracted.
    # Display the results in the command line/terminal. 
    print(str(num1) + " + " + str(num2) + " = " + str(add_output))
    print(str(num1) + " - " + str(num2) + " = " + str(subtract_output))

# Pushing the big red button.
main()

