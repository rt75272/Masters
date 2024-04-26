#####################################################
# Part 2 - Multiply/Divide
#
# Asks the user for two numbers then outputs the 
# result of the numbers being multiplied and
# the numbers being divided.
#
# Usage:
#   $ python part_2.py
#####################################################

# Asks the user for two numbers.
# Returns the two numbers as num1 and num2.
def get_numbers():
    # Convert num1 and num2 to floats, incase the user enters decimals.
    num1 = float(input("Enter a number: "))
    num2 = float(input("Enter another number: "))
    return num1, num2

# Takes in two numbers and muliplies them by each other.
# Returns the multiplication output.
def multiply(num1, num2):
    product = num1 * num2
    return product

# Takes in two numbers and divides them. 
# Returns the division output.
def divide(num1, num2):
    quotient = num1 / num2
    return quotient

# Main driver function.
# Makes use of all the other functions.
def main():
    num1, num2 = get_numbers() # Grabbing the numbers the user entered.
    multiply_output = multiply(num1, num2) # Process the numbers multiplied together.
    divide_output = divide(num1, num2) # Process the numbers being divided.
    # Display the results in the command line/terminal. 
    print(str(num1) + " * " + str(num2) + " = " + str(multiply_output))
    print(str(num1) + " / " + str(num2) + " = " + str(divide_output))

# Pushing the big red button.
main()

