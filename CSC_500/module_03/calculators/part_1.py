from termcolor import colored
#--------------------------------------------------------------------
# Part 1 - Meal Cost Calculator
#
# Usage:
#   $ python part_1.py
#   [follow prompt to complete program]
#--------------------------------------------------------------------

# Set the number of decimals for each cost, as a constant.
NUM_DECIMALS = 2

# Get the food cost from the user.
# Returns the food cost.
def get_food_cost():
    food_cost = float(input("Enter the charge for the food: "))
    food_cost = round(food_cost, NUM_DECIMALS)
    return food_cost

# Total tip amount, relative to the food cost.
# 18% tip.
# Returns the tip amount.
def get_tip(food_cost):
    tip_rate = 0.18
    tip_amount = food_cost * tip_rate
    tip_amount = round(tip_amount, NUM_DECIMALS)
    return tip_amount

# Total sales tax amount, relative to the food cost.
# 7% sales tax.
# Returns the sales tax amount.
def get_tax(food_cost):
    sales_tax_rate = 0.07
    sales_tax_amount = food_cost * sales_tax_rate
    sales_tax_amount = round(sales_tax_amount, NUM_DECIMALS)
    return sales_tax_amount

# Total cost of food, tip, and sales tax. 
# Returns the total cost.
def calculate_total(food_cost, tip_amount, sales_tax_amount):
    total_cost = food_cost + tip_amount + sales_tax_amount
    total_cost = round(total_cost, NUM_DECIMALS)
    return total_cost

# Output receipt to terminal.
# Returns none.
def printer(food_cost, tip_amount, sales_tax_amount, total_cost):
    # List containing all of the required receipt information.
    receipt_info = [food_cost, tip_amount, sales_tax_amount, total_cost]
    # Outputs all the receipt info from the list.
    print("\nFood cost:\t" + "$" + str(receipt_info[0]))
    print("Tip amount:\t" + "$" + str(receipt_info[1]))
    print("Sales tax:\t" + "$" + str(receipt_info[2]))
    print(colored("Total cost:\t" + "$" + str(receipt_info[3]), "white", attrs=['bold']))
    print()

# Main driver function.
# Returns none.
def main():
    food_cost = get_food_cost()
    tip = get_tip(food_cost)
    tax = get_tax(food_cost)
    total = calculate_total(food_cost, tip, tax)
    printer(food_cost, tip, tax, total)

# Pushing the big red button.
main()

