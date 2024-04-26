#------------------------------------------------------------------------------
# Module 4: Portfolio Milestone - Shopping Cart
#
# @author Ryan Thompson
#
# Usage:
#   $ python ItemToPurchase.py
#   [follow prompt to complete program]
#------------------------------------------------------------------------------

# Item object class.
# Builds up an item object with the item's name, cost, and quantity.
class ItemToPurchase:
    # Default constructor
    def __init__(myself, item_name="none", item_price=0.0, item_quantity=0):
        myself.item_name = item_name
        myself.item_price = item_price
        myself.item_quantity = item_quantity

    # Outputs the item's name, cost, and quantity.
    def print_item_cost(myself):
        item_total = myself.item_price * myself.item_quantity
        print(f"{myself.item_name} {myself.item_quantity} @ ${myself.item_price:.2f} = ${item_total:.2f}")

# Grabs user input to build item objects for purchase.
# Returns a list of items for purchase.
def get_items():
    items = []
    num_items = 2
    for i in range(num_items):
        name = input("\nEnter the item name: ")
        cost = float(input("Enter the item price: "))
        quantity = int(input("Enter the quantity: "))
        item = ItemToPurchase(name, cost, quantity)
        items.append(item)
    return items

# Calculates the total cost of the two items.
# Returns the total cost.
def get_total_cost(items):
    total_cost = 0
    for item in items:
        item_total = item.item_price * item.item_quantity
        total_cost += item_total
    return total_cost

# Prints the entire recept contents
# Returns none.
def print_receipt(items, total_cost):
    print()
    for item in items:
        item.print_item_cost()
    print(f"Total: ${total_cost:.2f}")

# Main driver function.
# Returns none.
def main():
    items = get_items()
    total_cost = get_total_cost(items)
    print_receipt(items, total_cost)

# Pushing the big red button.
main()