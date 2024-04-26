# Item object class.
# Builds up an item object with the item's name, cost, and quantity.
class ItemToPurchase:
    # Default constructor.
    def __init__(myself, item_name="none", item_price=0.0, item_quantity=0, description="none"):
        myself.item_name = item_name
        myself.item_price = item_price
        myself.item_quantity = item_quantity
        myself.description = description

    # Outputs the item's name, cost, and quantity.
    def print_item_cost(myself):
        item_total = myself.item_price * myself.item_quantity
        print(f"{myself.item_name} {myself.item_quantity} @ ${myself.item_price:.2f} = ${item_total:.2f}")

    # Displays an item's description.
    def print_item_description(myself):
        print(f"{myself.item_name}: {myself.description}")

# Shopping cart object class.
# Builds up a shopping with ItemToPurchase objects.
class ShoppingCart:
    # Default constructor.
    def __init__(self, customer_name="none", current_date="January 1, 2020"):
        self.customer_name = customer_name
        self.current_date = current_date
        self.cart_items = []

    # Adds an item to cart_item list.
    # Returns none.
    def add_item(self, ItemToPurchase):
        self.cart_items.append(ItemToPurchase)

    # Removes item from cart_items list
    # Returns none.
    def remove_item(self, item_name="none"):
        found = False
        for item in self.cart_items:
            if item.item_name == item_name:
                self.cart_items.remove(item)
                found = True
                print(f"{item_name} has been removed")
                break
            else:
                print("Item not found in cart. Nothing removed.")
    
    # Modifies an item's description, price, and/or quantity.
    # Returns none.
    def modify_item(self, ItemToPurchase):
        found = False
        for i, item in enumerate(self.cart_items):
            if item.item_name == ItemToPurchase.item_name:
                if ItemToPurchase.description != "none":
                    self.cart_items[i].description = ItemToPurchase.description
                if ItemToPurchase.item_price != 0.0:
                    self.cart_items[i].price = ItemToPurchase.item_price
                if ItemToPurchase.item_quantity != 0:
                    self.cart_items[i].quantity = ItemToPurchase.item_quantity
                found = True
                break
        if not found:
            print("Item not found in cart. Nothing modified.")
    
    # Returns quantity of all items in cart.
    def get_num_items_in_cart(self):
        quantity = sum(item.item_quantity for item in self.cart_items)
        return quantity
    
    # Determines and returns the total cost of items in cart.
    def get_cost_of_cart(self):
        cost = sum(item.item_price * item.item_quantity for item in self.cart_items)
        return cost
    
    # Outputs total of objects in cart.
    # Returns none.
    def print_total(self):
        total = self.get_cost_of_cart()
        if total == 0:
            print("SHOPPING CART IS EMPTY")
        else:
            print(f"{self.customer_name}'s Shopping Cart - {self.current_date}")
            print(f"Number of Items: {self.get_num_items_in_cart()}")
            for item in self.cart_items:
                item.print_item_cost()
            print(f"Total: ${total:.2f}")
        
    # Outputs each item's description.
    # Returns none.
    def print_descriptions(self):
        print(f"{self.customer_name}'s Shopping Cart - {self.current_date}")
        print("Item Descriptions")
        for item in self.cart_items:
            item.print_item_description()

# Displays the interactive menu for the user. 
# Returns none.
def print_menu(cart):
    menu = ("\n\tMENU\n"
            "a - Add item to cart\n"
            "r - Remove item from cart\n"
            "c - Change item quantity\n"
            "i - Output items' descriptions\n"
            "o - Output shopping cart\n"
            "q - Quit\n")
    command = ''
    while command != 'q':
        print(menu)
        command = input("Choose an option:\n")
        # Add an item.
        if command == 'a':
            print("\nADD ITEM TO CART")
            item_name = input("Enter the item name:\n")
            item_description = input("Enter the item description:\n")
            item_price = float(input("Enter the item price:\n"))
            item_quantity = int(input("Enter the item quantity:\n"))
            new_item = ItemToPurchase(item_name, item_price, item_quantity, item_description)
            cart.add_item(new_item)
        # Remove an item.
        elif command == 'r':
            print("\nREMOVE ITEM FROM CART")
            item_name = input("Enter name of item to remove:\n")
            cart.remove_item(item_name)
        # Change an item quantity.
        elif command == 'c':
            print("\nCHANGE ITEM QUANTITY")
            item_name = input("Enter the item name:\n")
            quantity = int(input("Enter the new quantity:\n"))
            item_to_modify = ItemToPurchase(item_name=item_name, item_quantity=quantity)
            cart.modify_item(item_to_modify)
        # Display item descriptions.
        elif command == 'i':
            print("\nOUTPUT ITEMS' DESCRIPTIONS")
            cart.print_descriptions()
        # Display shopping cart.
        elif command == 'o':
            print("\nOUTPUT SHOPPING CART")
            cart.print_total()
        # Quit the menu.
        elif command == 'q':
            print("Quit")
        # Incase something weird happens.
        else:
            print("Something blew up")

# Main driver function.
# Returns none.
def main():
    customer_name = input("Enter customer's name: ")
    current_date = input("Enter today's date: ")
    cart = ShoppingCart(customer_name, current_date)
    print_menu(cart)

# Pushing the big red button.
if __name__ == "__main__":
    main()