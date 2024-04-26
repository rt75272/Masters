class ItemToBuy:
    # Default constructor.
    def __init__(self, name="none", price=0.0, quantity=0, description="none"):
        self.name = name
        self.price = price
        self.quantity = quantity
        self.description = description        

    def print_info(self):
        total = float(self.price) * float(self.quantity)
        print(
            f"{self.quantity} {self.name} @ ${self.price} {self.description} = ${total}"
        )

class Cart:
    # Default constructor
    def __init__(self, customer_name="none"):
        self.customer_name = customer_name
        self.cart_items = []

    # Adding to our array of cart items.
    def add_item(self, ItemToBuy):
        if(ItemToBuy.name != "none"):
            self.cart_items.append(ItemToBuy)
            print(f"{ItemToBuy.name} added to cart")
        else:
            print(f"Nothing added to cart :(")

    def remove_item(self, name="none"):
        found = False
        for item in self.cart_items:
            if item.name == name:
                self.cart_items.remove(item)
                found = True
                print(f"{name} has been removed")
                break 
            else:
                print("Item not found in cart. Nothing removed.")

    def show_cart(self):
        n = len(self.cart_items)
        if n > 0:
            for item in self.cart_items:
                item.print_info()
        else:
            print("Your cart is empty :(")
    
    def get_total_cost(self):
        cost = 0
        for item in self.cart_items:
            cost += float(item.price) * float(item.quantity) 
        print("Total cost: $%.2f" % cost)   
        
    def change_quantity(self, name, quantity):
        for item in self.cart_items:
            if(item.name == name):
                item.quantity = quantity
    
    def change_description(self, name, description):
        for item in self.cart_items:
            if(item.name == name):
                item.description = description

def print_menu(cart):
    menu = ("\n\tMENU\n"
            "a - Add item to cart\n"
            "r - Remove item from cart\n"
            "c - Change item quantity\n"
            "o - Output shopping cart\n"
            "q - Quit shopping\n"
    )
    command = ''
    while command != 'q':
        print(menu)
        command = input("Choose an option:\n")
        if(command == 'a'):
            print("Add item")
            name = input("Enter item name: ")
            price = input("Enter item price: ")
            quantity = input("Enter item quantity: ")
            description = input("Enter item description: ")
            new_item = ItemToBuy(name, price, quantity, description)
            cart.add_item(new_item)
        elif(command == 'r'):
            print("Remove item")
        elif(command == 'c'):
            print("Change item quantity")
        elif(command == 'o'):
            print("Output cart")
            cart.show_cart()
        elif(command == 'q'):
            print("Shutting down")
        else:
            print("Something blew up")


def main():
    # x = ItemToBuy("Cookies", 5.99, 2, "Food")
    # y = ItemToBuy("Nikes", 199.99, 1, "Shoes")
    # cart = Cart("Bob")
    # cart.add_item(x)
    # cart.add_item(y)
    cart = Cart("Bob")
    print_menu(cart)
    cart.change_quantity("Cookies", 4)
    cart.change_description("Cookies", "Death")
    cart.show_cart()
    cart.get_total_cost()
    cart.remove_item("Cookies")
    cart.show_cart()

# Pushing the big red button.
if __name__ == "__main__":
    main()
    # print("hell")
    # x.print_info()