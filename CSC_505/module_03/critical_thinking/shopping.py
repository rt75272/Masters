# --------------------------------------------------------------
# Shopping List
#
# Builds ands saves a user specified shopping list
#
# Usage:
#   $ python shopping.py
# --------------------------------------------------------------

# Main driver function.
def main():
    add_more_items = True

    # Continues to loop until no more items are to be added to 
    # the shopping list.
    while add_more_items:
        # Get the shopping list item to be added.
        item = input("Enter shopping item: ")

        # File to save our shopping list.
        file = open("shopping_list.txt", "a")
        file.write(item + "\n")

        # Ask if more items are desired.
        more_items = input("Add more items?(y/n) ")
        if(more_items == "n"):
            add_more_items = False
    
    # Close and save our shopping list file.
    file.close()

# Pushing the big red button.
if __name__ == "__main__":
    main()