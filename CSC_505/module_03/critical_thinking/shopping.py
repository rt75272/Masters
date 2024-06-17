add_more_items = True

while add_more_items:
    item = input("Enter shopping item: ")
    file = open("shopping_list.txt", "a")
    file.write(item + "\n")
    more_items = input("Add more items?(y/n) ")
    if(more_items == "n"):
        add_more_items = False

file.close()