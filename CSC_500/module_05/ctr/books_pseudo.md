function get_num_books:
    num_books = input("How many books did you buy this month? ")
    return num_books

function get_points(num_books):
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

function main:
    num_books = get_num_books()
    points = get_points(num_books)
    print("You earned {points}!")

    