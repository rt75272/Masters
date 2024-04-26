# Ask if the user knows the answer to the universe.
user_input = int(input("What is the answer to the universe? ")) 
answer = 42 # Set the answer to the universe.

x = 0
y = 1

# AND example.
if user_input == answer and y == 1:
    print("Correct")
else:
    print("Incorrect")

# OR example.
if user_input == answer or x == 1:
    print("Correct")
else:
    print("Incorrect")

# Simple AND example.
if True and True:
    print("True")

# Simple OR example.
if True or False:
    print("True")


bomb = False
safe = True
if(safe and not bomb):
    print("False alarm.")
else:
    print("Boom.")