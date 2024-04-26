<h1 style="text-align: center;">AND OR Operators</h1>
AND/OR operators allow complex statements to evaluate to true or false. 
Thus, making them a powerful tool for making decisions in programming. 
For example, if you need to make sure a variable(s) is true before running 
a block of code. Conversely, you could also use AND/OR operators to make 
sure a variable(s) is false before running a block of code. 

```py
# Checking if a variable is a bomb. 
bomb = False 
safe = True
if(safe and not bomb):
    print("False alarm.")
else:
    print("Boom.") 
```

OR operators more often evaluate to true when compared to the AND 
operator. The OR truth table shows how a conditional's outcome can
be true 75% of the time. Whereas, the AND truth table shows the 
outcome evaluating to true 25% of the time. This is because the OR
operator only needs one of the variables to be true in order for the 
conditional's outcome to be true. Therefore, an OR operator will output
true as long as any of the variables are true. On the opposite end, the 
AND operator needs both variables to be true in order to output true. If
just one variable is true and the other is false, then the outcome is 
false.

&nbsp;  
__OR Truth Table:__
|   X    |   Y    |Outcome |
| :----: | :----: | :----: |
|   0    |   0    |  0     |
|   0    |   1    |  1     |
|   1    |   0    |  1     |
|   1    |   1    |  1     |
||  

&nbsp;  
__AND Truth Table:__
|   X    |   Y    |Outcome |
| :----: | :----: | :----: |
|   0    |   0    |  0     |
|   0    |   1    |  0     |
|   1    |   0    |  0     |
|   1    |   1    |  1     |
||

__AND OR Psuedocode:__

```py
x = 0
y = 1

# Simple AND example.
if y and y:
    print("True")
else:
    print("False")

# Simple OR example.
if x or y:
    print("True")
else:
    print("False")


# Ask if the user knows the answer to the universe.
user_input = int(input("What is the answer to the universe? ")) 
answer = 42 # Set the answer to the universe.

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
```

That is all!

Thank you,

-Ryan Thompson