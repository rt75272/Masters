<center> <h3><strong>LOOPS</strong><h3> </center>  

## Components of a loop
___Example loop:___
```py
arr = ['a', 'b', 'c', 'd', 'e']
while i < 5:
    print(arr[i])
    i+=1
```  
___Loop control variable:___  
    Creates the entry/exit conditions. Updated and checked during each iteration.  
    ```i+=1```

___Loop body:___  
    Instructions to execute during each iteration of the loop.  
    ```print(arr[i])```

___Entry conditions:___  
    Boolean expression to start the loop.  
    ```i < 5```  
    The variable ```i``` is less than 5, therefore the loop will start.

___Exit conditions:___  
    Boolean expression to end the loop.  
    ```i < 5```  
    The variable ```i``` is no longer less than 5, therefore the loop will end.

## Types of loops
___While loops:___  
    Repeated while an expression is true. May or may not execute at all.

___Do while loops:___  
    Repeated until an expression is false. Will run at least once. Checks 
    conditional at the end of the iteration. Could be useful to get user 
    input before running several potential loops. 

___For loops:___  
    Definite amount of iterations. May or may not execute at all. 

___Nested loops:___  
    Loop inside of another loop. Can be any type of loop. Helps iterating 
    through multi dimensional arrays such as matrices and tensors. Need to be 
    careful with nested loops due to the increased run time.

___Infinite loops:___  
    An endless loop that runs forever. Normally avoided.  

## Scenarios
___Scenario 1:___  
A do while loop to get user input while shopping at a store. The do while loop will  continue to execute until the user says they're done shopping.   
```py
shopping = True
shopping_cart = []
while shopping:
    shopping_cart.append(item)
    shopping = input("Keep shopping(True or False)?")
```

___Scenario 2:___  
A plain while loop to run until a user has spent $50 at a store. Once the user 
reaches $50 worth of groceries the loop will end and there will be no more shopping.
```py
shopping_cart = 0
while shopping_cart < 50:
    shopping_cart += item.cost()
```