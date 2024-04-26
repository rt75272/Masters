0: When choosing a predefined function or not it firstly depends if I am trying to learn by
    building the function myself or if I am already familiar with the concept and need to
    get the function into production as quickly as possible. 

1: The predefined function has been tested and does exactly what I need it to do. For 
    example, Instead of calculating the length of a variable, I would use Python's 
    built in len().  
```py
len(x)
```
2: I need a function to do something entirely custom for a program getting built. This would 
    not use len() or any other built in functions.
```py
def hello():
    print("Hola World")
```
3: Do I need to stretch a predefined function to something slightly custom. But saves time 
    by starting with the predefined function. I could use the built in len() function to get
    the length of a variable then expand on it by doing something with the output of len().
```py
def hello(x):
    num = len(x)
    print("Hola World " * x)
```