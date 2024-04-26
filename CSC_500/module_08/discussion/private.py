from termcolor import colored
# Class. 
class private_variables:
    # Hidden attribute "private variable".
    __private_var = 42;
    
    # Hidden attribute "private method".
    def __private_method(self):
        print(f"Inside private")
    
    # Public method which displays the private variable. 
    def hello(self):
        print(
            colored(
                f"\nPrivate variable value: {private_variables.__private_var}",
                "cyan", attrs=["bold"]
            )
        )

# Instance of the class.
foo = private_variables()

# Catching errors.
try:
    # Display private variable with a public method.
    foo.hello()
except:
    print("\nThe public hello method should be accessible.")

# Private variable error.
try:
    # Cant access outside of class. Throws an error.
    foo.__private_var
except:
    print(
        colored(
            f"\nPrivate variable not accessible.",
            "red", attrs=["bold"]
        )
    )

# Private method error.
try:
    foo.__private_method()
except:
    print(
        colored(
            f"\nPrivate method not accessible.\n",
            "red", attrs=["bold"]
        )
    )


    