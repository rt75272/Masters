# ---------------------------------------------------------------------
# Software Developer Traits
#
# Builds and displays the traits of an excellent software developer.
#
# Usage:
#   $ python traits.py
# ---------------------------------------------------------------------

# Software developer class object.
class software_dev:
    # Constructor.
    def __init__(self, name="Bob"):
        self.traits = []
        self.name = name
    
    # Adds a trait to the software developer object.
    def add_trait(self, trait):
        self.traits.append(trait)
        
    # Prints the traits via terminal.
    def printer(self):
        print(f"Traits of our excellent software developer, {self.name}: ")
        for trait in self.traits:
            print("\t",trait)

# Main driver function.
def main():
    dev_name = input("Enter the software developer's name: ")
    # Calling the software developer object.
    dev = software_dev(dev_name)

    # Adding traits to the software developer object.
    traits = [
        "Strong foundation on algorithms and data structures.",
        "Ability to work in a diverse team.",
        "Consistent persistence."
        ]
    for trait in traits:
        dev.add_trait(trait)    

    # Display the software developer's traits.
    dev.printer()

# Pushing the big red button.
if __name__ == "__main__":
    main()
    