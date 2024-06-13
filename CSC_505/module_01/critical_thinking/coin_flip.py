from helper import *
# ---------------------------------------------------------
# Coin Flip Simulation
#
# Simulates and displays heads vs tails win ratios for n
# number of coin flips.
#
# Usage:
#   $ python coin_flip.py
#   $ Enter the number of simualtions(n)
# ---------------------------------------------------------

# Printer function displays heads vs tails win ratios.
def printer(heads_ratio, tails_ratio):
    print("Heads win ratio: " + str(heads_ratio) + "%")
    print("Tails win ratio: " + str(tails_ratio) + "%")

# Main driver function.
def main():
    n = int(input("Enter the number of simulations(n): "))
    starter = helper(0,0,n) # Calling the helper class.
    starter.simulate() # Run the simulation.
    ratios = starter.get_ratios() # Grabbing the ratios.
    heads_ratio = ratios[0] # Assigning the heads win ratio.
    tails_ratio = ratios[1] # Assigning the tail win ratio.
    printer(heads_ratio, tails_ratio) # Calling the printer function.

# Pushing the big red button.
if __name__=="__main__":
    main() # Fire the main function.