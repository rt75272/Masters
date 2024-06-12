from random import random
# ---------------------------------------------------------
# Coin Flip Helper
#
# Provides the main coin flip file with the required 
# variables and functionality.
#
# Usage:
#   from helper import *
# ---------------------------------------------------------
class helper:
    # Constructor function.
    def __init__(self, heads_wins, tails_wins, n):
        self.heads_wins = heads_wins
        self.tails_wins = tails_wins
        self.n = n

    # Simulate n number of coin flips.
    # Increments heads wins and tails wins.
    def simulate(self):
        for i in range(self.n):
            x = random() * 10
            if(x < 5):
                self.heads_wins+=1
            elif(x > 5):
                self.tails_wins+=1

    # Ratio getter function. 
    # Returns heads winning ratio and tails winning ratio.
    def get_ratios(self):
        # Calculate and convert to percentage.
        heads_ratio = self.heads_wins / self.n * 100
        tails_ratio = self.tails_wins / self.n * 100

        # Round to four decimal places.
        heads_ratio = round(heads_ratio, 4)
        tails_ratio = round(tails_ratio, 4)

        # Return the ratios.
        return heads_ratio, tails_ratio