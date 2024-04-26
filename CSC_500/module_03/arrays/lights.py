from termcolor import colored
import curses
import time
#------------------------------------------------------------------------
# Practical lights array example.
#
# Usage:
#   $ python lights.py
#------------------------------------------------------------------------

# Setup terminal screen for light show.
stdscr = curses.initscr()

# Make cursor invisible for the light show.
curses.curs_set(0)

# Our terminal "lights".
lights = "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"

# Christmas light show colors array.
color_options = [
    "green",
    "white",
    "red"
]
print()

# Loop through color options and create the christmas themed light show.
for i in range(len(color_options)*9):
    option = color_options[i%3]
    print(colored(lights, option, attrs=["bold"]), end="\r",)
    time.sleep(0.5)

# Reset the terminal screen and cursor to normal functionality.
curses.curs_set(1)
curses.endwin()