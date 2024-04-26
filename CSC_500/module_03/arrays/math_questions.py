from random import random
from termcolor import colored
#------------------------------------------------------------------------
# Practical math array example.
#
# Usage:
#   $ python math_questions.py
#   [follow prompts to complete program]
#------------------------------------------------------------------------

# Math questions array.
math_questions = [
    "1 + 1",
    "1 - 1",
    "1 * 1",
    "1 / 1"
]

# Array of math question answers.
# Answer key array indexes corresponds to math questions array indexes.
answer_key = [
    2,
    0,
    1,
    1
]

# Loop through the math questions/answers until no more remain.
i = 1 # Question counter.
while(len(math_questions) > 0):
    print("\nQuestion(" + str(i) + "):\t", end="")

    # Randomly pick a math question from the array of questions.
    selection = int(random()*len(math_questions))

    # Output math question to terminal.
    print(math_questions[selection])

    # Get the user's answer to the question.
    user_answer = int(input("Your answer:\t"))

    # Check if user answered the math questions correctly or not.
    if(user_answer == answer_key[selection]):
        print(colored("Correct", "green"))
    else:
        print(colored("Incorrect", "red"))

    # Increment question counter.
    i+=1

    # Remove attempted math questions and their answers.
    math_questions.pop(selection)
    answer_key.pop(selection)

# Add some whitespace at the end.
print()
