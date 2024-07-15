import random
import math

ones = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine"
]

teens = [
    "ten"
    "eleven", 
    "twelve", 
    "thirteen", 
    "fourteen", 
    "fifteen", 
    "sixteen", 
    "seventeen", 
    "eighteen", 
    "nineteen"
]

tens = [
    "twenty", 
    "thirty", 
    "forty", 
    "fifty", 
    "sixty", 
    "seventy", 
    "eighty", 
    "ninety"
]

upper = [
    "hundred",
    "thousand",
    "million",
    "billion"
]

def get_check():
    check_amount = random.randint(100000000, 999999999)
    # check_amount = 804191014
    check_amount = str(check_amount)
    print(check_amount)
    return check_amount

check_amount = get_check()
n = len(str(check_amount))

# Setting up 3-digit sections.
num_sections = math.ceil(n / 3)

# Feed the check amount into the n*3 sections matrix.
sections = []
for i in range(num_sections):
    section = []
    for j in range(3):
        digit = check_amount[i+j+i*2]
        section.append(digit)
    sections.append(section)

# Display the money matrix.
for section in sections:
    print(section)

# Establish the base upper value.
for i in range(num_sections):
    for j in range(3):
        # Grab value of the digit in question.
        digit = int(sections[i][j])
        print(digit)
        # Check if not equal to 0.
        if digit != 0 and j == 0:
            # Output hundred.
            print(upper[0])

        # Initial check.
        # Column 1.
        if j == 0:
            # Millions section.
            if i == 0:
                # print(upper[0])

                print(upper[2])
            # Thousands section.
            if i == 1:
                # print(upper[0])
                print(upper[1])
            # if i == 2:
            #     print(upper[0])
        
        # Column 2
        if j == 1 and digit != 0:
            if digit != 1:
                print(tens[digit-2])
            # else:
            #     print(teens[digit])
        
        # Column 3
        if j == 2 and digit != 0:
            print(ones[digit])
    
    print()


# # Build up the output string.
# total_amount = ""
# teen = False
# zero = False
# for i in range(3):
#     for j in range(3):
#         # Check if the first digit of a row is zero.
#         if j==0:
#             zero_test = int(sections[i][j])
#             if zero_test == 0:
#                 zero = True
#         # Check the first value of each section/row.
#         if j==0 and i != 1:
#             # Don't add values from upper array for a value of zero.
#             if zero == False:
#                 total_amount += ones[int(sections[i][j])] + " "
#                 total_amount += upper[num_sections-(i+1)] + " "
#         # Check for hundred thousands.
#         if i == 1 and j == 0:
#             total_amount += ones[int(sections[i][j])] + " "
#             total_amount += upper[num_sections-(i+2)] + " "
#             total_amount += upper[num_sections-(i+1)] + " "
#         # Check for a teens value by checking for the value of 1 in the tens place.
#         elif j==1:
#             base = int(sections[i][j])
#             # If a teen value.
#             if base == 1:
#                 total_amount += teens[int(sections[i][j])+2] + " "
#                 teen = True
#             # If not a teen value.
#             else:
#                 total_amount += tens[int(sections[i][j])-2] + " "
#         # Add the final digit if it is not part of a teen value.
#         elif j==2 and teen != True and int(sections[i][j]) != 0:
#             total_amount += ones[int(sections[i][j])] + " "
        
# total_amount += "dollars" 


# print(total_amount)

