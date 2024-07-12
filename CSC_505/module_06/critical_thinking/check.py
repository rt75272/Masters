
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

check_amount = 123456719
check_amount = str(check_amount)
n = len(str(check_amount))
print(n)

num_sections = math.ceil(n / 3)
print(num_sections)
print(check_amount[0])
print()
sections = []
for i in range(num_sections):
    section = []
    for j in range(3):
        print(i, j, (i+j+i*2))
        digit = check_amount[i+j+i*2]
        section.append(digit)
    sections.append(section)
    print(section)
    print()
        # print(check_amount[i*j])

print(sections)
print(num_sections)
print()

# print(type(int(sections[0][0])))

print()
print(ones[int(sections[0][0])])
print(upper[num_sections-1])
print(tens[int(sections[0][1])-2])
print(ones[int(sections[0][2])])

print()
print(ones[int(sections[1][0])])
print(upper[num_sections-2])
print(tens[int(sections[1][1])-2])
print(ones[int(sections[1][2])])

print()
print(ones[int(sections[2][0])])
print(upper[num_sections-3])
print(teens[int(sections[2][1])-2])
#print(ones[int(sections[2][2])])

# for i in range(num_sections):
#     for j in range(3):
