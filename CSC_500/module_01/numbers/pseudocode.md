<!--------------------------
Pseudocode for Part 1 & 2
--------------------------->


function get_user_input()
    print "Enter two numbers"
    num1 = input()
    num2 = input()
    return num1, num2

function add(num1, num2)
    sum = num1 + num2
    return sum

function subtract(num1, num2)
    difference = num1 - num2
    return difference

function multiply(num1, num2)
    product = num1 * num2
    return product

function divide(num1, num2)
    quotient = num1 / num2
    return quotient

function main()
    num1, num2 = get_user_input()
    sum = add(num1, num2)
    difference = subtract(num1, num2)
    product = multiply(num1, num2)
    quotient = divide(num1, num2)
    print "num1 + num2 = sum"
    print "num1 - num2 = difference"
    print "num1 * num2 = product"
    print "num1 / num2 = quotient"


