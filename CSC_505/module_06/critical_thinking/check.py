# -----------------------------------------------------------------------------
# Numbers To Text
#
# Takes in a numerical check amount and converts the numeric amount into a 
# string representation. 
#
# Usage:
#   $ python check.py
#   [follow prompt to complete program]
# -----------------------------------------------------------------------------
# Set the maximum allowed digit to one trillion.
MAX = 1000000000000

# Main driver function.
def main():
    # Loop til a valid input is entered.
    while True:
        try:
            # Get the check amount.
            amount = float(input("Enter the check amount: "))
            
            # Check for a positive value.
            if amount <= 0:
                raise ValueError("Amount must be greater than zero.")
            # Check for numbers over a trillion.
            elif amount > MAX:  
                raise ValueError("Amount cannot exceed a trillion.")
            break
        # Catch invalid input data types.
        except ValueError as ex:
            print("Invalid input type:", ex)

    # Break up into dollars and cents.
    dollars, cents = divmod(amount, 1)
    dollars = int(dollars)
    cents = round(cents * 100)

    # Get the words for both dollars and cents.
    dollar_words = get_words(dollars)
    cent_words = get_words(cents)

    # Grab and display the cleaned up final output.
    final_output = clean_output(dollar_words, cent_words)
    print(final_output)

# Converts and returns numbers into words using a stepwise approach.
def get_words(digit):
    lower = [
        "zero", 
        "one", 
        "two", 
        "three", 
        "four", 
        "five", 
        "six", 
        "seven", 
        "eight", 
        "nine", 
        "ten",
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
        "",
        "", 
        "twenty", 
        "thirty", 
        "forty", 
        "fifty", 
        "sixty", 
        "seventy", 
        "eighty", 
        "ninety"
    ]
    large_base = ["", "thousand", "million", "billion"]

    # Check for lower numbers, below 20.
    if digit < 20:
        ret_val = lower[digit]
        return ret_val
    
    # Check for tens base.
    elif digit < 100:
        # Get the tens base value of the digit.
        base_value = digit // 10

        # Assign the tens using the base value.
        ret_val = tens[base_value]

        # Check for a remainder. 
        if digit % 10 != 0:
            # Grab the lower digit with the remainder.
            ret_val += " " + lower[digit % 10]
        return ret_val
    
    # Check for hundreds.
    elif digit < 1000:
        # Get the hundreds base value of the digit.
        base_value = digit // 100

        # Assign the hundreds using the base value.
        ret_val = lower[base_value]
        ret_val += " hundred " 

        # Check for a remainder. 
        if digit % 100 != 0:
            # Call the function again with the remainder.
            ret_val += get_words(digit % 100)
        return ret_val
    
    # Check for thousands or higher.
    else:
        for index, element in enumerate(large_base):
            # Check for less than a thousand, then a million, then a billion.
            if digit < 1000 ** (index + 1):
                # Call the function again with the largest base value.
                ret_val = get_words(digit // (1000 ** index))
                # Assign the words.
                ret_val += " " + element + " "
                # Check for a remainder. 
                if digit % (1000 ** index) != 0:
                    # Call the function again with the remainder.
                    ret_val += get_words(digit % (1000 ** index))
                return ret_val
        return "Maximum size exceeded"

# Clean up and return the final output. 
def clean_output(dollars, cents):
    dollars = dollars.capitalize()
    ret_val = f"{dollars} dollars and {cents} cents."
    return ret_val

# Pushing the big red button.
if __name__ == "__main__":
    main()
