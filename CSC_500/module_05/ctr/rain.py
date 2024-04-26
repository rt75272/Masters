#-----------------------------------------------------------------------
# Rainfall Stats
# 
# Module 5 Critical Thinking Assignment - Part 1 
#
# Asks the user for the number of years and monthly rainfall, then 
# calculates and displays the rainfall stats. 
#
# Usage:
#   $ python rain.py
#   [follow prompt to complete the program]
#-----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Gets user input to in order to build an array of rain data.
#
# Returns array of year's(s') rainfall 
# ----------------------------------------------------------------------
def get_rain_data(num_years):
    years = []
    # Dictionary of months and their respective rainfall amounts.
    monthly_rain = {
        "January" : 0,
        "Febuary" : 0,
        "March" : 0,
        "April" : 0,
        "May" : 0,
        "June" : 0,
        "July" : 0,
        "August" : 0,
        "September" : 0,
        "October" : 0,
        "November" : 0,
        "December" : 0
    }
    # Loading each month's rainfall amounts with user input.
    for i in range(num_years):
        for month in monthly_rain:
            monthly_rain[month] = int(input(f"Enter the inches of rainfall for {month}: "))
        years.append(monthly_rain) # Adds entire year of rain info as one entry each.
    return years

# ----------------------------------------------------------------------
# Total rainfall for all months recorded.
#
# Returns the total amount of rainfall.
# ----------------------------------------------------------------------
def total_rainfall(years, num_years):
    total_rain = 0
    for i in range(num_years):
        for month in years[i]:
            total_rain += years[i][month]
    return total_rain

# ----------------------------------------------------------------------
# Average rainfall over the course of all recorded months.
#
# Returns the average amount of rainfall per month.
# ----------------------------------------------------------------------
def avg_rainfall(total_rain, num_months):
    avg_rain = round((total_rain / num_months), 2)   
    return avg_rain

# ----------------------------------------------------------------------
# Final printer function. Outputs the final stats.
#
# Returns none.
# ----------------------------------------------------------------------
def printer(num_months, total_rain, avg_rain):
    print(f"\nNumber of months: {num_months}")
    print(f"Total inches of rainfall: {total_rain} inches")
    print(f"Average rainfall per month: {avg_rain} inches\n")
# ----------------------------------------------------------------------
# Main driver function.
#
# Returns none.
# ----------------------------------------------------------------------
def main():
    num_years = int(input("Enter the number of years: "))
    num_months = num_years * 12
    years = get_rain_data(num_years)
    total_rain = total_rainfall(years, num_years)
    avg_rain = avg_rainfall(total_rain, num_months)
    printer(num_months, total_rain, avg_rain)


# Pushing the big red button.
main()
