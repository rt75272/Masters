function get_rain_data:
    years = []
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
    for i in range(num_years):
        for month in monthly_rain:
            monthly_rain[month] = input("Enter rainfall amount")
        years.append(monthly_rain)
    return years

function total_rainfall:
    total_rain = 0
    for i in range(num_years):
        for month in years[i]:
            total_rain += years[i][month]
    return total_rain   

function avg_rainfall(total_rain, num_months):
    avg_rain = total_rain / num_months   
    return avg_rain

function printer:
    print("Number of months: {num_months}")
    print("Total inches of rainfall: {total_rain} inches")
    print("Average rainfall per month: {avg_rain} inches")

function main:
    num_years = input("Enter the number of years: ")
    num_months = num_years * 12
    years = get_rain_data(num_years)
    total_rain = total_rainfall(years, num_years)
    avg_rain = avg_rainfall(total_rain, num_months)
    printer(num_months, total_rain, avg_rain)

    