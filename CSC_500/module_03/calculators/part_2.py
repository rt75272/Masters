#--------------------------------------------------------------------
# Part 2 - Alarm Calculator
#
# Usage:
#   $ python part_2.py
#   [follow prompt to complete program]
#--------------------------------------------------------------------

# Set the number of hours in a day as a constant.
HOURS_PER_DAY = 24

# Grabs the current time and desired alarm duration from the user.
# Returns the user specified time and alarm length as a list.
def get_user_input():
    current_time = int(input("What time is it(in hours)? "))
    alarm_duration = int(input("In how many hours would you like the alarm to go off? "))
    return current_time, alarm_duration

# Takes in the current time and alarm duration as parameters,
# then calculates and sets the alarm. Outputs the alarm time.
# Returns none.
def set_alarm(time, alarm_length):
    alarm_target = (time + alarm_length) % HOURS_PER_DAY
    alarm_target = str(alarm_target) + ":00"
    print("Your alarm will go off at", alarm_target, "hours")

# Main driver function.
# Returns none.
def main():
    # List containing the current time and the alarm duration.
    times = get_user_input()
    set_alarm(times[0], times[1])

# Pushing the big red button.
main()