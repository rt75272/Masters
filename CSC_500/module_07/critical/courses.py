import re
from termcolor import colored
# ----------------------------------------------------------------------------- 
# Course Look Up
#
# Allows the user to enter their desired course number and displays all the 
# available information on the course.
#
# Usage:
#   $ python courses.py
#   [follow prompt to complete program]
# -----------------------------------------------------------------------------

# Dictionary of room numbers
room_numbers = {
    "CSC101" : 3004,
    "CSC102" : 4501,
    "CSC103" : 6755,
    "NET110" : 1244,
    "COM241" : 1411
}

# Dictionary of instructors 
instructors = {
    "CSC101" : "Haynes",
    "CSC102" : "Alvarado",
    "CSC103" : "Rich",
    "NET110" : "Burke",
    "COM241" : "Lee"
}

# Dictionary of times.
times = {
    "CSC101" : "8:00 a.m.",
    "CSC102" : "9:00 a.m.",
    "CSC103" : "10:00 a.m.",
    "NET110" : "11:00 a.m.",
    "COM241" : "1:00 p.m."
}

# ----------------------------------------------------------------------
# Get Course Information
#
# Gets all the course values and displays in the terminal for the user. 
# Returns none.
# ----------------------------------------------------------------------
def get_course_info(course):
    room_number = room_numbers[course]
    instructor = instructors[course]
    time = times[course]
    print(
        colored(
            f"\n{course} is in room {room_number} with {instructor} at {time}\n", 
            "cyan", attrs=['bold']
        )
    )

# Grab the desired course number from the user.
course = input("\nWhich course would you like to look up?\n")
# Regex to locate CSC101-CSC103.
csc_course = re.findall("CSC10[1-3]", course)
# Grabbing the length of x incase the course in not found.
n = len(csc_course)

# Check n(the length of a found csc course) and then check the course number. 
if n > 0 and course == csc_course[0]:
    get_course_info(course)
# Check for the course NET110.
elif course == "NET110":
    get_course_info(course)
# Check for the course COM241.
elif course == "COM241":
    get_course_info(course)
# Else for no course found.
else:
    print(
        colored(
            f"\nThe course {course} was not found\n", "red", attrs=['bold']
        )
    )