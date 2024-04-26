room_numbers = {
    "CSC101" : 3004,
    "CSC102" : 4501,
    "CSC103" : 6755,
    "NET110" : 1244,
    "COM241" : 1411
}

instructors = {
    "CSC101" : "Haynes",
    "CSC102" : "Alvarado",
    "CSC103" : "Rich",
    "NET110" : "Burke",
    "COM241" : "Lee"
}

times = {
    "CSC101" : "8:00 a.m.",
    "CSC102" : "9:00 a.m.",
    "CSC103" : "10:00 a.m.",
    "NET110" : "11:00 a.m.",
    "COM241" : "1:00 p.m."
}

function get_course_info(course)
    room_number = room_numbers[course]
    instructor = instructors[course]
    time = times[course]

course = input("Which course would you like to look up?")


