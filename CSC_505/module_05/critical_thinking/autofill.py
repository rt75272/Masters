import citizen
import pothole
import public_works
import random
# ------------------------------------------------------------------ 
# Automatic Data Generator.
#
# Automatically builds and returns a given class object.
#
# Usage:
#   $ import auto_generate
# ------------------------------------------------------------------
UPPER_LIMIT = 9999
directions = [" North", " South", " East", " West"]

# Builds and return a citizen class object.
def get_person():
    # Assign a random name.
    names = ["Jane Doe", "John Doe"]
    name = names[random.randint(0, len(names)-1)]

    # Assign a random phone number.
    phone_number = "(" + str(random.randint(100, 999)) + ")"
    phone_number += str(random.randint(100, 999)) + "-"
    phone_number += str(random.randint(1000, UPPER_LIMIT))

    # Assign a random street address.
    personal_address = str(random.randint(100, UPPER_LIMIT))
    personal_address += directions[random.randint(0, len(directions)-1)]
    personal_address += " Street"

    # Load parameters into the citizen object.
    person = citizen.citizen(name,
                            phone_number, 
                            personal_address)
    return person

# Builds and return a pothole class object.
def get_pothole():
    # Assign a random pothole size.
    size = random.randint(1, 10)

    # Assign a pothole location.
    locations = ["curb", "middle", "etc"]
    location = locations[random.randint(0, len(locations)-1)]

    # Assign a random pothole street address.
    pothole_address = str(random.randint(100, UPPER_LIMIT))
    pothole_address += directions[random.randint(0, len(directions)-1)]
    pothole_address += " Street"

    # Load and build the pothole class object.
    hole = pothole.pothole(size,
                        pothole_address,
                        location)
    return hole

# Builds and returns a public works class object.
def get_work_order():
    # Assign a random work crew number.
    repair_crew_id = random.randint(0, UPPER_LIMIT)

    # Number of workers involved.
    num_people = random.randint(1, 25)

    # Equipment used with rate per hour.
    equipment_used = random.random()*100
    equipment_used = float('%.2f'%equipment_used)

    # Total number of hours worked fixing the pothole.
    hours_worked = random.randint(32, 40)

    # Assign a pothole status option.
    hole_status_options = ["work-in-progress", "repaired", "temporary fix", "not started"]
    hole_status = hole_status_options[random.randint(0, len(hole_status_options)-1)]

    # Total amount of filler material used.
    material_used = random.randint(1, 99) 

    # Load and build the public works object.
    work_order = public_works.public_works(repair_crew_id, 
                                        num_people, 
                                        equipment_used, 
                                        hours_worked, 
                                        hole_status, 
                                        material_used)
    return work_order