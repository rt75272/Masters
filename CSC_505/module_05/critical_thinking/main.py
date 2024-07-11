import citizen
import pothole
import public_works
import random
# ------------------------------------------------------------------ 
# Main.
#
# Main driver file.
#
# Usage:
#   $ make report
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

# Outputs a class object's variable names along with their values.
def get_attributes(class_object, file):
    # Grabbing the variables from their respective classes.
    class_keys = class_object.__dict__.keys()

    # Number of variables in a given class object.
    n = len(class_keys)
    
    # The output string will be built up and written to the damage report file.
    output_string = "\n"
    
    # Looping through all of a class object's variables/values.
    for i in range(n):
        # Grab the variable name from the class object.
        item = list(class_keys)[i]
        output_string += str(item) + ": "
    
        # Grab the value of the current variable.
        output_string += str(class_object.__getattribute__(item))
        output_string += "\n"
    
    # Display output string and write to file.
    # print(output_string)
    file.write(output_string)

# Main driver function.
def main():
    # Build a unique filename.
    ticket_number = random.randint(1000, UPPER_LIMIT)
    filename = "damage_report_"
    filename += str(ticket_number)
    filename += ".txt"

    # File to be the damage report.
    file = open(filename, "a")
    file.write("=======================================\n")
    file.write("-------------DAMAGE REPORT-------------\n")
    file.write("=======================================\n")

    # Build and assign a work ticket number.
    ticket_info = "\nTICKET NUMBER: "
    ticket_info += str(ticket_number) + "\n"
    # print(ticket_info)
    file.write(ticket_info)

    # Build and assign class objects.
    person = get_person()
    hole = get_pothole()
    work_order = get_work_order()
    
    # Get all the variables from each class object.
    get_attributes(person, file)
    get_attributes(hole, file)
    get_attributes(work_order, file)

    file.close()

# Pushing the big red button.
if __name__ == "__main__":
    main()
