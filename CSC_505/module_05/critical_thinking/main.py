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
#   $ python main.py
# ------------------------------------------------------------------
# CITIZEN OBJECT 
# Assign a random name.
names = ["Jane Doe", "John Doe"]
name = names[random.randint(0,1)]

# Assign a random phone number.
phone_number = "(" + str(random.randint(111, 999)) + ")"
phone_number += str(random.randint(111,999)) + "-"
phone_number += str(random.randint(1111,9999))

# Assign a random street address.
personal_address = str(random.randint(111,9999))
personal_address += " North Street"

# Load parameters into the citizen object.
person = citizen.citizen(name,
                         phone_number, 
                         personal_address)

# POTHOLE OBJECT
# Assign a random pothole size.
size = random.randint(0,9)

# Assign a pothole location.
locations = ["curb", "middle", "etc"]
location = locations[random.randint(0,2)]

# Assign a random pothole street address.
pothole_address = str(random.randint(111,9999))
pothole_address += " North Street"

# Load and build the pothole class object.
hole = pothole.pothole(size,
                       pothole_address,
                       location)

# PUBLIC WORKS OBJECT
# Assign a random work crew number.
repair_crew_id = random.randint(0, 9999)

# Number of workers involved.
num_people = random.randint(1,25)

# Equipment rate per hour.
equipment_used = random.random()*100
equipment_used = float('%.2f'%equipment_used)

# Total number of hours worked fixing the pothole.
hours_worked = random.randint(32, 40)

# Assign a pothole status option.
hole_status_options = ["work-in-progress", "repaired", "temporary fix", "not started"]
hole_status = hole_status_options[random.randint(0,3)]

# Total amount of filler material used.
material_used = random.randint(1, 99) 

# Load and build the public works object.
work_order = public_works.public_works(repair_crew_id, 
                                       num_people, 
                                       equipment_used, 
                                       hours_worked, 
                                       hole_status, 
                                       material_used)

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
        x = list(class_keys)[i]
        output_string += str(x) + " : "
    
        # Grab the value of the current variable.
        output_string += str(class_object.__getattribute__(x))
        output_string += "\n"
    
    # Display output string and write to file.
    print(output_string)
    file.write(output_string)

# Main driver function.
def main():
    # Build a unique filename.
    ticket_number = random.randint(1111, 9999)
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
    print(ticket_info)
    file.write(ticket_info)
    
    # Get all the variables from each class object.
    get_attributes(person, file)
    get_attributes(hole, file)
    get_attributes(work_order, file)

    file.close()

# Pushing the big red button.
if __name__ == "__main__":
    main()
