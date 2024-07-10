import citizen
import pothole
import public_works

# CITIZEN OBJECT 
name = "Tom Smith"
phone_number = "(720)142-4542"
personal_address = "42nd Street"
person = citizen.citizen(name,
                        phone_number, 
                        personal_address)

# POTHOLE OBJECT
size = 2
location = "middle"
pothole_address = "42nd St"
hole = pothole.pothole(size,
                       pothole_address,
                       location)

# PUBLIC WORKS OBJECT
repair_crew_id = 723 
num_people = 7 # Number of workers involved.
equipment_used = 299.99 # Equipment used cost $299.99 per hour.
hours_worked = 7 # Total number of hours worked fixing the pothole.
hole_status = "work-in-progress"
material_used = 99 # Total amount of filler material used.
work_order = public_works.public_works(repair_crew_id, 
                                       num_people, 
                                       equipment_used, 
                                       hours_worked, 
                                       hole_status, 
                                       material_used)

# Outputs a class object's variable names along with their values.
def get_attributes(class_object):
    # File to be the damage report.
    file = open("damage_report.txt", "a")
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
    file.close()

# Main driver function.
def main():
    get_attributes(person)
    get_attributes(hole)
    get_attributes(work_order)

# Pushing the big red button.
if __name__ == "__main__":
    main()
