import random
# ------------------------------------------------------------------ 
# Helper.
#
# Helper functions for the main file.
#
# Usage:
#   $ import helper
# ------------------------------------------------------------------
# Generates and returns a random id number.
def get_id_number():
    id_number = random.randint(0, 99999)
    return id_number

# Builds and returns a unique filename. 
def get_filename(ticket_number):
    filename = "damage_report_"
    filename += str(ticket_number)
    filename += ".txt"
    return filename

# Writes to file a class object's variable names/values.
def write_attributes(class_object, file):
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
    file.write(output_string)