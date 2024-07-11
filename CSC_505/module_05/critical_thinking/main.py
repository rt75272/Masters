import random
import autofill
import helper
# ------------------------------------------------------------------ 
# Main.
#
# Main driver file.
#
# Usage:
#   $ make report
# ------------------------------------------------------------------
# Main driver function.
def main():
    # Build a unique ticket number  and filename.
    ticket_number = random.randint(1000, autofill.UPPER_LIMIT)
    filename = helper.get_filename(ticket_number)

    # File to be the damage report.
    file = open(filename, "a")
    file.write("=======================================\n")
    file.write("-------------DAMAGE REPORT-------------\n")
    file.write("=======================================\n")

    # Build and assign a work ticket number.
    ticket_info = "\nTICKET NUMBER: "
    ticket_info += str(ticket_number) + "\n"
    file.write(ticket_info)

    # Build and assign class objects.
    person = autofill.get_person()
    hole = autofill.get_pothole()
    work_order = autofill.get_work_order()
    
    # Get all the variables from each class object.
    helper.write_attributes(person, file)
    helper.write_attributes(hole, file)
    helper.write_attributes(work_order, file)

    file.close()

# Pushing the big red button.
if __name__ == "__main__":
    main()
