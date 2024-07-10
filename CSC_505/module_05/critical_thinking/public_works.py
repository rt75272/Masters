import random
# ------------------------------------------------------------------ 
# Public Works.
#
# Builds up a work order object.
#
# Usage:
#   import public_works
#   example_var = public_works.public_works(repair_crew_id, 
#                                           num_people, 
#                                           equipment_used, 
#                                           hours_worked, 
#                                           hole_status, 
#                                           material_used)
# ------------------------------------------------------------------
class public_works:
    # Assign a random work id number.
    random_work_id = random.randint(0,99999)
    # Constructor function.
    def __init__(self, 
                 repair_crew_id, 
                 num_people, 
                 equipment_used, 
                 hours_worked, 
                 hole_status, # work-in-progress, repaired, temp_repair, or not_repaired.
                 material_used, # Amount of filler material used.
                 repair_cost=0, # hours_worked * num_people * material_used * equipment_used.
                 work_id=random_work_id):
        self.repair_crew_id = repair_crew_id         
        self.num_people = num_people 
        self.equipment_used = equipment_used 
        self.hours_worked = hours_worked 
        self.hole_status = hole_status   
        self.material_used = material_used        
        self.repair_cost = repair_cost
        self.work_id = work_id    
        self.get_repair_cost()                 

    # Calculate the total repair cost.
    def get_repair_cost(self):
        self.repair_cost = (
            self.hours_worked * 
            self.num_people * 
            self.material_used * 
            self.equipment_used)
        
        # Round the total cost to the nearest two decimals.
        self.repair_cost = round(self.repair_cost, 2)

        # Formatting the cost to have commas every 3 digits and a dollar sign.
        self.repair_cost = '{:,}'.format(self.repair_cost)
        self.repair_cost = "$" + str(self.repair_cost)
