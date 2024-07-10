import random
# ------------------------------------------------------------------ 
# Pothole.
#
# Builds up a pothole object.
#
# Usage:
#   import pothole
#   example_var = pothole.pothole(size,
#                                 pothole_address,
#                                 location)
# ------------------------------------------------------------------
class pothole:
    # Assign a random pothole id number.
    random_pothole_id = random.randint(0, 99999)
    
    # Constructor function.
    def __init__(self,  
                 size, # Scales from 1 to 10.
                 pothole_address,
                 location=0, # Middle, curb, etc.
                 district=0, # Determined by pothole_address.
                 repair_priority=0, # Determined by size.
                 pothole_id=random_pothole_id):
        self.pothole_id = pothole_id
        self.pothole_address = pothole_address
        self.size = size
        self.location = location
        self.district = district
        self.repair_priority = repair_priority
        self.set_repair_priority(size)
        self.set_district()

    # Set repair priority based on size.
    def set_repair_priority(self, size):
        self.size = float(size)
        if(self.size >= 7.5):
            self.repair_priority = "HIGH"
        elif(self.size < 7.5 and self.size >= 5.0):
            self.repair_priority = "MEDIUM-HIGH"
        elif(self.size < 5.0 and self.size >= 2.5):
            self.repair_priority = "MEDIUM"
        else:
            self.repair_priority = "LOW"
        
    # Helper incase pothole location was skipped.
    def set_pothole_location(self):
        if self.location == 0:
            self.location = input("Enter pothole location (middle, curb, etc.): ")

    # Assign a district number to a given pothole address.
    def set_district(self):
        n = len(self.pothole_address)
        self.district = random.randint(0, n)
