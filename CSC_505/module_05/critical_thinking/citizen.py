import random
# ------------------------------------------------------------------ 
# Citizen.
#
# Builds up a citizen object.
#
# Usage:
#   import citizen
#   example_var = citizen.citizen(name,
#                                 phone,
#                                 citizen_address)
# ------------------------------------------------------------------
class citizen:
    # Assign a random citizen id number.
    random_citizen_id = random.randint(0,99999)
    # Constructor function.
    def __init__(self, 
                 name, 
                 phone, 
                 citizen_address, 
                 citizen_id=random_citizen_id):
        self.name = name
        self.citizen_address = citizen_address
        self.phone = phone
        self.citizen_id = citizen_id
