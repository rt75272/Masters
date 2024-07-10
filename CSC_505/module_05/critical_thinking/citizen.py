import random

class citizen:
    random_citizen_id = random.randint(0,99999)
    def __init__(self, 
                 name, 
                 phone, 
                 citizen_address, 
                 citizen_id=random_citizen_id):
        self.name = name
        self.citizen_address = citizen_address
        self.phone = phone
        self.citizen_id = citizen_id
