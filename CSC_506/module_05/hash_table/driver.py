import hash_table
import random
import string
#------------------------------------------------------------------------------
# Main Driver
#
# Main driver file for the hash table. Generates and provides testing data.
#
# Usage:
#   $ python driver.py
#------------------------------------------------------------------------------
# Generate a list of random patient_id, patient_details data pairs.
def generate_random_data(num_items):
    data = {} # To be our dictionary of patient data.
    # Loop til the desired number of items has been met.
    for i in range(num_items):
        patient_id = f'PID{str(i).zfill(4)}' # Unique patient_id (e.g., PID0001, PID0002, etc.).
        name = f'Patient_{i}'  # Simple patient name using the incrementer. 
        age = random.randint(0, 100)  # Random age between 0 and 100.
        # Store patient data as a dictionary.
        data[patient_id] = {
            'name': name, 
            'age': age 
        }
    return data # Return generated patient data.

# Simulate real-ish world inserting, retrieving, and deleting of patient data from the hash table. 
def simulate_real_world_usage(hash_table, num_items):
    print("\n\tINIT PATIENT RECORDS\n")
    data = generate_random_data(num_items) # Grab our patient data.
    # Loop through all items and insert items into the hash table.
    for patient_id, details in data.items():
        hash_table.insert(patient_id, details) # Add patient data to the hash table.
        print(f'Inserted: {patient_id}: {details}') # Display the transaction.
    print() # Add some space for readability in the terminal. 
    # Retrieve and display a sample of patient data.
    for patient_id in random.sample(list(data.keys()), min(5, len(data))):
        print(f'Retrieved {patient_id}:', hash_table.get(patient_id)) # Display the data stored on the hash table.
    print() # Add some space for readability in the terminal. 
    # Delete patient records. 
    for patient_id in random.sample(list(data.keys()), min(3, len(data))):
        hash_table.delete(patient_id) # Remove patient data from the hash table.
        print(f'Deleted: {patient_id}') # Display the transaction.
    print() # Add some space for readability in the terminal. 

# Big red button.
if __name__ == "__main__":
    records = hash_table.HashTable(size=50) # Generate the hash table to store patient data.
    simulate_real_world_usage(records, 20) # Simulate with random patient entries.
    # Display the final set of patient records.
    print("\n\tFINAL PATIENT RECORDS\n")
    print(records)