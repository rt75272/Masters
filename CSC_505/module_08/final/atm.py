# -----------------------------------------------------------------------
# ATM Prototype 
#
# Simulates the basic functionality of logging into an ATM and checking 
# the account balance.
#
# Usage:
#   $ python atm.py
# -----------------------------------------------------------------------
# Main driver function.
def main():
    login_attempts = 0
    correct_pin = 1234
    balance = 1.42
    access_granted = False
    account_locked = False

    # Get and check user-given PIN.
    while (access_granted == False) and (account_locked == False):
        given_pin = int(input("Enter PIN: "))
        # Check for correct PIN.
        if(given_pin == correct_pin):
            access_granted = True
            print("Login successful")
        else:
            print("Incorrect PIN")
        # Check number of login attempts.
        if(login_attempts == 2 and access_granted != True):
            account_locked = True
            print("Too many failed attempts. Account locked.")
        # Increment login attempts.
        login_attempts+=1

    # Check bank account balance. 
    if(access_granted):
        print(f"Your available balance is ${balance}")
    else:
        print("Access denied")

# Pushing the big red button. 
if __name__ == "__main__":
    main()

