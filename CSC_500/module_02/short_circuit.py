x = 1
y = 0
z = 1

# Short circuits.
if(x == z or y == z):
    print("Since x == z, the statement short circuits and skips checking y.")
else:
    print("Short circuit failed.")

# Short circuits.
if(y == 0 and x == 1 or z == 1):
    print("Short circuits with y == 0 and x == 1. Skips checking z." 
          + "But still has to check both sides of the and statement")
else:
    print("Short circuit failed.")

# Does not short circuit.
if(x == 1 and z == 1):
    print("True, but does not short circuit.")
else:
    print("Something blew up.")

# Short circuits.
if(x == 0 and z == 1):
    print("Short circuits since x immediately makes the and statement evaluate to false.")
else:
    print("Short circuit failed.")

