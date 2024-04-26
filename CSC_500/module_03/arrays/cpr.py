import time
#------------------------------------------------------------------------
# Practical math array example
#
# Usage:
#   $ python cpr.py
#
# Reference:
#   https://drraghu.com/hearty-life/how-to-perform-cpr-the-right-way/
#------------------------------------------------------------------------

# Array of CPR steps.
cpr_steps = [
    "Step 1: Assess the situation",
    "Step 2: Check for responsiveness",
    "Step 3: Call for help",
    "Step 4: Open the airway",
    "Step 5: Check for breathing",
    "Step 6: Begin chest compressions",
    "Step 7: Give rescue breaths",
    "Step 8: Continue cycles of compressions and breaths",
    "Step 9: Don't interrupt CPR unless necessary",
    "Step 10: Reach out to a heart specialist"
]

# Output CPR steps to terminal for user to read.
for step in cpr_steps:
    print("\n", step)
    time.sleep(1)

# Add whitespace to end.
print()