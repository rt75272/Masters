import subprocess
from thompson import *
# -------------------------------------------------------------------
# Driver for Thompson class and testing.
#
# Runs and tests the Thompson UML class.
#
# Usage:
#   $ python driver.py
# -------------------------------------------------------------------

# Pytest function
def tester(x):
    print("\nRunning unit tests...")
    y = subprocess.run(["pytest", "test_thompson.py"])
    if y.returncode == 0:
        print("Tests passed. Continue to deployment.")
        x.deployment()
    else:
        print("Test(s) failed. Revisit the code.")

# Main driver function
def main():
    x = thompson()
    requirements = x.communication()
    x.planning(requirements)

    # Check for successful modeling before moving on to the construction section.
    model = x.modeling()
    construction = False
    if(model == True):
        construction = x.construction()
    else:
        while(model == False):
            model = x.modeling()
        if(model == True):
            construction = x.construction()

    # Check for successful construction before moving on to deployment.
    if(construction == True):
        tester(x)
    else:
        while(construction == False):
            construction = x.construction()
        if(construction == True):
            tester(x)    

if __name__ == "__main__":
    main()