from termcolor import colored, cprint
# -------------------------------------------------------------------
# Thompson UML Class.
#
# Simulates a basic UML diagram 
#
# Usage:
#   from thompson import *
# -------------------------------------------------------------------
class thompson:
    # Constructor
    def __init__(self, data=""):
        self.data = data

    # Communication section.
    def communication(self):
        cprint("COMMUNICATION", "cyan", attrs=["bold"])
        project_initiation = "Project initiation"
        print(project_initiation)
        requirement_gather = int(input("Enter number of requirements: "))
        return requirement_gather

    # Planning section.
    def planning(self, requirements):
        cprint("\nPLANNING", "green", attrs=["bold"])
        # Number of requirements multiplied by 2 days(rough estimation). 
        estimating = requirements * 2

        # Broken up into 5 day work weeks.
        num_weeks = estimating / 5
        
        scheduling = "The project will be broken up into " + str(num_weeks) + " weeks" 
        print(scheduling)
        tracking = "Begin tracking the progress."
        print(tracking)

    # Modeling section.
    def modeling(self):
        cprint("\nMODELING", "red", attrs=["bold"])
        move_on = False
        design = "The design for the requirements has been presented."
        print(design)

        # Check for successful design review.
        design_review = input("Does the design checkout for the requirements?(y/n): ")
        if(design_review == "y"):
            print("Design checks out. Continue...")
            move_on = True
        else:
            print("Go back to the design phase.")
        return move_on

    # Construction section. 
    def construction(self):
        cprint("\nCONSTRUCTION", "magenta", attrs=["bold"])
        proceed = False
        code = "Code written. Time for peer review."
        print(code)

        # Check for successful peer validation.
        peer_validation = input("Does the code pass peer reviews?(y/n): ")
        if(peer_validation == "y"):
            print("Peer validation complete. Continue...")
            proceed = True
        else:
            print("Go back to the code.")
        return proceed
    
    # Deployment section.
    def deployment(self):
        cprint("\nDEPLOYMENT", "yellow", attrs=["bold"])
        delivery = "Code deployed."
        support = "Time to provide maintenance for the platform."
        feedback = "Also, gather user feedback."
        print(delivery)
        print(support)
        print(feedback)