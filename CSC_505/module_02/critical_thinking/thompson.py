from test_thompson import *

class thompson:
    def __init__(self, data=""):
        self.data = data

    def communication(self):
        project_initiation = "Lets begin"
        print(project_initiation)
        requirement_gather = int(input("Enter number of requirements: "))
        return requirement_gather

    def planning(self, requirements):
        # Number of requirements multiplied by 2 days(rough estimation). 
        estimating = requirements * 2
        num_weeks = estimating / 5
        scheduling = "The project will be broken up into " + str(num_weeks) + " weeks" 
        print(scheduling)
        tracking = "Begin tracking the progress"
        # if(tracking == "y"):
        #     print("continue")
        # else:
        #     print("restart")


    def modeling(self):
        analysis = input("Is it a good plan?(y/n)")
        if(analysis == "y"):
            print("Continue")
        else:
            print("Restart")
        design = "The design for the requirements has been presented."
        design_review = input("Does the design checkout for the requirements?(y/n)")

    def construction(self):
        code = "Code written. Time for testing..."
        test = "Running unit test..."
        # insert pytest module
        peer_validation = input("Do your peers approve the code?(y/n)")
        return 42
    
    def deployment(self):
        delivery = "Deploy the code"
        support = "Maintain the platform"
        feedback = "Gather user feedback"
        # self.tam = ""
    
