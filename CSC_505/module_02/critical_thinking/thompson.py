class thompson:
    def __init__(self, data):
        self.data = data

    def communication(self):
        self.project_initiation = "Lets begin"
        self.requirement_gather = int(input("Enter number of requirements: "))

    def planning(self, requirements):
        # Number of requirements multiplied by 2 days(rough estimation). 
        self.estimating = requirements * 2
        num_weeks = self.estimating / 5
        self.scheduling = "The project will be broken up into {num_weeks} weeks" 
        self.tracking = "Are the requirements being completed on time?"

    def modeling(self):
        self.analysis = "Is it a good plan?"
        self.design = "The design for the requirements?"
        self.design_review = "Does the design checkout for the requirements?"

    def construction(self):
        self.code = ""
        self.test = "Running unit test"
        self.peer_validation = "Have others look over the code."
    
    def deployment(self):
        self.delivery = "Deploy the code"
        self.support = "Maintain the platform"
        self.feedback = "Gather user feedback"
        # self.tam = ""