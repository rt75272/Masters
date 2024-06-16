from thompson import *
from test_thompson import *

x = thompson()

requirements = x.communication()

x.planning(requirements)
x.construction()

# y = test_thompson()
# y.test_construction()

import subprocess
subprocess.run(["pytest", "test_thompson.py"])
print("Tests passed, continue on")