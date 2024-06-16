from thompson import *
# -------------------------------------------------------------------
# Basic Testing
#
# Runs pytest on Thompson UML functions.
#
# Usage:
#   $ pytest
# -------------------------------------------------------------------
def test_construction():
    x = thompson()
    assert x.construction

def test_modeling():
    y = thompson()
    assert y.modeling

def test_deployment():
    z = thompson()
    assert z.deployment