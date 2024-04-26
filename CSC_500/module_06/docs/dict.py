"""""""""""""""""""""""""""""""""""""""""""""""
Lists VS Dictionaries

Comparing Python's lists and dictionaries

Usage:
  $ python dict.py
"""""""""""""""""""""""""""""""""""""""""""""""
# ---------------------------------------
# SETUP
# 
# Starting a list and dictionary with 
# the same data.
# ---------------------------------------

# Dictionary version.
# Better for "pair entries".
# Overkill for "single entries".
dictionary_example = {
    "bob" : "9436196",
    "frank" : "1234567",
    "joe" : "7654321" 
}

# An list version.
# Better for "standalone entries" and where order matters.
list_example_1 = ["bob", "frank", "joe"]
list_example_2 = ["9436196", "1234567", "7654321"]

# Another list version.
# Have to enter each one separately and in order.
list_example = ["bob","9436196","frank","1234567","joe","7654321"]

# ---------------------------------------
# INSERT
#
# Comparing the insert process between
# a list and a dictionary. 
# ---------------------------------------
# dictionary insert example.
dictionary_example["bobby"] = "7457384"

# list insert example.
list_example.append("bobby")
list_example.append("7457384")

# ---------------------------------------
# UPDATE
#
# Updating both data types with the same 
# data.
# ---------------------------------------
dictionary_example.update({"bobby" : "6457384"})
# Updating a dictionary key is a bit different.
dictionary_example["robert"] = dictionary_example.pop("bobby")

# list update example.
list_example[-2] = "robert"
list_example[-1] = "6457384" 

# ---------------------------------------
# REMOVE
#
# Taking off the last entry from each
# data type. 
# ---------------------------------------
dictionary_example.pop("robert")

list_example.remove("robert")
list_example.remove("6457384")

# Display our final list and dictionary.
print(dictionary_example)
print(list_example)