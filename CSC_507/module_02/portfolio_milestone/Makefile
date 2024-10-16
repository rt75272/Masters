# ---------------------------------------------------------------------------------------
# Makefile Number Generator Automation.
#
# Automates the process of running, storing, and displaying the outputs of the BASH and 
# Python number generators.
#
# Usage:
#	$ make <target>
# ---------------------------------------------------------------------------------------
# Store filenames in variables.
file1 = file1.txt # BASH output file.
file2 = file2.txt # Python output file.

# Runs all targets in order.
all: init bash python display clean

# Displays the starting message.
init:
	@echo "\nSOURCE FILE EXECUTION:" | lolcat

# Runs the BASH numbers file.
bash:
	@echo "\tRunning numbers.sh..." | lolcat
	@chmod +x numbers.sh
	@bash numbers.sh
	@echo "\t\tnumbers.sh complete!" | lolcat

# Runs the Python numbers file.
python:
	@echo "\tRunning numbers.py..." | lolcat
	@python3 numbers.py
	@echo "\t\tnumbers.py complete!" | lolcat

# Display the number of lines in each file.
display:
	@echo "\nLINES & WORDS PER FILE:" | lolcat 
	@echo "\tfile1.txt: $$(wc -lw < $(file1))" | lolcat
	@echo "\tfile2.txt: $$(wc -lw < $(file2))" | lolcat
	@echo ""

# Remove old output files.
clean:
	@rm *.txt

