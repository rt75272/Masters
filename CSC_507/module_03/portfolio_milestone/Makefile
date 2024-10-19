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
all: start init bash python display clean end

# Displays starting message.
start:
	@echo "=============================================="
	@echo "\t\tINITIATE"
	@echo "=============================================="
	@echo "\n\n\n"

# Displays ending message.
end:
	@echo "\n\n\n"
	@echo "=============================================="
	@echo "\t\tCOMPLETE"
	@echo "=============================================="

# Displays the starting message.
init:
	@echo "------------SOURCE FILE EXECUTION------------" | lolcat

# Runs the BASH numbers file.
bash:
	@echo "Running numbers.sh..." | lolcat
	@chmod +x numbers.sh
	@bash numbers.sh | lolcat
	@echo "numbers.sh complete!" | lolcat

# Runs the Python numbers file.
python:
	@echo "\nRunning numbers.py..." | lolcat
	@python3 numbers.py | lolcat

# Display the number of lines in each file.
display:
	@echo "\n\n\n------------LINES & WORDS PER FILE-----------" | lolcat 
	@echo "file1.txt: $$(wc -lw < $(file1))" | lolcat
	@echo "file2.txt: $$(wc -lw < $(file2))" | lolcat

# Remove old output files.
clean:
	@rm *.txt

