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
all: start init bash python close display clean end

# Displays starting message.
start:
	@echo "=========================================" | lolcat -h 0.05
	@echo -e "|\t\tINITIATE\t\t|" | lolcat -h 0
	@echo -e "=========================================\n\n\n" | lolcat -h 0.05

# Displays ending message.
end:
	@echo -e "\n\n\n=========================================" | lolcat -h 0.05
	@echo -e "|\t\tCOMPLETE\t\t|" | lolcat -h 0
	@echo "=========================================" | lolcat -h 0.05

# Displays the starting message.
init:
	@echo "----------SOURCE FILE EXECUTION----------" | lolcat -h 0.075

# Ending border.
close:
	@echo -e "-----------------------------------------\n" | lolcat -h 0.075

# Runs the BASH numbers file.
bash:
	@echo "Running numbers.sh..." | lolcat -h 0.25
	@chmod +x numbers.sh
	@bash numbers.sh | lolcat -h 0.25
	@echo "numbers.sh complete!" | lolcat -h 0.25

# Runs the Python numbers file.
python:
	@echo -e "\nRunning numbers.py..." | lolcat -h 0.25
	@python3 numbers.py | lolcat -h 0.25

# Display the number of lines in each file.
display:
	@echo -e "\n----------LINES & WORDS PER FILE---------" | lolcat -h 0.075 
	@echo "file1.txt: $$(wc -lw < $(file1))" | lolcat -h 0.25
	@echo "file2.txt: $$(wc -lw < $(file2))" | lolcat -h 0.25
	@echo "-----------------------------------------" | lolcat -h 0.075

# Remove old output files.
clean:
	@rm *.txt
