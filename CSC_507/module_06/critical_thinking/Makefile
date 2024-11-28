# ---------------------------------------------------------------------------------------
# Makefile Automation
#
# Drives the running of the source code and cleans up the directory.
#
# Usage:
#	$ make <target> <file> <n>
#	ex. 
#		$ make file=file2.txt n=5
# ---------------------------------------------------------------------------------------
file ?= file2.txt # input file.
n ?=  # Number of parts to break the input file up into.

# Main driver target.
all: run clean

# Runs the Python script with the command-line arguments.
run:
	@echo "Using input file: $(file)"
	@echo "Using $(n) processes"
	@python3 splitter.py $(n) $(file)

# Clean up the temporary files.
clean:
	@rm -f file_part_* processed_*
