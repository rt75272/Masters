# --------------------------------------------------------------------------------------
# Makefile Automation.
#
# This Makefile automates the installation of dependencies, running the main script,
# and cleaning up temporary files.
#
# Usage:
#   make install  		# Install required Python packages.
#   make run      		# Run the main application.
#   make clean    		# Remove temporary files.
#   make all | make     # Run install, run, and clean in sequence.
# -------------------------------------------------------------------------------------- 
all: install run test clean

install:
	@pip install -r requirements.txt

run:
	@python main.py

test:
	@python run_tests.py --verbose --coverage

clean:
	@rm -rf __pycache__
	@rm -rf *.pyc