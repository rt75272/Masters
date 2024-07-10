# ------------------------------------------------------------------ 
# Makefile.
#
# Streamlines running multiple terminal commands.
#
# Usage:
#   $ make
#   # OR
#	$ make all
#	# OR
#   $ make report
#	OR
#	$ make clean
# ------------------------------------------------------------------
# Runs the main python file.
all:
	@python3 main.py

# Runs the main python file and display the damage report file.
report: all
	@echo "Printing Damage Report..."
	@cat *.txt

# Cleans up by removing old damage report files.
clean: all
	@rm *.txt
	@rm -rf __pycache__/
	@ls