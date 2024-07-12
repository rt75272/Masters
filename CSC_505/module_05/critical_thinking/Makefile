# ------------------------------------------------------------------ 
# Makefile.
#
# Streamlines running multiple terminal commands.
#
# Usage:
#   $ make <target>
# ------------------------------------------------------------------
# Gets the most recent damage report txt filename.
x := $$(ls *.txt | head -1)

# Runs the main python file.
all:
	@python3 main.py

# Runs the main python file and display the damage report file.
report: all
	@echo "Printing Damage Report..."
	@cat $x

# Cleans up by removing old damage report files.
clean: all
	@rm *.txt
	@rm -rf __pycache__/
	@ls -atloghA1