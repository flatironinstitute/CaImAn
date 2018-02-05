#!/bin/bash

# TODO: Set pythonpath to wherever this was unpacked
##########################
# test_demos.sh
#
# This is intended to run all the python (.py) demos in the
# codebase for testing. Jenkins will eventually invoke this.
#
# dependencies:
#   xvfb-run - This starts a dummy X server for anything that might
#              need to plot things or show movies. Note that any demos
#              we test MUST NOT wait for input while showing a movie,
#              because in the test environment nobody will show up to provide
#              that input (like clicking things or hitting q)

# TODO: Enter appropriate conda environment, unless my wrapper script does that

# Make sure the xvfb-run command exists before starting demos, so we can give a better
# error message
command -v xvfb-run
if [ $? != 0 ]; then
	echo "xvfb-run command not found"
	exit 1
fi

# Tell matplotlib to try to plot less to begin with by specifying a postscript backend
export MPLCONFIG=ps

for demo in demos/general/*; do
	if [ $demo == "demos/general/demo_behavior.py" ]; then
		echo "	Skipping tests on $demo: This is interactive"
	elif [ -d $demo ]; then
		true
	else
		echo Testing demo [$demo]
		xvfb-run -s "-screen 0 800x600x16" python $demo
		err=$?
		if [ $err != 0 ]; then
			echo "	Tests failed with returncode $err"
			echo "	Failed test is $demo"
			exit 2
		fi
		echo "=========================================="
	fi
done
