#!/usr/bin/env ipython
import argparse
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.writers import StdoutWriter
import os

# Run a ipython notebook.
#
# This is based on http://nbconvert.readthedocs.io/en/latest/execute_api.html
#
# TODO: Make the notebook and the path scriptable

def handle_args():
	parser = argparse.ArgumentParser(description="Evaluate an ipython notebook")
	parser.add_argument("execpath", help="Working directory to use during evaluation")
	parser.add_argument("notebook", help="notebook to parse")
	parser.add_argument("--retain_x11", help="Whether to retain X11 for the tests", action='store_true')
	ret = parser.parse_args()
	return ret

def main():
	cfg = handle_args()
	if "DISPLAY" in os.environ and not cfg.retain_x11:
		del os.environ["DISPLAY"]
	os.environ["MPLCONFIG"] = "ps"

	with open(cfg.notebook) as f:
		nb = nbformat.read(f, as_version=4)

	# Prepare preprocessing engine
	ep = ExecutePreprocessor(timeout=6000)

	# Run it
	print("Running notebook")
	ep.preprocess(nb, {'metadata': {'path': cfg.execpath}})
	print("Preprocess done")

main()
