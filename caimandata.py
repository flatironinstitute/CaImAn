#!/usr/bin/env python

import argparse
import filecmp
import os
import shutil
import string
import sys # for sys.prefix

from caiman.paths import caiman_datadir

sourcedir_base = os.path.join(sys.prefix, "share", "caiman") # Setuptools will drop our datadir off here

###############
# caimandata - A tool to manage the caiman data directory for the current user
#
# The caiman data directory is a directory, usually under the user's home directory
# but configurable with the CAIMAN_DATA environment variable, that is used to hold:
#   - sample movie data
#   - code samples
#   - misc json files used by Caiman libraries
#   - temporary data files
#
# Usually you'll want to work out of that directory. If you keep upgrading Caiman, you'll
# need to deal with API and demo changes; this tool aims to make that easier to manage.

###############
# commands

def do_install_to(targdir):
	if os.path.isdir(targdir):
		raise Exception(targdir + " already exists")
	shutil.copytree(sourcedir_base, targdir)
	print("Installed " + targdir)

def do_check_install(targdir):
	ok = True
	comparitor = filecmp.dircmp(sourcedir_base, targdir)
	alldiffs = comparitor_all_diff_files(comparitor, '.')
	if alldiffs != []:
		print("These files differ: " + " ,".join(alldiffs))
		ok = False
	leftonly = comparitor_all_left_only_files(comparitor, ".")
	if leftonly != []:
		print("These files don't exist in the target: " + " ,".join(leftonly))
		ok = False
	if ok:
		print("OK")

###############
#

def comparitor_all_diff_files(comparitor, path_prepend):
	ret = list(map(lambda x: os.path.join(path_prepend, x), comparitor.diff_files)) # Initial
	for dirname in comparitor.subdirs.keys():
		to_append = comparitor_all_diff_files(comparitor.subdirs[dirname], os.path.join(path_prepend, dirname))
		if to_append != []:
			ret.append(*to_append)
	return ret

def comparitor_all_left_only_files(comparitor, path_prepend):
	ret = list(map(lambda x: os.path.join(path_prepend, x), comparitor.left_only)) # Initial
	for dirname in comparitor.subdirs.keys():
		to_append = comparitor_all_left_only_files(comparitor.subdirs[dirname], os.path.join(path_prepend, dirname))
		if to_append != []:
			ret.append(*to_append)
	return ret

###############
def main():
	cfg = handle_args()
	if   cfg.command == 'install':
		do_install_to(cfg.userdir)
	elif cfg.command == 'check':
		do_check_install(cfg.userdir)
	else:
		raise Exception("Unknown command")

def handle_args():
	parser = argparse.ArgumentParser(description="Tool to manage Caiman data directory")
	parser.add_argument("command", help="Subcommand to run. install/check/clean")
	cfg = parser.parse_args()
	cfg.userdir = caiman_datadir()
	return cfg

###############

main()
