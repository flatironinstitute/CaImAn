#!/usr/bin/env python

import argparse
import filecmp
import os
import shutil
import string
import subprocess
import sys # for sys.prefix

from caiman.paths import caiman_datadir

sourcedir_base = os.path.join(sys.prefix, "share", "caiman") # Setuptools will drop our datadir off here

###############
# caimanmanager - A tool to manage the caiman install
#
# The caiman data directory is a directory, usually under the user's home directory
# but configurable with the CAIMAN_DATA environment variable, that is used to hold:
#   - sample movie data
#   - code samples
#   - misc json files used by Caiman libraries
#
# Usually you'll want to work out of that directory. If you keep upgrading Caiman, you'll
# need to deal with API and demo changes; this tool aims to make that easier to manage.

###############
# commands

def do_install_to(targdir):
	if os.path.isdir(targdir):
		raise Exception(targdir + " already exists")
	shutil.copytree(sourcedir_base, targdir)
	os.makedirs(os.path.join(targdir, 'memmap'), exist_ok=True) # Eventually refactor memmap code to put files here
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

def do_run_nosetests(targdir):
	out, err, ret = runcmd(["nosetests", "--traverse-namespace", "caiman"])
	if ret != 0:
		print("Nosetests failed with return code " + str(ret))
	else:
		print("Nosetests success!")

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

def runcmd(cmdlist, ignore_error=False, verbose=True):
        if verbose:
                print("runcmd[" + " ".join(cmdlist) + "]")
        pipeline = subprocess.Popen(cmdlist, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        (stdout, stderr) = pipeline.communicate()
        ret = pipeline.returncode
        if ret != 0 and not ignore_error:
                print("Error in runcmd: " + stderr)
                sys.exit(1)
        return stdout, stderr, ret

###############
def main():
	cfg = handle_args()
	if   cfg.command == 'install':
		do_install_to(cfg.userdir)
	elif cfg.command == 'check':
		do_check_install(cfg.userdir)
	elif cfg.command == 'test':
		do_run_nosetests(cfg.userdir)
	else:
		raise Exception("Unknown command")

def handle_args():
	parser = argparse.ArgumentParser(description="Tool to manage Caiman data directory")
	parser.add_argument("command", help="Subcommand to run. install/check/clean/tests")
	cfg = parser.parse_args()
	cfg.userdir = caiman_datadir()
	return cfg

###############

main()
