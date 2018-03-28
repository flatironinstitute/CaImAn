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
	# os.makedirs(os.path.join(targdir, 'memmap'), exist_ok=True) # Under discussion
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
		sys.exit(ret)
	else:
		print("Nosetests success!")

def do_run_demotests(targdir):
	out, err, ret = runcmd(["test_demos.sh"])
	if ret != 0:
		print("Demos failed with return code " + str(ret))
		sys.exit(ret)
	else:
		print("Demos success!")

def do_nt_run_demotests(targdir):
	# Windows platform can't run shell scripts, and doing it in batch files
	# is a terrible idea. So we'll do a minimal implementation of run_demos for
	# windows inline here.
	os.environ['MPLCONFIG'] = 'ps' # Not sure this does anything on windows
	demos = glob.glob('demos/general/*.py') # Should still work on windows I think
	for demo in demos:
		print("========================================")
		print("Testing " + str(demo))
		if "demo_behavior.py" in demo:
			print("  Skipping tests on " + demo + ": This is interactive")
		else:
			out, err, ret = runcmd(["python", demo], ignore_error=False)
			if ret != 0:
				print("  Tests failed with returncode " + str(ret))
				print("  Failed test is " + str(demo))
				sys.exit(2)
			print("===================================")
	print("Demos succeeded!")

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
                print("Error in runcmd: " + str(stderr))
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
	elif cfg.command == 'demotest':
		if os.name == 'nt':
			do_nt_run_demotests(cfg.userdir)
		else:
			do_run_demotests(cfg.userdir)
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
