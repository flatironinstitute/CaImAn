# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:07:09 2016

@author: agiovann
"""
import subprocess
import time
import ipyparallel
from ipyparallel import Client
import shutil
import glob
import shlex
import psutil
import sys
import os
#%%
def start_server(slurm_script=None, ipcluster="ipcluster"):
    '''
    programmatically start the ipyparallel server

    Parameters
    ----------
    ncpus: int
        number of processors
    ipcluster : str
        ipcluster binary file name; requires 4 path separators on Windows. ipcluster="C:\\\\Anaconda2\\\\Scripts\\\\ipcluster.exe"
         Default: "ipcluster"    
    '''
    sys.stdout.write("Starting cluster...")
    sys.stdout.flush()
    ncpus=psutil.cpu_count()
    
    if slurm_script is None:
        if ipcluster == "ipcluster":
            p1 = subprocess.Popen("ipcluster start -n {0}".format(ncpus), shell=True, close_fds=(os.name != 'nt'))
        else:
            p1 = subprocess.Popen(shlex.split("{0} start -n {1}".format(ipcluster, ncpus)), shell=True, close_fds=(os.name != 'nt'))
#
        while True:
            try:
                c = ipyparallel.Client()
                if len(c) < ncpus:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                    raise ipyparallel.error.TimeoutError
                c.close()                

                break
            except (IOError, ipyparallel.error.TimeoutError):
                sys.stdout.write(".")
                sys.stdout.flush()
                time.sleep(1)
                
    else:
        shell_source(slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)
        ee = c[:]
        ne = len(ee)
        print 'Running on %d engines.' % (ne)
        c.close()
        sys.stdout.write(" done\n")
        

#%%


def shell_source(script):
    """Sometime you want to emulate the action of "source" in bash,
    settings some environment variables. Here is a way to do it."""


    pipe = subprocess.Popen(". %s; env" % script,  stdout=subprocess.PIPE, shell=True)
    output = pipe.communicate()[0]
    env = dict()
    for line in output.splitlines():
        lsp = line.split("=", 1)
        if len(lsp) > 1:
            env[lsp[0]] = lsp[1]
#    env = dict((line.split("=", 1) for line in output.splitlines()))
    os.environ.update(env)
    pipe.stdout.close()
#%%


def stop_server(is_slurm=False, ipcluster='ipcluster',pdir=None,profile=None):
    '''
    programmatically stops the ipyparallel server
    Parameters
     ----------
     ipcluster : str
         ipcluster binary file name; requires 4 path separators on Windows
         Default: "ipcluster"
    '''
    sys.stdout.write("Stopping cluster...\n")
    sys.stdout.flush()

    if is_slurm:
        
        if pdir is None and profile is None:
            pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)
        ee = c[:]
        ne = len(ee)
        print 'Shutting down %d engines.' % (ne)
        c.shutdown(hub=True)
        shutil.rmtree('profile_' + str(profile))
        try:
            shutil.rmtree('./log/')
        except:
            print 'creating log folder'

        files = glob.glob('*.log')
        os.mkdir('./log')

        for fl in files:
            shutil.move(fl, './log/')

    else:
        if ipcluster == "ipcluster":
            proc = subprocess.Popen("ipcluster stop", shell=True, stderr=subprocess.PIPE, close_fds=(os.name != 'nt'))
        else:
            proc = subprocess.Popen(shlex.split(ipcluster + " stop"),
                                    shell=True, stderr=subprocess.PIPE, close_fds=(os.name != 'nt'))

        line_out = proc.stderr.readline()
        if 'CRITICAL' in line_out:
            sys.stdout.write("No cluster to stop...")
            sys.stdout.flush()
        elif 'Stopping' in line_out:
            st = time.time()
            sys.stdout.write('Waiting for cluster to stop...')
            while (time.time() - st) < 4:
                sys.stdout.write('.')
                sys.stdout.flush()
                time.sleep(1)
        else:
            print line_out
            print '**** Unrecognized Syntax in ipcluster output, waiting for server to stop anyways ****'
        
        proc.stderr.close()
        
    sys.stdout.write(" done\n")