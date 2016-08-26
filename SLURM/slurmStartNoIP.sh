#!/bin/bash

[[ ${SLURM_NODELIST} ]] || { echo "SLURM environment not detected." ; exit 1 ; }

pdir=$(pwd)
profile="${SLURM_JOBID}_profile"
clog="ipcontroller_${SLURM_JOBID}.log"
#/mnt/xfs1/bioinfoCentos7/software/installs/python/2.7.10/bin/ipcontroller --profile=${profile} --ipython-dir=${pdir} --ip='*' > ${clog} 2>&1  &
#/mnt/xfs1/home/agiovann/anaconda2/envs/CNMF/bin/ipcontroller --profile=${profile} --ipython-dir=${pdir} --ip='*' > ${clog} 2>&1  &
/mnt/xfs1/home/agiovann/anaconda2/envs/CNMF/bin/ipcontroller --location=10.4.36.111 --profile=${profile} --ipython-dir=${pdir} --ip='*' > ${clog} 2>&1  &
cpid=$!

started=0
for d in 1 1 2 4
do
    grep -q 'Scheduler started' ${clog} && { started=1 ; break ; }
    sleep $d
done

[[ ${started} == 1 ]] || { echo "ipcontroller took too long to start. Exiting." ; exit 1 ; }

srun bash -c '/mnt/xfs1/home/agiovann/anaconda2/envs/CNMF/bin/ipengine  --profile='${profile}' --ipython-dir='${pdir}' > ipengine_${SLURM_JOBID}_${SLURM_PROCID}.log 2>&1 &'
#srun bash -c '/mnt/xfs1/bioinfoCentos7/software/installs/python/2.7.10/bin/ipengine  --profile='${profile}' --ipython-dir='${pdir}' > ipengine_${SLURM_JOBID}_${SLURM_PROCID}.log 2>&1 &'

started=0
for d in 2 2 4 4 8 8 8
do
    [[ $(grep 'Engine Connected:' ${clog} | wc -l) == ${SLURM_NTASKS} ]] && { started=1 ; break ; }
    sleep $d
done
[[ ${started} == 1 ]] || { echo "ipengines took too long to start. Exiting." ; exit 1 ; }

export IPPPDIR=${pdir} IPPPROFILE=${profile} 
