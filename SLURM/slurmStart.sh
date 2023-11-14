#!/bin/bash

[[ ${SLURM_NODELIST} ]] || { echo "SLURM environment not detected." ; exit 1 ; }

pdir=$(pwd)
profile="${SLURM_JOBID}_profile"
clog="ipcontroller_${SLURM_JOBID}.log"

ipcontroller --location=$(hostname -i) --profile=${profile} --ipython-dir=${pdir} --ip='*' > ${clog} 2>&1  &
cpid=$!

started=0
for d in 1 1 2 4
do
    grep -q 'Scheduler started' ${clog} && { started=1 ; break ; }
    sleep $d
done

[[ ${started} == 1 ]] || { echo "ipcontroller took too long to start. Exiting." ; exit 1 ; }

srun bash -c 'ipengine  --profile='${profile}' --ipython-dir='${pdir}' > ipengine_${SLURM_JOBID}_${SLURM_PROCID}.log 2>&1' &

started=0
for d in 2 2 4 4 8 8 8
do
    [[ $(grep 'engine::Engine Connected: ' ${clog} | wc -l) == ${SLURM_NTASKS} ]] && { started=1 ; break ; }
    grep 'engine::Engine Connected: ' ${clog}
    sleep $d
done
[[ ${started} == 1 ]] || { echo "ipengines took too long to start. Exiting." ; exit 1 ; }

export IPPPDIR=${pdir} IPPPROFILE=${profile} 

