#!/bin/bash

world_size=${1}
node_size=${2}
host="localhost"
host="env://"
num_cpu=${3}

job_id=$(qsub -terse -cwd -V  -l m_mem_free=12G -l h_rt=3:00:00 -l gpu=$node_size -l cuda_memory=35G -now no -pe smp ${num_cpu} -b yes torchrun --standalone --nnodes=$world_size --nproc_per_node=$node_size dinov2/run/train/train.py --config-file dinov2/configs/train/whoi.yaml --output-dir /fast/AG_Kainmueller/jrumber/PhD/plankton-dinov2/whoi_test )

sleep 3

echo $job_id

while true; do
    running=$(qstat -j ${job_id} -ext | grep "exec_host_list" | wc -l)
    if [[ "${running}" == 1 ]]
    then
        break
    fi
         sleep 3
done

host=$(qstat -j ${job_id} -ext | grep "exec_host_list" | tr -s "[:blank:]"  | cut -d " " -f 3 | cut -d ":" -f 1)

port=29500
host="${host}.mdc-berlin.net:${port}"


qsub -terse -cwd -V  -l m_mem_free=12G -l h_rt=3:00:00 -l gpu=$node_size -l cuda_memory=35G -now no -pe smp ${num_cpu} -b yes torchrun --standalone --nnodes=$world_size --nproc_per_node=$node_size dinov2/run/train/train.py --config-file dinov2/configs/train/whoi.yaml --output-dir /fast/AG_Kainmueller/jrumber/PhD/plankton-dinov2/whoi_test