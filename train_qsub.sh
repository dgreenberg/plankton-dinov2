#$ -l gpu=1
#$ -l cuda_memory=45G
#$ -pe smp 4
#$ -cwd
#$ -V
#$ -e /home/nkoreub/output_dir/log_$JOB_ID.err
#$ -o /home/nkoreub/output_dir/log_$JOB_ID.out
#$ -l h_rt=150:00:00
#$ -A kainmueller

N_GPUS=1
echo $JOB_ID
torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS dinov2/run/train/train.py --ngpus $N_GPUS --config-file dinov2/configs/train/whoi.yaml --run_name=${JOB_ID}_${N_GPUS}gpu_norastestrun train.augmentations=kornia_gpu train.output_dir=/home/nkoreub/output_dir train.use_torch_compile=true train.dataset_path=LMDBDataset:split=ALL:root=/fast/AG_Kainmueller/plankton/data/WHOI/preprocessed_hdf5/lmdb/:extra=*


retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Error"
    exit 100
fi
exit 0