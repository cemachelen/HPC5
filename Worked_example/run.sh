module load cuda
nvcc add.cu -o add_cuda.x
qsub job.sh
