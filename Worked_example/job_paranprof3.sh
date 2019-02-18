#$ -cwd -V
#$ -l h_rt=00:10:00
#$ -l coproc_p100=1

module load cuda
./add_block.x
nvprof ./add_block.x
