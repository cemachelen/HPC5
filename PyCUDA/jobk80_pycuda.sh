#$ -cwd -V
#$ -l h_rt=00:10:00
#$ -l coproc_k80=1
module switch cuda/9.0.176 cuda/8.0.61
module switch intel/17.0.1 gnu/native
module load python
module load python-libs

python  pycuda_test.py
