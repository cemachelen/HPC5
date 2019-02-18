cp /nobackup/issmcal/learnCUDA.tgz .
tar -zxvf learnCUDA.tgz
cp /nobackup/issmcal/learnCUDAsolutions.tgz .
tar -zxvf learnCUDAsolutions.tgz
nvcc intro.cu -o ../intro.x
module load cuda
