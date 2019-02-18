source .env
qrsh -l ports=1,coproc_p100=1,h_rt=1:0:0 -pty y /bin/bash -i
# or 
qrsh -l ports=1,coproc_k80=1,h_rt=1:0:0 -pty y /bin/bash -i
