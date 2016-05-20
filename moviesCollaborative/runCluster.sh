END=$1
BEGIN=$2

for i in $(seq $BEGIN $END); do jsub "progSBO.nbs $i" -mfail -email toscano.saul@gmail.com -xhost sys_pf -queue medium; done
