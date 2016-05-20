END=$1
BEGIN=$2

for i in $(seq $BEGIN $END); do jsub "createTraining.nbs $i" -mfail -email toscano.saul@gmail.com -xhost sys_pf -queue long; done
