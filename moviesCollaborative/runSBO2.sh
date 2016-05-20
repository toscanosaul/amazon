END=$1
BEGIN=$2

for i in $(seq $BEGIN $END); do python createTrainingPoints.py $i 50; done
