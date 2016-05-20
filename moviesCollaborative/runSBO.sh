END=$1
BEGIN=$2

for i in $(seq $BEGIN $END); do python SBOcorregional.py $i 50 5 200 T 10 ; done 
