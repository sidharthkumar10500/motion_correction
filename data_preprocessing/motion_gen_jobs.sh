#!/bin/bash
# bash script to recon multiple subject data in a single go
export ETL=18
export TE_effective=110
export TE_bw_echoes=10
for subject in {12..27}
do  
    echo "starting subject $subject "
    for slice in {20..40}
    do 
        echo "starting slice $slice"
        python motion_gen_FSE.py -sb $subject -sl $slice -etl $ETL -te $TE_effective -teb $TE_bw_echoes
    done
done
echo "All done"
