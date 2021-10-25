#!/bin/bash
for target_scan in 1 2 3 4 5 6 7 8 9 10
do
    for mini_contrast in 't2' 't1' #10 11 13
    do
        python realnoisedata_gen.py -m $mini_contrast -t $target_scan 
    done
done
echo "All done"