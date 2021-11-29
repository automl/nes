#!/bin/bash

scheduler="cosine step"
layers="8 11 14"
channels="16 36 48"
lrs=$(awk 'BEGIN{for(i=0.025;i<=0.1;i*=2)print i}')


for sch in $scheduler; do
	for l in $layers; do
		for c in $channels; do
			for lr in $lrs; do
				sbatch cluster_scripts/tiny/eval_clip.sh $sch $l $c $lr
				echo submmited job $sch $l $c $lr
			done
		done
	done
done

