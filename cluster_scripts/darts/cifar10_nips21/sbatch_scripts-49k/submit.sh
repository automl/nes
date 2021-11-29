#!/bin/bash

#algs=(nes_rs nes_re deepens_rs)
#nasalgs=(deepens_darts deepens_amoebanet deepens_darts_anchor)
nasalgs=(deepens_darts_anchor)

for m in {2,3,5,7,10,15,20,30}
do
	#for alg in ${algs[@]}
	#do
		##scancel -n ${m}-${alg}
		#sbatch --bosch -J ${m}-${alg} -a 1-3 cluster_scripts/cifar10/sbatch_scripts/evaluate_ensembles.sh $alg $m
		#echo ${m}-${alg}
	#done
	for nasalg in ${nasalgs[@]}
	do
		#scancel -n ${m}-${alg}
		sbatch -J ${m}-${nasalg} -a 1 cluster_scripts/cifar10/sbatch_scripts/evaluate_ensembles.sh $nasalg $m
		echo ${m}-${nasalg}
	done
done

