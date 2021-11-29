#!/bin/bash

algs=(nes_rs nes_re deepens_rs)
nasalgs=(deepens_darts deepens_amoebanet deepens_darts_anchor)

#for m in {2,3,5,7,10,15}
#do
	#for alg in ${algs[@]}
	#do
		##scancel -n ${m}-${alg}
		#sbatch --bosch -J ${m}-${alg} -a 1-5 cluster_scripts/tiny/sbatch_scripts/evaluate_ensembles.sh $alg $m
		#echo ${m}-${alg}
	#done
	#for nasalg in ${nasalgs[@]}
	#do
		##scancel -n ${m}-${alg}
		#sbatch --bosch -J ${m}-${nasalg} -a 1 cluster_scripts/tiny/sbatch_scripts/evaluate_ensembles.sh $nasalg $m
		#echo ${m}-${nasalg}
	#done
#done

for m in {2,3,5,7,10,15}
do
	sbatch -J ${m}-dartsesa -a 1 -p alldlc_gpu-rtx2080 cluster_scripts/tiny/sbatch_scripts/evaluate_ensembles_nas.sh darts_esa $m
	sbatch -J ${m}-amoebaesa -a 1 -p alldlc_gpu-rtx2080 cluster_scripts/tiny/sbatch_scripts/evaluate_ensembles_nas.sh amoebanet_esa $m
	sbatch -J ${m}-dartsesa -a 3 -p alldlc_gpu-rtx2080 cluster_scripts/tiny/sbatch_scripts/evaluate_ensembles_nas.sh nes_rs_esa $m
done

