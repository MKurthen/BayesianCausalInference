#!/bin/bash
#
# Copyright (c) 2008-2015  Joris Mooij  <j.m.mooij@uva.nl>
# All rights reserved.  See the file LICENSE for license terms.

function runmatlab {
	matlab -nodisplay -nojvm -singleCompThread -r "pair=$1,addpath([pwd '/$2']),experiment,exit" < /dev/null > $2/$1.stdout 2> $2/$1.stderr
}
export -f runmatlab

PARALLEL=25

DATADIRS='CEP SIM SIM-c SIM-ln SIM-G'

#######

for DATADIR in $DATADIRS; do

	if [ 0 == 1 ]; then
		# compare Nmax for cep_anm
		for Nmax in 100 200 500 1000; do
			methodstr=("pHSIC" "HSIC" "entropy" "entropy" "MML")
			enteststr=("" "" "Shannon_KDP" "Shannon_PSD_SzegoT" "")
			for method in 0 1 2 3 4; do
				for FITC in 0 100; do
					dirmethod=${methodstr[$method]}_${enteststr[$method]}
					OUTDIR="$DATADIR""_out/anm_N=$Nmax""_FITC=$FITC""_$dirmethod""_lbfgsb"
					mkdir -p $OUTDIR
					echo "Nmax=$Nmax, rep=1, outdir='$OUTDIR', datadir='$DATADIR', pp=struct; pp.randseed=pair*rep; pp.undisc=0; pp.disc=0; pp.maxN=Nmax; pp.disturbance=0; mp=struct; mp.nrperm=0; mp.gaussianize=0; mp.FITC=$FITC; mp.splitdata=0; mp.evaluation='${methodstr[$method]}'; mp.meanf='meanConst'; mp.minimize='minimize_lbfgsb'; mp.bandwidths=[0,0]; mp.entest='${enteststr[$method]}'; run_method (pair,'cep_anm',mp,sprintf('%s/%d',outdir,rep),pp,datadir);" > $OUTDIR/experiment.m
					parallel -j $PARALLEL runmatlab {1} $OUTDIR ::: `seq $1 $2`
					matlab -nodisplay -nodesktop -r "plot_results_curves('$DATADIR',Inf,{'$OUTDIR'},'','shapes',{'$OUTDIR'},0,0); print('-depsc2',['$OUTDIR' '/experiment.pdf']); exit" < /dev/null
				done
			done
		done
	fi
	if [ 0 == 1 ]; then
		# compare Nmax for cep_igci
		for Nmax in 10 50 100 200 500 1000; do
			for refMeasure in 1 2; do
				methodstr=("org_entropy" "slope" "slope++" "entropy"  "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy")
				enteststr=("" "" "" "KSG" "Shannon_kNN_k" "Shannon_Edgeworth" "Shannon_spacing_V" "Shannon_spacing_Vb" "Shannon_spacing_Vpconst" "Shannon_spacing_Vplin" "Shannon_spacing_Vplin2" "Shannon_KDP" "Shannon_MaxEnt1" "Shannon_MaxEnt2" "Shannon_PSD_SzegoT" "Shannon_expF" "Shannon_spacing_VKDE")
				for method in `seq 0 16`; do
					dirmethod=${methodstr[$method]}_${enteststr[$method]}
					OUTDIR="$DATADIR""_out/igci_N=$Nmax""_$dirmethod""_norm$refMeasure"
					mkdir -p $OUTDIR
					echo "Nmax=$Nmax, rep=1, outdir='$OUTDIR', datadir='$DATADIR', pp=struct; pp.randseed=pair*rep; pp.undisc=0; pp.disc=0; pp.maxN=Nmax; pp.disturbance=0; mp=struct; mp.refMeasure=$refMeasure; mp.estimator='${methodstr[$method]}'; mp.entest='${$enteststr[$method]}'; run_method (pair,'cep_igci',mp,sprintf('%s/%d',outdir,rep),pp,datadir);" > $OUTDIR/experiment.m
					parallel -j $PARALLEL runmatlab {1} $OUTDIR ::: `seq $1 $2`
					matlab -nodisplay -nodesktop -r "plot_results_curves('$DATADIR',Inf,{'$OUTDIR'},'','shapes',{'$OUTDIR'},0,0); print('-depsc2',['$OUTDIR' '/experiment.pdf']); exit" < /dev/null
				done
			done
		done
	fi

	# run ANM-HSIC with data splitting, fixed kernel, and both options  [IN ARTICLE]
	if [ 1 == 1 ]; then
		method=HSIC
		splitstr=("" "_split")
		fkstr=("" "_fk")
		bandwidthstr=("[0,0]" "[0.5,0.5]")
		perturbstr=("" "_disc" "_undisc" "_dist=1e-9")
		undisc=("0" "0" "1" "0")
		disc=("0" "1" "0" "0")
		dist=("0" "0" "0" "1e-9")
		for split in 0 1; do
			for fk in 0 1; do
				if [[ "$split" == "1" || "$fk" == "1" ]]; then 
					for perturb in 0 1 2 3; do
						OUTDIR="$DATADIR""_out/anm_N=Inf_FITC=100_$method"_""${splitstr[$split]}${fkstr[$fk]}"_lbfgsb"${perturbstr[$perturb]}
						mkdir -p $OUTDIR
						echo "Nmax=Inf, rep=1, outdir='$OUTDIR', datadir='$DATADIR', pp=struct; pp.randseed=pair*rep; pp.undisc="${undisc[$perturb]}"; pp.disc="${undisc[$perturb]}"; pp.maxN=Nmax; pp.disturbance="${dist[$perturb]}"; mp=struct; mp.nrperm=0; mp.gaussianize=0; mp.FITC=100; mp.splitdata=$split; mp.evaluation='$method'; mp.meanf='meanConst'; mp.minimize='minimize_lbfgsb'; mp.bandwidths="${bandwidthstr[$fk]}"; run_method (pair,'cep_anm',mp,sprintf('%s/%d',outdir,rep),pp,datadir);" > $OUTDIR/experiment.m
						parallel -j $PARALLEL runmatlab {1} $OUTDIR ::: `seq $1 $2`
						matlab -nodisplay -nodesktop -r "plot_results_curves('$DATADIR',Inf,{'$OUTDIR'},'','shapes',{'$OUTDIR'},0,0); print('-depsc2',['$OUTDIR' '/experiment.pdf']); exit" < /dev/null
					done
				fi
			done
		done
	fi

	if [ 1 == 1 ]; then  # [IN ARTICLE]
		# compare different model selection methods for cep_anm
		methodstr=("pHSIC" "HSIC" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "FN" "Gauss" "MML")
		enteststr=("" "" "KSG" "Shannon_kNN_k" "Shannon_Edgeworth" "Shannon_spacing_V" "Shannon_spacing_Vb" "Shannon_spacing_Vpconst" "Shannon_spacing_Vplin" "Shannon_spacing_Vplin2" "Shannon_KDP" "Shannon_MaxEnt1" "Shannon_MaxEnt2" "Shannon_PSD_SzegoT" "Shannon_expF" "Shannon_spacing_VKDE" "" "" "")
		for method in `seq 0 18`; do
			perturbstr=("" "_disc" "_undisc" "_dist=1e-9")
			undisc=("0" "0" "1" "0")
			disc=("0" "1" "0" "0")
			dist=("0" "0" "0" "1e-9")
			for perturb in 0 1 2 3; do
				dirmethod=${methodstr[$method]}_${enteststr[$method]}
				OUTDIR="$DATADIR""_out/anm_N=Inf_FITC=100_$dirmethod""_lbfgsb"${perturbstr[$perturb]}
				mkdir -p $OUTDIR
				echo "Nmax=Inf, rep=1, outdir='$OUTDIR', datadir='$DATADIR', pp=struct; pp.randseed=pair*rep; pp.undisc="${undisc[$perturb]}"; pp.disc="${undisc[$perturb]}"; pp.maxN=Nmax; pp.disturbance="${dist[$perturb]}"; mp=struct; mp.nrperm=0; mp.gaussianize=0; mp.FITC=100; mp.splitdata=0; mp.evaluation='${methodstr[$method]}'; mp.meanf='meanConst'; mp.minimize='minimize_lbfgsb'; mp.bandwidths=[0,0]; mp.entest='${enteststr[$method]}'; run_method (pair,'cep_anm',mp,sprintf('%s/%d',outdir,rep),pp,datadir);" > $OUTDIR/experiment.m
				parallel -j $PARALLEL runmatlab {1} $OUTDIR ::: `seq $1 $2`
				matlab -nodisplay -nodesktop -r "plot_results_curves('$DATADIR',Inf,{'$OUTDIR'},'','shapes',{'$OUTDIR'},0,0); print('-depsc2',['$OUTDIR' '/experiment.pdf']); exit" < /dev/null
			done
		done
	fi

	if [ 1 == 1 ]; then  # [IN ARTICLE]
		# compare different entropy estimators for cep_igci
		methodstr=("org_entropy" "slope" "slope++" "entropy"  "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy" "entropy")
		enteststr=("" "" "" "KSG" "Shannon_kNN_k" "Shannon_Edgeworth" "Shannon_spacing_V" "Shannon_spacing_Vb" "Shannon_spacing_Vpconst" "Shannon_spacing_Vplin" "Shannon_spacing_Vplin2" "Shannon_KDP" "Shannon_MaxEnt1" "Shannon_MaxEnt2" "Shannon_PSD_SzegoT" "Shannon_expF" "Shannon_spacing_VKDE")
		for method in `seq 0 16`; do
			for refMeasure in 1 2; do
				perturbstr=("" "_disc" "_undisc" "_dist=1e-9")
				undisc=("0" "0" "1" "0")
				disc=("0" "1" "0" "0")
				dist=("0" "0" "0" "1e-9")
				for perturb in 0 1 2 3; do
					dirmethod=${methodstr[$method]}_${enteststr[$method]}
					OUTDIR="$DATADIR""_out/igci_N=Inf_$dirmethod""_norm$refMeasure"${perturbstr[$perturb]}
					mkdir -p $OUTDIR
					echo "Nmax=Inf, rep=1, outdir='$OUTDIR', datadir='$DATADIR', pp=struct; pp.randseed=pair*rep; pp.undisc="${undisc[$perturb]}"; pp.disc="${undisc[$perturb]}"; pp.maxN=Nmax; pp.disturbance="${dist[$perturb]}"; mp=struct; mp.refMeasure=$refMeasure; mp.estimator='${methodstr[$method]}'; mp.entest='${enteststr[$method]}'; run_method (pair,'cep_igci',mp,sprintf('%s/%d',outdir,rep),pp,datadir);" > $OUTDIR/experiment.m
					parallel -j $PARALLEL runmatlab {1} $OUTDIR ::: `seq $1 $2`
					matlab -nodisplay -nodesktop -r "plot_results_curves('$DATADIR',Inf,{'$OUTDIR'},'','shapes',{'$OUTDIR'},0,0); print('-depsc2',['$OUTDIR' '/experiment.pdf']); exit" < /dev/null
				done
			done
		done
	fi

	if [ 0 == 1 ]; then
		# cep_count
		perturbstr=("" "_disc" "_undisc" "_dist=1e-9")
		undisc=("0" "0" "1" "0")
		disc=("0" "1" "0" "0")
		dist=("0" "0" "0" "1e-9")
		for perturb in 0 1 2 3; do
			OUTDIR="$DATADIR""_out/count_N=Inf"${perturbstr[$perturb]}
			mkdir -p $OUTDIR
			echo "Nmax=Inf, rep=1, outdir='$OUTDIR', datadir='$DATADIR', pp=struct; pp.randseed=pair*rep; pp.undisc="${undisc[$perturb]}"; pp.disc="${undisc[$perturb]}"; pp.maxN=Nmax; pp.disturbance="${dist[$perturb]}"; mp=struct; run_method (pair,'cep_count',mp,sprintf('%s/%d',outdir,rep),pp,datadir);" > $OUTDIR/experiment.m
			parallel -j $PARALLEL runmatlab {1} $OUTDIR ::: `seq $1 $2`
			matlab -nodisplay -nodesktop -r "plot_results_curves('$DATADIR',Inf,{'$OUTDIR'},'','shapes',{'$OUTDIR'},0,0); print('-depsc2',['$OUTDIR' '/experiment.pdf']); exit" < /dev/null
		done
	fi

done

#####

if [ 0 == 1 ]; then
	# compare number of FITC points for cep_anm
	for FITC in 200 500 1000; do
		for DATADIR in $DATADIRS; do
			methodstr=("pHSIC" "HSIC" "entropy" "entropy" "MML")
			enteststr=("" "" "Shannon_KDP" "Shannon_PSD_SzegoT" "")
			for method in 0 1 2 3 4; do
				dirmethod=${methodstr[$method]}_${enteststr[$method]}
				OUTDIR="$DATADIR""_out/anm_N=Inf_FITC=$FITC""_$dirmethod""_lbfgsb"
				mkdir -p $OUTDIR
				echo "Nmax=Inf, rep=1, outdir='$OUTDIR', datadir='$DATADIR', pp=struct; pp.randseed=pair*rep; pp.undisc=0; pp.disc=0; pp.maxN=Nmax; pp.disturbance=0; mp=struct; mp.nrperm=0; mp.gaussianize=0; mp.FITC=$FITC; mp.splitdata=0; mp.evaluation='${methodstr[$method]}'; mp.meanf='meanConst'; mp.minimize='minimize_lbfgsb'; mp.bandwidths=[0,0]; mp.entest='${enteststr[$method]}'; run_method (pair,'cep_anm',mp,sprintf('%s/%d',outdir,rep),pp,datadir);" > $OUTDIR/experiment.m
				parallel -j $PARALLEL runmatlab {1} $OUTDIR ::: `seq $1 $2`
				matlab -nodisplay -nodesktop -r "plot_results_curves('$DATADIR',Inf,{'$OUTDIR'},'','shapes',{'$OUTDIR'},0,0); print('-depsc2',['$OUTDIR' '/experiment.pdf']); exit"
			done
		done
	done

	# pHSIC__minimize is a special case
	for FITC in 100 200 500 1000; do
		for DATADIR in $DATADIRS; do
			OUTDIR="$DATADIR""_out/anm_N=Inf_FITC=$FITC""_pHSIC__minimize"
			mkdir -p $OUTDIR
			echo "Nmax=Inf, rep=1, outdir='$OUTDIR', datadir='$DATADIR', pp=struct; pp.randseed=pair*rep; pp.undisc=0; pp.disc=0; pp.maxN=Nmax; pp.disturbance=0; mp=struct; mp.nrperm=0; mp.gaussianize=0; mp.FITC=$FITC; mp.splitdata=0; mp.evaluation='pHSIC'; mp.meanf='meanConst'; mp.minimize='minimize'; mp.bandwidths=[0,0]; mp.entest=''; run_method (pair,'cep_anm',mp,sprintf('%s/%d',outdir,rep),pp,datadir);" > $OUTDIR/experiment.m
			parallel -j $PARALLEL runmatlab {1} $OUTDIR ::: `seq $1 $2`
			matlab -nodisplay -nodesktop -r "plot_results_curves('$DATADIR',Inf,{'$OUTDIR'},'','shapes',{'$OUTDIR'},0,0); print('-depsc2',['$OUTDIR' '/experiment.pdf']); exit"
		done
	done
fi
