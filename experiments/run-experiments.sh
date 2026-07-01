#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

for seed in {1001..1100}; do
    # Synthetic linear regression setup (coverage)
    python prepare-dgp.py id="linreg-01" data_size=20 dgp=regression-fixed seed=${seed}
    python prepare-dgp.py id="linreg-02" data_size=20 dgp=regression-fixed-dependent dgp.s_small=0.25 dgp.s_mod=0.5 seed=${seed}
    python prepare-dgp.py id="linreg-03" data_size=20 dgp=regression-fixed-dependent dgp.s_small=0.05 dgp.s_mod=0.25 seed=${seed}
    python prepare-dgp.py id="linreg-04" data_size=20 dgp=regression-fixed-dependent dgp.s_small=0.01 dgp.s_mod=0.1 seed=${seed}
    python prepare-dgp.py id="linreg-05" data_size=20 dgp=regression-fixed-non-normal dgp.df=5 seed=${seed}
    python prepare-dgp.py id="linreg-06" data_size=20 dgp=regression-fixed-non-normal dgp.df=4 seed=${seed}
    python prepare-dgp.py id="linreg-07" data_size=20 dgp=regression-fixed-non-normal dgp.df=3 seed=${seed}

    # Synthetic logistic regression setup (coverage)
    python prepare-dgp.py id="logreg-01" data_size=100 dgp=classification-fixed seed=${seed}
    python prepare-dgp.py id="logreg-02" data_size=100 dgp=classification-fixed-gmm dgp.a=0 seed=${seed}
    python prepare-dgp.py id="logreg-03" data_size=100 dgp=classification-fixed-gmm dgp.a=-1 seed=${seed}
    python prepare-dgp.py id="logreg-04" data_size=100 dgp=classification-fixed-gmm dgp.a=-2 seed=${seed}

    # Real linear regression setup (coverage)
    python prepare-dgp.py id="linreg-real-01" data_size=50  dgp=openml dgp.name=quake seed=${seed}
    python prepare-dgp.py id="linreg-real-02" data_size=50  dgp=openml dgp.name=airfoil seed=${seed}
    python prepare-dgp.py id="linreg-real-03" data_size=50  dgp=openml dgp.name=kin8nm seed=${seed}
    python prepare-dgp.py id="linreg-real-04" data_size=100 dgp=openml dgp.name=concrete seed=${seed}
    python prepare-dgp.py id="linreg-real-05" data_size=50  dgp=openml dgp.name=energy seed=${seed}
    python prepare-dgp.py id="linreg-real-06" data_size=50  dgp=openml dgp.name=grid seed=${seed}
    python prepare-dgp.py id="linreg-real-07" data_size=20  dgp=openml dgp.name=abalone seed=${seed}
    python prepare-dgp.py id="linreg-real-08" data_size=50  dgp=openml dgp.name=fish seed=${seed}
    python prepare-dgp.py id="linreg-real-09" data_size=50  dgp=openml dgp.name=auction seed=${seed}

    # Real logistic regression setup (coverage)
    python prepare-dgp.py id="logreg-real-01" data_size=50  dgp=openml dgp.name=blood seed=${seed}
    python prepare-dgp.py id="logreg-real-02" data_size=50  dgp=openml dgp.name=phoneme seed=${seed}
    python prepare-dgp.py id="logreg-real-03" data_size=50  dgp=openml dgp.name=skin seed=${seed}
    python prepare-dgp.py id="logreg-real-04" data_size=100 dgp=openml dgp.name=rice seed=${seed}
    python prepare-dgp.py id="logreg-real-05" data_size=100 dgp=openml dgp.name=mozilla seed=${seed}
    python prepare-dgp.py id="logreg-real-06" data_size=50  dgp=openml dgp.name=telescope seed=${seed}
    python prepare-dgp.py id="logreg-real-07" data_size=100 dgp=openml dgp.name=sepsis seed=${seed}
    python prepare-dgp.py id="logreg-real-08" data_size=200 dgp=openml dgp.name=yeast seed=${seed}
    python prepare-dgp.py id="logreg-real-09" data_size=200 dgp=openml dgp.name=wine seed=${seed}
    python prepare-dgp.py id="logreg-real-10" data_size=100 dgp=openml dgp.name=banknote seed=${seed}
done


# Semi-real linear regression setup (coverage)
for seed in {1001..1050}; do
    python prepare-dgp.py id="semireal-01" data_size=100 dgp=openml dgp.name=concrete-semireal seed=${seed}
    python prepare-dgp.py id="semireal-02" data_size=100 dgp=openml dgp.name=abalone-semireal seed=${seed}
    python prepare-dgp.py id="semireal-03" data_size=50 dgp=openml dgp.name=concrete-semireal seed=${seed}
    python prepare-dgp.py id="semireal-04" data_size=20 dgp=openml dgp.name=abalone-semireal seed=${seed}
done


# Check for convergence with larger rollout length (1000)
python prepare-dgp.py id="longroll-01" data_size=50 dgp=openml dgp.name=skin seed=1001
python prepare-dgp.py id="longroll-02" data_size=200 dgp=openml dgp.name=yeast seed=1001
python prepare-dgp.py id="longroll-03" data_size=200 dgp=openml dgp.name=wine seed=1001
python prepare-dgp.py id="longroll-04" data_size=100 dgp=classification-fixed-gmm dgp.a=0 seed=1001
python prepare-dgp.py id="longroll-05" data_size=100 dgp=classification-fixed-gmm dgp.a=-1 seed=1001
python prepare-dgp.py id="longroll-06" data_size=100 dgp=classification-fixed-gmm dgp.a=-2 seed=1001


# Check the coverage with larger rollout length (1000)
for seed in {1002..1050}; do
    python prepare-dgp.py id="longroll-04" data_size=100 dgp=classification-fixed-gmm dgp.a=0 seed=${seed}
    python prepare-dgp.py id="longroll-01" data_size=50 dgp=openml dgp.name=skin seed=${seed}
    python prepare-dgp.py id="longroll-02" data_size=200 dgp=openml dgp.name=yeast seed=${seed}
    python prepare-dgp.py id="longroll-03" data_size=200 dgp=openml dgp.name=wine seed=${seed}
done

# Concentration experiment, start with data_size=500, 1000, 1500, 2000
for data_size in 500 1000 1500 2000; do
    python prepare-dgp.py id="concentration-01" data_size=${data_size} dgp=classification-fixed seed=1001
    python prepare-dgp.py id="concentration-02" data_size=${data_size} dgp=regression-fixed seed=1001
done


# ACID experiments
python prepare-dgp.py id="acid-01" data_size=100 dgp=classification-fixed dgp.dim_x=2 seed=1001


# Go into each setup and compute rollouts/posteriors
OUTPUT_PATH="./outputs" # The folder that contains all experiment setups
while read -r -d $'\0' path; do
    dirname=$(basename "$path")
    exp_id="${path#${OUTPUT_PATH}/}"
    exp_id="${exp_id%%/*}"
    seed_part="${dirname#*seed=}" # Remove prefix up to 'seed='
    seed="${seed_part%% *}"       # Remove suffix starting from the first space (or end of string)
    tabmgp_args=()

    if [[ "$exp_id" == acid-* ]]; then
        for sample_idx in {0..9}; do
            python run-acid.py "expdir='${path}'" "sample_idx=${sample_idx}"
        done
        continue
    fi

    # Set n_estimators=4 for logistic regression and n_estimators=8 for linear regression
    case "$exp_id" in
        linreg-*|linreg-real-*|semireal-*|concentration-02)
            tabmgp_args+=("n_estimators=8")
            ;;
        logreg-*|logreg-real-*|longroll-*|concentration-01)
            tabmgp_args+=("n_estimators=4")
            ;;
        *)
            echo "No n_estimators setting for experiment id: ${exp_id}" >&2
            exit 1
            ;;
    esac

    # Coverage and diagnostic at various rollout lengths
    if [[ "$exp_id" == longroll-* ]]; then
        tabmgp_args+=("forward_steps=[100,200,300,400,500,600,700,800,900,1000]")
        if [[ "$seed" == 1001 ]]; then
            python run-tabmgp.py "expdir='${path}'" "trace=True" "${tabmgp_args[@]}"
        else
            python run-tabmgp.py "expdir='${path}'" "trace=False" "${tabmgp_args[@]}"
        fi
    fi


    # Concentration of TabMGP
    if [[ "$exp_id" == concentration-* ]]; then
        tabmgp_args+=("forward_steps=[500,2000]")
        python run-tabmgp.py "expdir='${path}'" "trace=False" "${tabmgp_args[@]}"
    fi

    case "$exp_id" in
        linreg-*|linreg-real-*|logreg-*|logreg-real-*|semireal-*)
            if [ "$seed" -ge 1001 ] && [ "$seed" -le 1020 ]; then
                # For seeds 1001 through 1020, also compute the trace
                python run-tabmgp.py "expdir='${path}'" "trace=True" "${tabmgp_args[@]}"
                python run-bb.py "expdir='${path}'" "trace=True"
                python run-copula.py "expdir='${path}'" "trace=True" "init=std"
                python run-copula.py "expdir='${path}'" "trace=True" "init=tabpfn"
            else
                python run-tabmgp.py "expdir='${path}'" "trace=False" "${tabmgp_args[@]}"
                python run-bb.py "expdir='${path}'" "trace=False"
                python run-copula.py "expdir='${path}'" "trace=False" "init=std"
                python run-copula.py "expdir='${path}'" "trace=False" "init=tabpfn"
            fi
            python run-bayes.py "expdir='${path}'" "prior=flat"
            python run-bayes.py "expdir='${path}'" "prior=asymp"
            ;;
    esac



done < <(find "$OUTPUT_PATH" -mindepth 2 -maxdepth 2 -type d -print0 | sort -z -u)
