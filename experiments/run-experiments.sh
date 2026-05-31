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
python prepare-dgp.py id="longroll-01" data_size=50 dgp=openml dgp.name=skin seed=${seed}
python prepare-dgp.py id="longroll-02" data_size=200 dgp=openml dgp.name=yeast seed=${seed}
python prepare-dgp.py id="longroll-03" data_size=200 dgp=openml dgp.name=wine seed=${seed}
python prepare-dgp.py id="longroll-04" data_size=100 dgp=classification-fixed-gmm dgp.a=0 seed=${seed}
python prepare-dgp.py id="longroll-05" data_size=100 dgp=classification-fixed-gmm dgp.a=-1 seed=${seed}
python prepare-dgp.py id="longroll-06" data_size=100 dgp=classification-fixed-gmm dgp.a=-2 seed=${seed}


# Check the coverage with larger rollout length (1000)
for seed in {1001..1050}; do
    python prepare-dgp.py id="longroll-04" data_size=100 dgp=classification-fixed-gmm dgp.a=0 seed=${seed}
    python prepare-dgp.py id="longroll-01" data_size=50 dgp=openml dgp.name=skin seed=${seed}
    python prepare-dgp.py id="longroll-02" data_size=200 dgp=openml dgp.name=yeast seed=${seed}
    python prepare-dgp.py id="longroll-03" data_size=200 dgp=openml dgp.name=wine seed=${seed}
done

# concentration experiment, start with data_size=50, 100, 150, 200, 250, 300
for data_size in 50 100 150 200 250 300; do
    python prepare-dgp.py id="concentration-01" data_size=${data_size} dgp=classification-fixed seed=1001
    python prepare-dgp.py id="concentration-02" data_size=${data_size} dgp=regression-fixed seed=1001
done


# Go into each setup and compute rollouts/posteriors
OUTPUT_PATH="./outputs" # The folder that contains all experiment setups
while read -r -d $'\0' path; do
    dirname=$(basename "$path")
    seed_part="${dirname#*seed=}" # Remove prefix up to 'seed='
    seed="${seed_part%% *}"       # Remove suffix starting from the first space (or end of string)
    tabmgp_args=()

    case "$path" in
        */linreg-*/*|*/linreg-real-*/*|*/semireal-*/*)
            tabmgp_args+=("n_estimators=8")
            ;;
    esac

    case "$path" in
        */longroll-*/*)
            tabmgp_args+=("forward_steps=1000")
            ;;
    esac

    if [ "$seed" = "1001" ]; then
        # For seed 1001, also compute the trace
        python run-tabmgp.py "expdir='${path}'" "trace=True" "${tabmgp_args[@]}"
        python run-bb.py "expdir='${path}'" "trace=True"
        python run-copula.py "expdir='${path}'" "trace=True" "init=std"
        python run-copula.py "expdir='${path}'" "trace=True" "init=tabpfn"
        python run-bayes.py "expdir='${path}'" "prior=asymp"
    fi

    if [ "$seed" != "1001" ]; then
        python run-tabmgp.py "expdir='${path}'" "trace=False" "${tabmgp_args[@]}"
        python run-bb.py "expdir='${path}'" "trace=False"
        python run-copula.py "expdir='${path}'" "trace=False" "init=std"
        python run-copula.py "expdir='${path}'" "trace=False" "init=tabpfn"
        python run-bayes.py "expdir='${path}'" "prior=asymp"
    fi

done < <(find "$OUTPUT_PATH" -mindepth 2 -maxdepth 2 -type d -print0 | sort -z -u)
