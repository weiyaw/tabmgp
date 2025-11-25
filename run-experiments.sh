#!/bin/bash

for seed in {1001..1100}; do
    # Synthetic linear regression setup (coverage)
    python run-rollout.py date="2025-06-02"   data_size=20 n_estimators=8 dgp=regression-fixed seed=${seed}
    python run-rollout.py date="2025-06-06-a" data_size=20 n_estimators=8 dgp=regression-fixed-dependent dgp.s_small=0.25 dgp.s_mod=0.5 seed=${seed}
    python run-rollout.py date="2025-06-06-b" data_size=20 n_estimators=8 dgp=regression-fixed-dependent dgp.s_small=0.05 dgp.s_mod=0.25 seed=${seed}
    python run-rollout.py date="2025-06-06-c" data_size=20 n_estimators=8 dgp=regression-fixed-dependent dgp.s_small=0.01 dgp.s_mod=0.1 seed=${seed}
    python run-rollout.py date="2025-06-07-a" data_size=20 n_estimators=8 dgp=regression-fixed-non-normal dgp.df=5 seed=${seed}
    python run-rollout.py date="2025-06-07-b" data_size=20 n_estimators=8 dgp=regression-fixed-non-normal dgp.df=4 seed=${seed}
    python run-rollout.py date="2025-06-07-c" data_size=20 n_estimators=8 dgp=regression-fixed-non-normal dgp.df=3 seed=${seed}

    # Synthetic logistic regression setup (coverage)
    python run-rollout.py date="2025-06-01"   data_size=100 n_estimators=4 dgp=classification-fixed seed=${seed}
    python run-rollout.py date="2025-06-05-a" data_size=100 n_estimators=4 dgp=classification-fixed-gmm dgp.a=0 seed=${seed}
    python run-rollout.py date="2025-06-05-b" data_size=100 n_estimators=4 dgp=classification-fixed-gmm dgp.a=-1 seed=${seed}
    python run-rollout.py date="2025-06-05-c" data_size=100 n_estimators=4 dgp=classification-fixed-gmm dgp.a=-2 seed=${seed}

    # Real linear regression setup (coverage)
    python run-rollout.py date="2025-07-01" data_size=50  n_estimators=8 dgp=openml dgp.name=quake seed=${seed}
    python run-rollout.py date="2025-07-02" data_size=50  n_estimators=8 dgp=openml dgp.name=airfoil seed=${seed}
    python run-rollout.py date="2025-07-03" data_size=50  n_estimators=8 dgp=openml dgp.name=kin8nm seed=${seed}
    python run-rollout.py date="2025-07-04" data_size=100 n_estimators=8 dgp=openml dgp.name=concrete seed=${seed}
    python run-rollout.py date="2025-07-05" data_size=50  n_estimators=8 dgp=openml dgp.name=energy seed=${seed}
    python run-rollout.py date="2025-07-06" data_size=50  n_estimators=8 dgp=openml dgp.name=grid seed=${seed}
    python run-rollout.py date="2025-07-07" data_size=20  n_estimators=8 dgp=openml dgp.name=abalone seed=${seed}
    python run-rollout.py date="2025-07-08" data_size=50  n_estimators=8 dgp=openml dgp.name=fish seed=${seed}
    python run-rollout.py date="2025-07-09" data_size=50  n_estimators=8 dgp=openml dgp.name=auction seed=${seed}

    # Real logistic regression setup (coverage)
    python run-rollout.py date="2025-07-51" data_size=50  n_estimators=4 dgp=openml dgp.name=blood seed=${seed}
    python run-rollout.py date="2025-07-52" data_size=50  n_estimators=4 dgp=openml dgp.name=phoneme seed=${seed}
    python run-rollout.py date="2025-07-53" data_size=50  n_estimators=4 dgp=openml dgp.name=skin seed=${seed}
    python run-rollout.py date="2025-07-54" data_size=100 n_estimators=4 dgp=openml dgp.name=rice seed=${seed}
    python run-rollout.py date="2025-07-55" data_size=100 n_estimators=4 dgp=openml dgp.name=mozilla seed=${seed}
    python run-rollout.py date="2025-07-56" data_size=50  n_estimators=4 dgp=openml dgp.name=telescope seed=${seed}
    python run-rollout.py date="2025-07-57" data_size=100 n_estimators=4 dgp=openml dgp.name=sepsis seed=${seed}
    python run-rollout.py date="2025-07-58" data_size=200 n_estimators=4 dgp=openml dgp.name=yeast seed=${seed}
    python run-rollout.py date="2025-07-59" data_size=200 n_estimators=4 dgp=openml dgp.name=wine seed=${seed}
    python run-rollout.py date="2025-07-60" data_size=100 n_estimators=4 dgp=openml dgp.name=banknote seed=${seed} rollout_times=2
done


# concentration experiment, start with data_size=50, 100, 150, 200, 250, 300
for data_size in 50 100 150 200 250 300; do
    python run-rollout.py date="2025-06-11" data_size=${data_size} dgp=classification-fixed seed=1001
    python run-rollout.py date="2025-06-12" data_size=${data_size} dgp=regression-fixed seed=1001
done


# Go into each rollout and compute posteriors
OUTPUT_PATH="./outputs" # The folder that contains all the tabpfn rollout
while read -r -d $'\0' path; do
    dirname=$(basename "$path")
    seed_part="${dirname#*seed=}" # Remove prefix up to 'seed='
    seed="${seed_part%% *}"       # Remove suffix starting from the first space (or end of string)

    if [ "$seed" = "1001" ]; then
        # For seed 1001, also compute the trace
        python run-posterior.py "expdir='${path}'" "trace=True"
    fi

    if [ "$seed" != "1001" ]; then
        python run-posterior.py "expdir='${path}'" "trace=False"
    fi

done < <(find "$OUTPUT_PATH" -mindepth 2 -maxdepth 2 -type d -print0 | sort -z -u)
