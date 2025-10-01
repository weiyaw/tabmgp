## TabMGP: Martingale Posterior with TabPFN

## Requirements
We provide `requirements.txt` and `requirements-posterior.txt` files that can be used with pip. The recommended
way is to create a conda env, then pip install:

```bash
conda create --name <env> python=3.11
conda activate <env>
pip install -r requirements.txt
pip install -r requirements-posterior.txt
```
This will install a CPU version of JAX. Please see the [_JAX repo_](https://github.com/google/jax) for the latest
instructions on how to install JAX on your hardware.

## File Structure
```
+-- acid.py (Run acid test)
+-- credible_set.py (Credible sets related files)
+-- data.py (All data generating distribution and dataset downloading)
+-- forward.py (Forward sampling with TabPFN)
+-- posterior.py (Compute all type of posteriors)
+-- plot.py (Generate plots)
+-- table.py (Generate tables)
+-- utils.py (Utility functions)
```

### Example
```bash
python forward.py data_size=20 recursion_length=500 n_estimators=8 resample_x=bb dgp=regression-fixed
```
This will run forward sampling for TabPFN and save the samples in a folder "output/name-of-sample-folder".

``` bash
python posterior.py expdir="output/name-of-sample-folder"
```
This will compute all martingale posterior (TabMGP, BB, Copula) and Bayes. This has to be run after running `forward.py`.

