# Environment Setup

This repository is organized so the active training, evaluation, and profiling
paths use the local `film_osg` package and do not require the external `due`
package.

## Tested Environment

Smoke tests were run with:

- Python `3.11.14`
- PyTorch `2.0.1`
- PyTorch CUDA runtime `11.7`
- NumPy `1.26.4`
- SciPy `1.16.3`
- Matplotlib `3.10.7`
- PyYAML `6.0.3`
- NVIDIA CUDA GPU with sufficient memory for the selected batch size

The launchers are not tied to a specific GPU model. They print the
PyTorch-visible CUDA mapping and accept any available CUDA GPU by default.
Memory profiling uses PyTorch CUDA APIs; if `nvidia-smi` reports an NVML
driver/library warning, check whether PyTorch can still see CUDA before
launching full experiments.

## Recommended Clean Conda Environment

Create a clean environment:

```bash
conda create -n film_osg_clean python=3.11 -y
conda activate film_osg_clean
```

Install PyTorch. The smoke-tested setup used CUDA 11.7 wheels:

```bash
python -m pip install torch==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117
```

Then install the repository's minimal Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Optional Dependencies

The current smoke-tested `.mat` files are readable through `scipy.io.loadmat`.
If future MATLAB files are saved as v7.3/HDF5 files, install optional readers:

```bash
python -m pip install h5py mat73
```

These optional packages are not required for the current smoke-tested data path.

## Minimal Checks

Run these before launching any full experiment:

```bash
python -c "from film_osg.datasets.pde import pde_dataset_osg; from film_osg.networks.fno import osg_fno1d, osg_fno2d; from film_osg.models.pde_osg import PDE_osg; print('film_osg imports ok')"
python train/run_burgers_fno.py --help
python eval/eval_burgers_fno.py --help
python profiling/profile_ns_overhead.py --check-only --models fno,fno_film
```

For a short real smoke test when the Burgers `.mat` files are present:

```bash
python train/run_burgers_fno.py --model fno --seed 0 --epochs 1 --batch-size 100 --tag nodue_smoke
python train/run_burgers_fno.py --model fno_film --seed 0 --epochs 1 --batch-size 100 --tag nodue_smoke
python eval/eval_burgers_fno.py --models fno,fno_film --seeds 0 --tag nodue_smoke --eval-steps 1 --save-dir eval_outputs_burgers_nodue_smoke
```
