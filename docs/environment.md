# Environment Setup

This repository is organized so the active training, evaluation, and profiling
paths use the local `film_osg` package and do not require the external `due`
package.

## Tested Environment

The code was tested with:

- Python `3.11.14`
- PyTorch `2.0.1`
- PyTorch CUDA runtime `11.7`
- NumPy `1.26.4`
- SciPy `1.16.3`
- Matplotlib `3.10.7`
- NVIDIA CUDA GPU with sufficient memory for the selected batch size

The launchers are not tied to a specific GPU model. GPU ids are passed through
`--gpus`, and each child process receives one id through
`CUDA_VISIBLE_DEVICES`. Memory profiling uses PyTorch CUDA APIs.

## Recommended Clean Conda Environment

Create a clean environment:

```bash
conda create -n film_osg_clean python=3.11 -y
conda activate film_osg_clean
```

Install PyTorch. The tested setup used CUDA 11.7 wheels:

```bash
python -m pip install torch==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117
```

Then install the repository's minimal Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data Readers

The current `.mat` files are read with `scipy.io.loadmat`. The `h5py` package
listed in `requirements.txt` is used by the PDEBench conversion script. If a
MATLAB dataset is saved in the v7.3 format, install `mat73` separately:

```bash
python -m pip install mat73
```

This additional package is not required for the datasets described in
`data/README.md`.

## Minimal Checks

Run these before launching any full experiment:

```bash
python -c "from film_osg.datasets.pde import pde_dataset_osg; from film_osg.networks.fno import osg_fno1d, osg_fno2d; from film_osg.models.pde_osg import PDE_osg; print('film_osg imports ok')"
python train/run_burgers_fno.py --help
python eval/eval_burgers_fno.py --help
python profiling/profile_ns_overhead.py --help
```

For a short verification run when the Burgers `.mat` files are present:

```bash
python train/run_burgers_fno.py --model fno --seed 0 --epochs 1 --batch-size 100 --tag verification
python train/run_burgers_fno.py --model fno_film --seed 0 --epochs 1 --batch-size 100 --tag verification
python eval/eval_burgers_fno.py --models fno,fno_film --seeds 0 --tag verification --eval-steps 1 --save-dir eval_outputs_burgers_verification
```
