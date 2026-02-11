# CIFAR-10: Standard CNN vs Depthwise-Separable CNN

CSL7590 Deep Learning assignment: image classification on CIFAR-10 with a CNN where at least two standard convolution layers are replaced by depthwise + pointwise convolution, and comparison in parameter count and computational efficiency (MACs). Includes Grid Search and Optuna for hyperparameter tuning (3 hyperparameters, 3 values each).

## Requirements

- Python 3.10+
- PyTorch, torchvision
- numpy, matplotlib, seaborn, scikit-learn, optuna

On the cluster:
```bash
module load python/3.10.pytorch
pip install --user optuna seaborn scikit-learn
```

## Data

CIFAR-10 is downloaded automatically to `./data/` on first run (official train/test split). Training set is split 80/20 into train/validation with a fixed seed (42) for reproducibility.

## Run

**Local:**
```bash
python3 q2_final.py
```

**Slurm:**
```bash
sbatch run_job.sh
```
Logs go to `slurm_q2_<jobid>.out`.

## Outputs

- **visualizations/**  
  - `Standard_CNN_learning_curves.png`, `Separable_CNN_learning_curves.png`  
  - `Standard_CNN_confusion_matrix.png`, `Separable_CNN_confusion_matrix.png`  
  - `efficiency_comparison.png` (parameters and MACs)

- **report/**  
  LaTeX source: `report/main.tex`. Build the PDF from the project root:
  ```bash
  pdflatex report/main.tex
  pdflatex report/main.tex
  ```
  Output: `report/main.pdf`. Figures are loaded from `visualizations/`.

## Assignment compliance

See `ASSIGNMENT_COMPLIANCE.md` for a requirement checklist and where each is implemented.
