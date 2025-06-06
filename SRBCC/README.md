# SRBCC (Semi-Supervised Robust Binary Classification)

This repository contains the implementation of SRBCC, along with evaluation tools and datasets.

## Project Structure

```
.
├── src/
│   ├── SRBCC_main.py      # Main implementation of the SRBCC algorithm
│   ├── noise_eval.py      # Noise evaluation and analysis tools
│   └── webKB_runner.py    # WebKB dataset processing and evaluation
├── data/
│   ├── synthetic/         # Synthetic datasets
│   ├── WebKB datasets     # Various WebKB datasets (cornell, texas, washington, wisconsin)
│   ├── Yale_32x32.mat     # Yale face dataset
│   ├── IMDb datasets      # Movie-related datasets
│   └── cora.mat          # Cora citation network dataset
```

## Requirements

The project requires Python 3.x and the following dependencies:
- NumPy
- SciPy
- scikit-learn
- Matplotlib (for visualization)

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd SRBCC
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Algorithm

To run the main SRBCC algorithm:

```bash
python src/SRBCC_main.py
```

### Noise Evaluation

To perform noise evaluation:

```bash
python src/noise_eval.py
```

### WebKB and benchmark Dataset Processing

To process and evaluate benchmark datasets:

```bash
python src/webKB_runner.py
```

## Datasets

The repository includes several datasets:

1. **WebKB Datasets**
   - Cornell
   - Texas
   - Washington
   - Wisconsin

2. **IMDb Datasets**
   - Movies Keywords
   - Movies Actors

3. **Other Datasets**
   - Yale Face Dataset (32x32)
   - Cora Citation Network

