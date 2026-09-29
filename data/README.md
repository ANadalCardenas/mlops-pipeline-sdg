# 📁 Data Versions Overview

### v1. Baseline Dataset

* `v1/sdg.csv`: 500 rows, 30 anonymised features (`F1`–`F30`), a `Timestamp` and a 4-class `Target` (A, B, C, D)
* Represents initial system deployment
* Used to train the first production model

### v2. Growth and Distribution Drift

* Derived from v1
* Simulates:
  * Increased volume (+40% rows, resampled with small noise)
  * More rows of the minority class D
  * A small offset and extra noise on `F22` (the most predictive feature)
  * More `error` values in the status column `F1`
* Represents real-world data evolution after deployment
* Used to demonstrate monitoring, drift detection, and retraining


## Data generation

v1/sdg.csv is the original dataset. v2/sdg.csv is generated following the rules above using the python script `generate_v2.py` located in this directory.

To re-generate the data:

1. Remove the contents of v2 (if any).
2. Execute the script:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 generate_v2.py
```
