# DSLR

**Data Science × Logistic Regression: Harry Potter and the Data Scientist.**

The Sorting Hat is bewitched. This project rebuilds it: explore the Hogwarts
students' grades, visualize them, then train a one-vs-all logistic regression
that sorts each student into a house, with at least 98% accuracy.

## Setup

Requires Python 3.10+.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

| Step | Command | Output |
|---|---|---|
| Data analysis | `python describe.py datasets/dataset_train.csv` | Statistics table in the terminal |
| Histogram | `python histogram.py datasets/dataset_train.csv` | Window with one histogram per course |
| Scatter plot | `python scatter_plot.py datasets/dataset_train.csv` | Window with the most correlated pair |
| Pair plot | `python pair_plot.py datasets/dataset_train.csv` | Window with the scatter plot matrix |
| Training | `python logreg_train.py datasets/dataset_train.csv [batch\|minibatch\|sgd]` | `weights.json` |
| Prediction | `python logreg_predict.py datasets/dataset_test.csv weights.json` | `houses.csv` |

Every script validates its input and exits with a clear error message (exit
code 1) on a missing file, malformed CSV, missing columns or an invalid model
file.

## V.1 Data analysis: `describe.py`

Prints, for every numerical feature, the same statistics as the subject's
example: Count, Mean, Std, Min, 25%, 50%, 75% and Max. Missing values are
ignored per feature.

No library statistic is used. Every value is computed by hand in
`toolkit/stats.py`:

* **Std** is the sample standard deviation (divides by `n - 1`).
* **Percentiles** use linear interpolation between the two closest ranks:
  position `p / 100 × (n - 1)` in the sorted values.

The output matches pandas' `describe()` exactly.

## V.2 Data visualization

### Histogram: which course has a homogeneous score distribution between all four houses?

**Arithmancy**, closely followed by **Care of Magical Creatures**.

In `histogram.py`, each house is drawn as a density histogram over the same
bins, so houses of different sizes can be compared. For these two courses the
four houses overlap almost perfectly. The spread between the house means is
only 0.07 standard deviations for Arithmancy and 0.15 for Care of Magical
Creatures. Every other course is above 1.8.

### Scatter plot: what are the two features that are similar?

**Astronomy** and **Defense Against the Dark Arts**.

`scatter_plot.py` computes the Pearson correlation of every pair of courses
(by hand, `toolkit/stats.py`) and plots the pair with the largest |r|. These
two have r = -1.00: Defense Against the Dark Arts is exactly Astronomy scaled
by -1/100. They carry the same information.

### Pair plot: which features will you use for the logistic regression?

`pair_plot.py` draws every course against every other, colored by house.
From it we keep the courses whose clusters separate the houses, and drop the
ones that don't help:

| Dropped | Why |
|---|---|
| Arithmancy, Care of Magical Creatures | Homogeneous across houses (see the histogram), so they carry no information about the house |
| Defense Against the Dark Arts | Duplicate of Astronomy (r = -1) |
| Potions | Houses overlap heavily, and adding it does not improve accuracy |

**Kept (9):** Astronomy, Herbology, Divination, Muggle Studies, Ancient Runes,
History of Magic, Transfiguration, Charms, Flying.

## V.3 Logistic regression

### The model

For one house, logistic regression estimates the probability that a student
belongs to it:

```
h(x) = g(θᵀx)        g(z) = 1 / (1 + e^(-z))
```

`g` (the sigmoid) squashes any score into a probability between 0 and 1.
Training minimizes the log loss:

```
J(θ) = -1/m Σ [ y log h(x) + (1 - y) log(1 - h(x)) ]
```

with gradient descent, using the partial derivative from the subject:

```
∂J/∂θⱼ = 1/m Σ (h(xᵢ) - yᵢ) xᵢⱼ
θⱼ := θⱼ - α ∂J/∂θⱼ
```

### One-vs-all

Logistic regression separates two classes, and there are four houses. So we
train **four** classifiers, one per house: "Gryffindor or not", "Hufflepuff
or not", and so on. To predict, each classifier gives its probability and the
student goes to the house with the highest one.

### Preprocessing

* **Standardization:** each feature is rescaled to mean 0 and standard
  deviation 1 (z-score), using the training set's mean and std. Without it,
  Arithmancy (around 50,000) and Care of Magical Creatures (around 0) would
  need very different learning rates, and gradient descent would converge
  badly. The means and stds are saved in `weights.json` so the test set is
  scaled exactly like the training set.
* **Missing values:** 349 of the 1,600 training students miss at least one
  grade. Dropping them would lose 22% of the data, so a missing grade is
  replaced by the feature's mean (0 after scaling), which leaves that feature
  neutral in the prediction.

### Files

* `logreg_train.py` trains the four classifiers and writes `weights.json`:
  the feature list, the scaling means and stds, the houses, and each house's
  bias `θ₀` and weights `θ`.
* `logreg_predict.py` loads `weights.json`, scales the test set the same way,
  and writes `houses.csv`:

  ```
  Index,Hogwarts House
  0,Hufflepuff
  1,Ravenclaw
  ...
  ```

### Accuracy

Training accuracy is **98.19%**, and 10-fold cross-validation gives the same
average.

That looks like the ceiling for this data. The same 29 students are
misclassified whatever features, learning rate or number of epochs we use, and
a reference logistic regression library scores exactly the same. Those
students' grades look like those of another house, which no linear model can
fix.

## Bonus

### More fields in `describe.py`

Six extra rows after Max, also computed by hand:

| Field | Meaning |
|---|---|
| Missing | Number of empty values |
| Var | Sample variance (Std²) |
| Range | Max − Min |
| IQR | Interquartile range, 75% − 25% |
| Skew | Asymmetry of the distribution: 0 is symmetric, > 0 has a longer right tail |
| Kurtosis | Excess kurtosis, how heavy the tails are: 0 is like a normal distribution |

Skew and Kurtosis use the bias-corrected sample formulas, the same as pandas.

### Optimization algorithms

`logreg_train.py` takes an optional optimizer (default `batch`):

| Optimizer | One update uses | Learning rate | Epochs |
|---|---|---|---|
| `batch` | All 1,600 students | 0.1 | 1000 |
| `minibatch` | 32 random students | 0.1 | 50 |
| `sgd` | 1 random student | 0.01 | 10 |

* **Batch gradient descent** computes the exact gradient each step. It is
  smooth and stable but each step is expensive.
* **Stochastic gradient descent (SGD)** updates after every single student. Each step
  is cheap but noisy, so it uses a smaller learning rate.
* **Mini-batch gradient descent** is in between: less noisy than SGD, and
  cheaper per step than batch.

Students are reshuffled every epoch with a fixed seed, so results are
reproducible. All three reach the same 98.19% accuracy and the same
predictions on the test set.

## Project structure

```
.
├── describe.py            V.1   statistics table
├── histogram.py           V.2.1 homogeneous course
├── scatter_plot.py        V.2.2 similar features
├── pair_plot.py           V.2.3 feature selection
├── logreg_train.py        V.3   training -> weights.json
├── logreg_predict.py      V.3   prediction -> houses.csv
├── datasets/
│   ├── dataset_train.csv
│   └── dataset_test.csv
└── toolkit/
    ├── stats.py           hand-written statistics
    ├── load.py            CSV loading and NaN handling
    ├── groups.py          numeric columns, houses, grouping
    ├── preprocess.py      feature checks and z-score scaling
    ├── logreg.py          sigmoid, gradient descent, prediction
    ├── model_io.py        save and load weights.json
    └── plot.py            scatter plot helper
```
