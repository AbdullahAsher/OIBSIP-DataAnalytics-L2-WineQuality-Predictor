# Wine Quality Prediction

Train and compare multiple classification models to predict wine quality from physicochemical properties (acidity, density, alcohol content, etc.).

## Dataset

`winequality-red.csv` — 1,599 red wine samples with 11 physicochemical features and a `quality` score (integer, 3–8) assigned by wine tasters.

**Features:**
fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol

**Target:** `quality` (raw score), transformed into two derived targets (see below).

## Project Structure

```
.
├── wine_quality_prediction.ipynb   # Main analysis notebook
├── winequality-red.csv             # Dataset (not included — add your own)
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

1. Place `winequality-red.csv` in the same directory as the notebook.
2. Launch Jupyter and open `wine_quality_prediction.ipynb`:
   ```bash
   jupyter notebook wine_quality_prediction.ipynb
   ```
3. Run all cells in order.

## Pipeline Overview

1. **Load & inspect** the dataset; check shape, dtypes, missing values, and summary statistics.
2. **EDA** — feature distributions, boxplots of features by quality, and a correlation heatmap.
3. **Class imbalance discussion** — the raw 6-class quality distribution is heavily skewed toward scores 5 and 6 (~82% of samples combined), with scores 3, 4, and 8 severely underrepresented.
4. **Feature engineering** — quality is binned into two alternative targets:
   - **Binary:** `quality >= 7` → Good (1), else → Bad (0) — used as the primary target.
   - **3-class:** `quality <= 4` → Low, `5–6` → Medium, `quality >= 7` → High — included for reference.
5. **Stratified train/test split** (75/25) to preserve the good/bad ratio in both sets, plus feature scaling (`StandardScaler`) for the distance/gradient-based models.
6. **Model training** — three classifiers, all with `class_weight="balanced"`:
   - Random Forest (`n_estimators=300`, `max_depth=10`)
   - SGD Classifier (log loss, i.e. logistic regression via SGD)
   - SVC (RBF kernel, `probability=True`)
7. **Evaluation** — accuracy, precision, recall, F1-score, classification report, and confusion matrix for each model.
8. **Feature importance** (Random Forest) — ranks which chemical properties drive predicted quality.
9. **Side-by-side model comparison** — bar chart of accuracy/precision/recall/F1 across all three models.
10. **Conclusion** — recommends a deployment model based on the comparison.

## Key Findings

- **Alcohol** has the strongest positive correlation with quality; **volatile acidity** has a notable negative correlation.
- Density/fixed acidity and free/total sulfur dioxide are strongly correlated with each other (multicollinearity), which affects linear models more than tree-based ones.
- Binning quality into a binary target turns an impractical 6-class imbalanced problem into a workable ~86% Bad / ~14% Good classification task.
- **Random Forest** delivers the strongest overall performance and is recommended for deployment, offering a good precision/recall balance on the minority ("Good") class plus interpretable feature importances. SVC is competitive on minority-class recall/F1 but is more hyperparameter-sensitive; SGD (linear) underperforms since the underlying relationships are non-linear.

## Notes

- `RANDOM_STATE = 42` is fixed throughout for reproducibility.
- Feature scaling is applied uniformly to all models for a clean comparison, even though Random Forest doesn't require it.
- The 3-class target is computed but not used for model training in this notebook — it's available for further experimentation.
