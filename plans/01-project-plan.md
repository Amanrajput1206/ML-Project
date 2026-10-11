# Project Plan — Domestic Flight Price Prediction (ML Project-I)

## 1. Context

- **Course:** AIT 511 Machine Learning, Project-I (Kaggle competition format).
- **Team:** #12 **Savage** — Swapan Aggarwal, Anshuman Joshi, Vijay Kumar S, Aman Rajput. TA: Akshat Batra.
- **Deadline:** **15 Oct 2026** (today is 9 Oct, so about 6 days).
- **Two competitions per team** (from `docs/Machine Learning Project-I.xlsx`):
  - Regression: `kaggle.com/t/ec0b1e254b7b442cafc64d9073d7dafe`. This is the flight-price data in `dataset/regression/`.
  - Classification: `kaggle.com/t/d351b70c13654f209350dc08629f8955`. Vehicle-class data in `dataset/classification/`. **This plan doesn't cover it; see `02-classification-plan.md`.**
- **Grading (15M):** Preprocessing 2 · Variety of models 3 · Viva + Report 5 · Private-LB performance 5.
  The public leaderboard doesn't count, so **a sound CV setup matters more than chasing the public LB.**
- **Hard rules:** only use models taught in Part 1; try several models (one good model isn't enough); use fixed random seeds everywhere; put a GitHub link in the report; keep our own code (no sharing between teams).

## 2. What the data looks like (initial profiling)

| | train | test |
|---|---|---|
| Rows | 15,589 | 3,898 |
| Target | `price` (3,801 – 323,177; median 10,183, mean 29,817 → **heavily right-skewed**) | — |

Columns: `airline, flight, source_city, departure_time, stops, arrival_time, destination_city, flight_class, duration, days_left`.

Findings that drive the plan:

1. **`airline` is about 20% missing, but we can fill it back in exactly.** The flight-code prefix maps 1-to-1 to an airline (AI/IX → Wayfarer Aviation, 6E → Oceanic Airlines, QP → Sandpiper Air, …, 20 prefixes in total, with no conflicts). This is a strong preprocessing point to make in the report and viva.
2. **`duration` is about 20% missing.** For 3,784 of the 3,871 missing rows (train + test), the same (flight, source, destination, stops) group has a known duration, so we fill from the group median. The few rows left get the (route, stops) median.
3. **`flight_class`** (Economy 59% / Business 41%) almost certainly splits the price into two regimes. That's why the mean is about 3× the median. We should look at the target per class and consider separate models per class, or class-interaction terms.
4. `stops` is ordinal (zero / one / two_or_more), but `two_or_more` is rare (66 rows).
5. `flight` has about 1,070 distinct values, and 35 test rows use flights never seen in train. We'll use the prefix (airline) and maybe the numeric part, not the raw code one-hot encoded.
6. 314 train rows have identical features but may have different prices. This is a noise floor to mention under EDA.
7. Low-cardinality categoricals: 6 cities × 2, 6 time slots × 2. `days_left` ranges 1–57 and price likely depends on it non-linearly (fares jump close to departure).
8. **Open question:** Kaggle's evaluation metric (RMSE / MAE / R² / RMSLE). Check it on the competition page. It decides whether to train on log(price) and how to transform predictions back.

## 3. Models in scope (Part 1 of the course)

Taken from the decks in `../PPTs/`: Linear Regression (least squares, polynomial, Ridge, Lasso mentioned), MLE (least squares = Gaussian MLE), and **Decision Trees, including regression trees and pruning** (29 Sept deck).
Bayes / Naive Bayes / Logistic Regression are classifiers, so they belong to the classification competition, not this one.
**Random Forest and boosting are only named in the intro syllabus and aren't taught yet, so don't use them** unless the TA confirms they're allowed.

Planned model ladder (every row gets a "best score" entry in the report table):

| # | Model | Why it's there |
|---|---|---|
| 0 | Baseline: predict the median / per-class mean | Reference point |
| 1 | OLS linear regression (one-hot features) | Plain least squares |
| 2 | OLS on log(price) | Shows the effect of handling skew; ties to MLE / Gaussian noise |
| 3 | Polynomial / interaction features + OLS | Non-linear `days_left`, class × route; shows overfitting |
| 4 | Ridge (L2), tuned α | Regularization, bias–variance |
| 5 | Lasso (L1), tuned α | Feature selection angle |
| 6 | Separate linear models per `flight_class` | Data-driven model structure |
| 7 | Decision Tree Regressor | Captures interactions without hand-made features |
| 8 | Pruned tree (cost-complexity `ccp_alpha`, max_depth, min_samples_leaf) | Pruning from the DT lecture |

## 4. Validation & reproducibility

- `SEED = 42`, used in every split and every model.
- **5-fold KFold CV on train.** All preprocessing that's fitted on data (imputation medians, scaling, encoders) goes inside an sklearn `Pipeline`/`ColumnTransformer`, so it's re-fitted on each fold. Otherwise the CV score leaks information from the validation fold.
  Exception: filling `airline` from the flight prefix is a fixed rule, not something learned, so it can run before CV.
- Report CV mean ± std in the competition metric, plus RMSE / MAE / R² for discussion.
- Pick the final model by **CV score**, using the public LB only as a sanity check, because private-LB performance is what's graded.
- If we train on log(price), transform predictions back with `exp`, and check whether that hurts an RMSE-on-raw-price metric. If it does, compare against training on raw price.

## 5. Repo layout (to create in the next step)

```
Project/
  dataset/            # given, git-ignored if large
    regression/       #   train/test/sample_submission + dataset-details.md
    classification/   #   same layout (see 02-classification-plan.md)
  docs/               # given
  plans/              # this plan
  notebooks/
    01_eda.ipynb
    02_preprocessing.ipynb
    03_models.ipynb
  src/
    config.py         # SEED, paths
    preprocess.py     # airline recovery, duration imputation, feature pipeline
    evaluate.py       # CV helper -> appends row to results/experiments.csv
    train_<model>.py  # or one script with a --model flag
  results/
    experiments.csv   # model, params, cv_mean, cv_std, public_lb
    figures/
  submissions/        # <date>_<model>.csv
  report/
  requirements.txt
  README.md
```

## 6. Phases & schedule

| Day | Phase | Output |
|---|---|---|
| 9 Oct | **0. Setup:** confirm the Kaggle metric and daily submission limit; `pip install pandas numpy scikit-learn matplotlib seaborn jupyter`; create a GitHub repo and invite the team | env, repo, `requirements.txt` |
| 9–10 Oct | **1. EDA:** distributions (price, log price per class), price vs `days_left`/`duration`/`stops`/airline/route, missing-value patterns, correlations, train vs test distribution check | `01_eda.ipynb` + a figure folder + a bullet list of observations for the report |
| 10 Oct | **2. Preprocessing:** airline recovery, duration imputation, encoding (one-hot for nominal, ordinal for `stops`), scaling for linear models, log target, optional engineered features (`days_left` bins/poly, class × route) | `preprocess.py`, first valid submission (baseline) to test the pipeline end to end |
| 11–12 Oct | **3. Models 1–8** with CV; log every run to `experiments.csv` | model comparison table |
| 12–13 Oct | **4. Tuning:** grid search over α (Ridge/Lasso), polynomial degree, tree depth / leaf size / `ccp_alpha`; plot validation curves (bias–variance) | tuned scores, curves for the report |
| 13–14 Oct | **5. Submissions:** submit the top 2–3 CV models; pick the 2 final submissions for the private LB (best CV + a more conservative one, if Kaggle allows choosing) | public LB scores recorded |
| 14–15 Oct | **6. Report + GitHub:** follow the 8 required sections; model table with best score per model; best public-LB model named; README with steps to reproduce | report PDF, public repo link |
| Ongoing | **7. Viva prep:** every member can explain the airline recovery, why log target, why CV inside the pipeline, Ridge vs Lasso, tree pruning, least squares = Gaussian MLE | — |

## 7. Suggested team split (4 people)

- A: EDA + report sections 1–2
- B: preprocessing pipeline + CV harness (blocks everyone else, so do it first)
- C: linear-family models (1–6) + tuning
- D: tree models (7–8) + tuning + submissions
  Everyone reviews the final report and does viva prep together.

## 8. Open questions before starting

1. What's the Kaggle evaluation metric and the daily submission limit? (Check the competition Overview/Evaluation tab.)
2. ~~Is the classification competition going to be planned and done in parallel?~~ Yes. Data was added on 11 Oct; see `02-classification-plan.md`.
3. Are KNN / Random Forest / boosting allowed? Assume **no** unless the TA says yes.
4. Who on the team takes which part? (See §7.)
