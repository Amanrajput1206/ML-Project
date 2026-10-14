# Project Plan — Vehicle Class Classification (ML Project-I)

## 1. Context

- **Course:** AIT 511 Machine Learning, Project-I (Kaggle competition format). This is the second of the two competitions; the regression one is in `01-project-plan.md`.
- **Team:** #12 **Savage**: Swapan Aggarwal, Anshuman Joshi, Vijay Kumar S, Aman Rajput. TA: Akshat Batra.
- **Deadline:** **15 Oct 2026** (written 11 Oct, so about 4 days, running in parallel with regression).
- **Competition link:** `kaggle.com/t/d351b70c13654f209350dc08629f8955` (from `docs/Machine Learning Project-I.xlsx`).
- **Data:** `dataset/classification/` (`train.csv`, `test.csv`, `sample_submission.csv`; column list in `dataset-details.md`).
- **Grading and hard rules are the same as regression** (see `01` §1 and `docs/competition-specific-rules.md`): only Part-1 models, several models, fixed seeds, GitHub link, private-LB is what's graded.

## 2. What the data looks like (initial profiling)

| | train | test |
|---|---|---|
| Rows | 10,000 (**7,500 unique**) | 2,500 |
| Columns | `vehicle_id` + 20 numeric features + `vehicle_class` | `vehicle_id` + 20 features |
| Missing values | none | none |

Findings that drive the plan:

1. **Train rows 7501–10000 are exact copies of earlier rows.** There are 2,500 pairs with identical features *and* identical labels, so only 7,500 rows are unique. **Drop the duplicates before any CV split.** If you keep them, a row and its copy can land in different folds, so the model is validated on data it has already seen and the CV score comes out too high (leakage). This is the main preprocessing point for the report and viva. (Dropping them also stops those 2,500 rows from getting double weight in training.)
2. **The target is balanced across 3 classes:** 0 → 3,389, 1 → 3,326, 2 → 3,285. Accuracy is a fair metric and no resampling or class weights are needed. Still use *stratified* folds.
3. **Classes (official, from Kaggle): 0 = Sedan, 1 = SUV, 2 = Sports Car.** The per-class medians match:
   - Sedan: best fuel efficiency, lowest hp/torque, mid-sized.
   - SUV: heaviest, longest, widest, most cargo, highest ground clearance, longest braking distance.
   - Sports Car: most hp/torque/cylinders, 0–100 in ~4.6 s, lowest clearance and drag.
   Several features separate the classes clearly on their own (clearance, acceleration, cargo, weight), so expect high accuracy on clean data. The interesting question is *which* model and why, not squeezing out the last 0.1%.
3b. **⚠ The hidden (private) evaluation data has extra random measurement noise** (Kaggle: "a model that performs well on the training data is not necessarily the model that will perform best on the final evaluation"). **This is the most important fact for model choice.**
   - Checked 11 Oct: the public `test.csv` has the same feature spreads as train (std ratio 0.98–1.04 on every feature), which rules out *large* noise there. **Revised 12 Oct (EDA 2.7):** the floor spikes (e.g. 23% of train at ground clearance 110 mm vs 17% of test) are partly smeared in test, which is consistent with *mild* noise in the public test too; its size can't be pinned down. The public LB may reflect a little noise, but it can't judge robustness to larger noise, and the private set may be noisier.
   - So we have to **simulate the noise in CV ourselves**: add Gaussian noise to the *validation* fold's features (σ = k × each feature's std, for k ∈ {0, 0.05, 0.1, 0.2, 0.3}) and plot accuracy vs k for each model. Pick the model that stays accurate as k grows, not the one with the best k = 0 score.
   - Expectations to test: a deep unpruned tree is fragile, because a small shift can push a point across a hard threshold. Smoother models (logistic, LDA/NB) and models that use *many* features at once (averaging out per-feature noise) should degrade more gracefully. Pruning and stronger regularization (smaller `C`) should help robustness even if they cost a little clean accuracy. Also try **training-time noise augmentation** (add the same kind of noise to training rows) as a robustness technique.
4. **All 20 features are numeric, on very different scales** (drag 0.22–0.42 vs weight 1,100–2,600 kg). Logistic regression (especially with regularization) needs standardization. Trees and Naive Bayes don't.
5. **Expect strong correlation between features** (weight/length/wheelbase/tank/width; hp/torque/displacement/cylinders; city vs highway efficiency). This breaks the *naive* independence assumption, which makes it a good case for comparing Naive Bayes with a full-covariance Bayes classifier. Check with a correlation heatmap in EDA.
6. No test row exactly matches a train row, and test has no internal duplicates. The sample submission ids (10001–12500) are in the same order as test.
7. **Metric: accuracy** (confirmed on Kaggle). With balanced classes that's fair. **Daily submission limit:** regression shows 100/day; check this competition's submit dialog.

## 3. Models in scope (Part 1 of the course)

From the decks in `../PPTs/`: the Bayes classifier → Naive Bayes → multiclass classification (10 Sept), logistic regression (15/17 Sept), decision trees with pruning (29 Sept). Linear/Ridge/Lasso regression are for the regression competition.
**Update (TA, 11 Oct):** models built on concepts taught before the deadline are allowed, as long as we can explain how they work in the viva. Decision trees were taught, so **tree ensembles are allowed: Random Forest, XGBoost, CatBoost**. SVM and neural networks come after the midsem and stay **out**. KNN wasn't mentioned, so keep it out unless asked.

Planned model ladder (each row gets a "best score" entry in the report table):

| # | Model | Why it's there |
|---|---|---|
| 0 | Baseline: majority class | Reference point (about 34% accuracy) |
| 1 | Gaussian Naive Bayes | Bayes classifier with the independence assumption |
| 2 | Bayes classifier with full-covariance Gaussian class-conditionals (QDA; LDA if a shared covariance is used) | Drops the naive assumption, so it shows what correlated features cost NB |
| 3 | Logistic regression, multiclass (softmax / one-vs-rest) | Discriminative vs generative; multiclass from the 10 Sept deck |
| 4 | Logistic + L2, tuned `C` | Regularization, bias–variance |
| 5 | Logistic + L1, tuned `C` | Sparse weights, so it picks out the features that matter |
| 6 | Logistic on polynomial / interaction features | Non-linear boundaries with a linear model; shows overfitting |
| 7 | Decision Tree Classifier | Axis-aligned splits, no scaling needed, readable rules |
| 8 | Pruned tree (`ccp_alpha`, `max_depth`, `min_samples_leaf`) | Pruning from the DT lecture |

Note: if model 2 (QDA/LDA) isn't considered "taught", keep it only as a supporting comparison and confirm with the TA.

## 4. Validation & reproducibility

- `SEED = 42`, used in every split and every model.
- **Step 0: `train.drop_duplicates(subset=features)`, giving 7,500 rows.** Do this before splitting.
- **`StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`** on the deduplicated train set.
- Anything fitted on data (the `StandardScaler`, polynomial expansion) goes inside an sklearn `Pipeline` so it's re-fitted on each fold.
- Report CV accuracy mean ± std, plus macro-F1 and a confusion matrix (which classes get mixed up, e.g. Sedan vs SUV?).
- **Also report noisy-CV accuracy** (validation features perturbed at each noise level k, see §2 item 3b), as a robustness curve per model.
- Pick the final model by **noisy-CV accuracy** (at a moderate k, e.g. 0.1–0.2) together with clean CV. The public LB is only a format/sanity check here, because it doesn't contain the extra noise.
- Submission format: `vehicle_id,vehicle_class` with integer labels, same row order as `test.csv`.

## 5. Repo layout

Same structure as `01` §5, split per competition:

```
Project/
  dataset/
    regression/       # flight price
    classification/   # vehicle class
  src/
    config.py         # SEED, paths for both tasks
    evaluate.py       # shared CV helper -> appends to results/<task>/experiments.csv
    classification/   # preprocess.py, models
    regression/
  notebooks/
    classification/   # 01_eda.ipynb, 02_models.ipynb
    regression/
  results/{regression,classification}/
  submissions/{regression,classification}/   # <date>_<model>.csv
```

## 6. Phases & schedule (in parallel with regression)

| Day | Phase | Output |
|---|---|---|
| 11 Oct | **0. Setup + EDA:** confirm the metric and submission limit; class balance, per-class boxplots for each feature, correlation heatmap, duplicate analysis, train vs test distribution check | `01_eda.ipynb`, figures, list of observations |
| 11–12 Oct | **1. Preprocessing + baseline:** dedupe, scaling pipeline, majority baseline submission to test the pipeline end to end | `preprocess.py`, first valid submission |
| 12–13 Oct | **2. Models 1–8** with stratified CV; log every run | comparison table |
| 13 Oct | **3. Tuning:** `C` for L1/L2, polynomial degree, tree depth / leaf size / `ccp_alpha`; validation curves | tuned scores, curves |
| 14 Oct | **4. Submissions:** top 2–3 CV models; pick the final 2 for the private LB | public LB scores recorded |
| 14–15 Oct | **5. Report + GitHub:** the same 8 required sections | report, repo link |
| Ongoing | **6. Viva prep:** why public LB ≠ private LB here (noise), how we simulated it, why dedupe before CV, NB independence vs full covariance, generative vs discriminative, why scale for logistic but not trees, pruning | — |

## 7. Open questions

1. ~~What's the metric?~~ Accuracy. Submission limit: check the submit dialog (regression is 100/day).
2. Are QDA/LDA acceptable as "Bayes classifier" models? ~~Are KNN/RF/boosting allowed?~~ Resolved 11 Oct: RF/XGBoost/CatBoost yes (must be explainable in the viva), SVM/NN no, KNN unconfirmed.
3. ~~Class meanings?~~ 0 Sedan, 1 SUV, 2 Sports Car.
3b. How large is the private-set noise? Unknown, so we use a range of k values. Worth asking the TA whether a rough magnitude can be shared.
4. Is there one report covering both competitions, or two separate reports? (This decides whether the report prototype gets a classification section or a separate file.)
