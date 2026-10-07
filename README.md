# hurdle-health
hurdle-health-trajectories
# Modelling Health Trajectories: Next-Diagnosis Prediction

Interview task for the **Biomedical and Health AI Scientist (KTP Associate)** role, University of Essex and Hurdle.

**Author:** Shivasmi Sharma

---

## What this project does

Each patient in the cohort is a timeline of diagnoses (three-character ICD-10 codes) with the age at which each was recorded, from birth to about age 80.

**Prediction target:** at every point in a patient's history, predict **which diagnosis is recorded next**, from about 920 possible codes.

**What the model sees at prediction time:**
- the patient's sex (recorded at age 0)
- every earlier diagnosis, in order, with the age at which it was recorded

**What the model does *not* see:**
- the age at which the next diagnosis happens (in practice you don't know when the next diagnosis will come)
- any diagnosis recorded on the same day as the one being predicted

**Models compared**, each using more information than the last:

| Model | Information used |
|---|---|
| Markov | the last diagnosis only |
| Logistic regression (baseline) | all earlier diagnoses, without order or timing |
| **GRU (main model)** | all earlier diagnoses, **with order and timing** |

The comparison answers one question: *does knowing the order and timing of a patient's diagnoses improve prediction?*

---

## How to run

The whole project is one Google Colab notebook. No installation is needed; Colab already includes every library used.

1. Open `notebook.ipynb` in Google Colab (use the **Open in Colab** badge, or upload it via File → Upload notebook).
2. In the Colab **Files** panel (folder icon on the left), upload the three data files to `/content/`:
   - `train.bin`
   - `val.bin`
   - `labels.csv`
3. *(Optional)* Runtime → Change runtime type → **T4 GPU**. CPU also works.
4. **Runtime → Run all.**

Cell 1 checks that the three files are found. All figures, results tables and the trained model are saved to `/content/outputs/`. Cell 23 downloads them as a zip.

**Run time:** about 1–2 minutes for EDA and preprocessing, and about 5 minutes for the models on CPU.

> **Note:** Colab deletes uploaded and generated files when the session ends. Download `outputs.zip` (cell 23) before closing.

### Dependencies

All are pre-installed on Google Colab:

| Library | Used for |
|---|---|
| numpy, pandas | data handling |
| matplotlib | plots |
| scipy | sparse matrices, statistical tests |
| scikit-learn | ROC AUC |
| PyTorch | logistic regression and GRU |

Also tested locally with Python 3.13, PyTorch 2.14, numpy 2.5, pandas 3.0, scikit-learn 1.9. All random seeds are fixed (`seed = 0`).

---

## Notebook structure

| Cells | Section | What it does |
|---|---|---|
| 1–5 | **Load** | Read the `.bin` files, decode tokens (stored value + 1 = vocabulary index), convert days to years |
| 6–6b | **Sanity checks** | File format, sorting, duplicates, patient overlap between splits, missing sex; fix for one patient |
| 7–12e | **EDA** | History length, age at diagnosis, top codes, body systems, time gaps, key conditions by age, sex and age patterns, data-quality checks |
| 13–18 | **Preprocessing** | Fit / tune / test split, patient sequences, vocabulary, prediction examples, leakage checks |
| 19–21c | **Preprocessing (continued)** | Long-tail analysis, scaling of time inputs, train-vs-test statistical tests, model inputs (multi-hot table and padded sequences) |
| 22–23 | **Save** | Save processed data, download outputs |
| 24–25 | **Evaluation setup** | Shared scoring rules applied identically to every model |
| 26–28 | **Models** | Markov, logistic regression, GRU |
| 29–39 | **Evaluation** | Training curves, test results with confidence intervals, paired comparison, top-k curve, key clinical conditions, calibration, subgroups, failure analysis, behavioural sanity checks, patient example |

---

## Design decisions and assumptions

### Data splits
- **`val.bin` is used as the held-out test set**, scored once at the very end.
- **`train.bin` is split by patient** into *fit* (90%, 6,429 patients) for training and *tune* (10%, 714 patients) for choosing settings (Markov smoothing, early stopping). The test set is never used for any decision.
- I kept the patient-level split provided by Hurdle rather than re-splitting (e.g. 70/30).

### Prediction examples
- Every diagnosis becomes one prediction example: *(history so far) → (next diagnosis)*. This gives 155,303 fit, 16,945 tune and 172,280 test examples.
- **Same-day diagnoses** (about 0.3%) are excluded from each other's history, since their order within a day is arbitrary.

### Data-specific choices
- **First occurrences only:** each code appears at most once per patient, so the data records the *first* time each disease was diagnosed. A code a patient already has cannot be their next diagnosis, so **every model's prediction for those codes is set to zero before scoring** (same rule for all models).
- **Vocabulary from the fit set only:** codes never seen in fit patients are mapped to an `UNK` input token in tune and test. The 108 test targets with such codes (0.06%) cannot be predicted and are **counted as misses**, not removed.
- **Scaling:** age is divided by 80 (range 0–1); the gap since the previous diagnosis is transformed with `log(1 + years)` because gaps range from days to decades. Diagnosis codes are categories and use learned embeddings (GRU) or multi-hot encoding (logistic regression).
- **No resampling for class imbalance:** over- or under-sampling would distort the predicted probabilities, and calibration is one of the evaluation goals. Imbalance is handled in evaluation instead (per-condition and per-frequency results).

### Assumptions and things worth flagging
- **One patient (402867) had no sex record.** Their history includes prostate cancer (C61) and no female-specific diagnoses, so they were recorded as **male**. Strictly, this uses a diagnosis at age 65 to fill in something known at birth; it affects 1 of 7,143 patients.
- **Sex-inconsistent diagnoses:** 17 women have a male-only diagnosis and 38 men have a female-only diagnosis (0.03% of all diagnoses). Kept, and treated as noise in the synthetic data. This also means the sex inference above is strong evidence, not proof.
- **End of follow-up is not recorded.** 9.2% of records end before age 75; the data cannot tell whether these patients died or left the records. The next-diagnosis task does not need follow-up length, but a time-to-event task would.
- **Train vs test comparability** was checked with chi-square (sex) and Mann-Whitney U (number of diagnoses, age at last record; these are skewed, so a t-test is not appropriate). Only age at last record reached p = 0.017, with medians differing by about one month: statistically but not practically significant.

---

## Results (test set: 7,143 patients, 172,280 predictions)

| Model | Top-1 | Top-5 | Top-10 | Top-10 95% CI | Log-loss |
|---|---|---|---|---|---|
| Markov | 4.65% | 15.37% | 23.46% | 23.3–23.7 | 5.37 |
| Logistic regression | 4.70% | 15.51% | 24.18% | 24.0–24.4 | 5.38 |
| **GRU** | **5.43%** | **16.82%** | **25.84%** | **25.6–26.0** | **5.18** |

*Top-k accuracy = how often the true next diagnosis is among the model's k most likely codes. Log-loss: lower is better. For reference, always guessing the most common code (hypertension) gives about 2.4% top-1.*

**Is the GRU really better?** Paired bootstrap over patients (500 resamples):
- GRU − logistic regression: **+1.66 points** top-10 accuracy (95% CI +1.49 to +1.85)
- GRU − Markov: **+2.37 points** (95% CI +2.21 to +2.53)

Neither interval includes zero, so the improvement is unlikely to be chance.

### Clinical conditions (AUC for "is the next diagnosis this condition?")

| Condition | Test cases | Markov | Logistic | GRU |
|---|---|---|---|---|
| Coronary heart disease (I20–I25) | 3,424 | 0.621 | 0.708 | **0.765** |
| Atrial fibrillation (I48) | 1,420 | 0.656 | 0.709 | **0.723** |
| Heart failure (I50) | 508 | 0.717 | 0.813 | **0.842** |
| Stroke (I63–I64) | 534 | 0.637 | 0.635 | **0.711** |
| Type 2 diabetes (E11) | 1,234 | 0.631 | 0.760 | 0.761 |
| Chronic kidney disease (N18) | 988 | 0.622 | 0.684 | 0.682 |

The GRU's advantage is clearest for **cardiovascular conditions**, where order and timing matter (e.g. heart failure following a heart attack). For type 2 diabetes and chronic kidney disease, knowing *which* diagnoses a patient has is as informative as knowing their order.

### Other checks
- **Calibration:** predicted probabilities closely match observed frequencies for both logistic regression and GRU (expected calibration error ≤ 0.3 percentage points).
- **Subgroups:** the GRU is best in every subgroup by sex, age and history length. All models are weakest at ages 20–39.
- **Uses timing:** setting age and gap inputs to zero at test time drops GRU top-10 accuracy from 25.8% to 17.4%. (These inputs are out-of-distribution, so the drop overstates the effect somewhat.)
- **Uses sex sensibly:** with an identical history, changing only the sex token moves the probability of menopause (N95) and breast cancer (C50) up for women and of prostate disease (N40, C61) up for men.

---

## Limitations

1. **Rare diagnoses are almost never predicted.** Top-10 accuracy is 53% for the 37 codes seen 1,000+ times in training, but close to 0% for the ~520 codes seen fewer than 50 times. 18% of the GRU's top-1 guesses are hypertension.
2. **Predictions are not time-bounded.** "9% chance the next diagnosis is heart disease" says nothing about *when*. For screening, a fixed-window risk (e.g. 5-year) or a time-to-next-event model would be needed.
3. **Small, synthetic cohort.** About 7,000 patients per split. Results may differ on real records, and the sex-inconsistent diagnoses show the data contains some noise.
4. **Censoring is unknown.** Records that end early cannot be distinguished as death or loss to follow-up.

## Possible next steps

- Predict **time to next diagnosis** alongside the code (as in Delphi-2M), giving time-bounded risks.
- Help rare codes by sharing information between related diagnoses: ICD-10 chapter embeddings, or embeddings of the code descriptions from a biomedical language model (e.g. BioBERT).
- Compare a small **transformer** (BEHRT / Delphi-style) against the GRU.
- Add other data types Hurdle works with (e.g. lab values, omics) as extra inputs.

---

## Repository contents

```
├── notebook.ipynb    # all code, with outputs
├── README.md         # this file
├── report.pdf        # short report / slides
└── figures/          # all plots produced by the notebook
```

The data files (`train.bin`, `val.bin`, `labels.csv`) are **not included**, as they belong to Hurdle. Place them in `/content/` as described above.

## References

- Shmatko, A., Gerstung, M. et al. (2025). Learning the natural history of human disease with generative transformers. *Nature*.
- Choi, E. et al. (2016). Doctor AI: Predicting clinical events via recurrent neural networks. *Machine Learning for Healthcare Conference*.
- Li, Y. et al. (2020). BEHRT: Transformer for electronic health records. *Scientific Reports*.
- Christodoulou, E. et al. (2019). A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. *Journal of Clinical Epidemiology*.
- Van Calster, B. et al. (2019). Calibration: the Achilles heel of predictive analytics. *BMC Medicine*.
