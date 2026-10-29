# Fraud detection project: how to run everything

## Files
| File | What it is |
|---|---|
| `fraud_detection_complete.ipynb` | Colab notebook: data, features, 6 models, thresholds, SHAP, leakage experiment, export |
| `app.py` | Streamlit demo (single-transaction form, batch CSV, performance tab) |
| `requirements.txt` | Packages for the app |

## Step 1: train (in Google Colab)
1. Open https://colab.research.google.com, then **File -> Upload notebook** and choose `fraud_detection_complete.ipynb`.
2. Get the dataset (https://www.kaggle.com/datasets/ealaxi/paysim1) into Colab using Option A, B, or C in the notebook's section 1. Option A (Kaggle API) avoids Google Drive problems.
3. Run all cells (**Runtime -> Run all**). Expect roughly **45 to 90 minutes** on the free runtime: the Random Forest, SMOTE, hyperparameter search (section 5b), and cross-validation cells are the slow ones. To make it faster, lower `N_ITER` (default 20) and `TUNE_FRAC` (default 0.15) in section 5b. To make the search more thorough, raise them.
4. The last cell downloads `fraud_artifacts.zip`.

## Step 2: run the app (on your laptop)
```bash
mkdir fraud_app && cd fraud_app
# put app.py and requirements.txt here, and unzip fraud_artifacts.zip into the same folder
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501.

Folder should look like:
```
fraud_app/
  app.py
  requirements.txt
  fraud_model.joblib
  model_meta.json
  sample_transactions.csv
  results_table.csv  cost_table.csv  leakage_comparison.csv
  figures/
```

## If something goes wrong
- **Version warning or unpickle error in the app:** install the same versions the notebook used. They are printed in `model_meta.json`, for example `pip install scikit-learn==1.6.1 xgboost==2.1.4`.
- **Colab runs out of memory:** in section 2 of the notebook, add `nrows=3_000_000` to `pd.read_csv(...)`. You will lose some data but everything else works.
- **Colab disconnects mid-run:** just rerun from the top. Each model cell is independent, so you can also rerun only the cells after the failure, as long as the earlier ones ran in the same session.

## Optional: put the app online for free
Push `app.py`, `requirements.txt`, and the files from the zip to a GitHub repository, then deploy it on Streamlit Community Cloud (https://share.streamlit.io). Keep versions pinned in `requirements.txt` so the model loads.

## Feature mode (important)
In the notebook's feature cell (section 2) there is one switch:

```python
FEATURE_MODE = 'pre'   # 'pre' = only information known before approval | 'full' = all PaySim columns
```
- `'pre'` (default) uses only information known before a transaction is approved: amount, hour, type, and the balances before the transaction.
- `'full'` also uses the balances after the transaction. PaySim cancels fraudulent transactions, so those columns reveal the outcome. Use it only if you want to show the comparison.

Section 9 of the notebook trains both and writes a comparison table, which the app shows in its *Model performance* tab.

## Using the app
- **Single transaction form:** type, amount, hour, the sender's and receiver's balance before the transaction, and (in `full` mode) their balances after it.
- **Batch CSV columns:** `type`, `amount`, `oldbalanceOrg`, `oldbalanceDest`, `step` (or `hour`), plus `newbalanceOrig` and `newbalanceDest` only in `full` mode. Optional `isFraud` shows precision and recall on your file.
