# Customer Lifetime Value & Segmentation Engine

An online retailer treats every customer the same. This project groups customers by behaviour (RFM clustering)
and predicts how much each will spend in the next 90 days (regression), so marketing knows who to reward,
who to win back and where to spend its budget.

**Live app:** _paste your Streamlit Community Cloud link here_

## What the app does
* **Segment dashboard:** size, behaviour profile, share of revenue and predicted next-90-day revenue of every segment, plus the recommended action.
* **Customer lookup:** type a Customer ID to see the segment, predicted 90-day spend and the recommended action.
* **Try a new customer:** enter days since last order, number of orders and total spend to get a segment and forecast from the saved models.

## Project structure
```
clv-segmentation-app/
├── app.py                  # the Streamlit app (segment names and actions are edited at the top)
├── requirements.txt        # pinned dependencies (scikit-learn must match the version that saved the models)
├── README.md
├── models/
│   ├── segmentation_pipeline.joblib      # log1p + scaling + K-Means (k=4): raw R/F/M -> cluster
│   └── spend_regression_pipeline.joblib  # log1p + scaling + Random Forest: raw R/F/M -> 90-day spend
├── data/
│   ├── customer_forecast.csv   # one row per customer: R, F, M, cluster, predicted_spend_90d
│   └── model_info.json         # snapshot date, number of clusters
└── notebooks/
    ├── customer_segmentation_clustering.ipynb   # full analysis: EDA, clustering, regression, insights
    └── rfm_utils.py            # cleaning + RFM functions used by the notebook
```

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Troubleshooting
* **Nothing opens / "missing ScriptRunContext" warnings:** the app was started with `python app.py`. Use `streamlit run app.py`.
* **`InconsistentVersionWarning` (scikit-learn):** the `.joblib` models must be loaded with the same scikit-learn version that saved them (1.9.1).
  Run `pip install scikit-learn==1.9.1`. If pip cannot find that version, your Python is too old: use a newer Python (3.11+), or re-run
  the notebook in your own environment and copy the new `.joblib` files into `models/`.

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository (`app.py` at the repository root).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Pick the repository, branch `main`, main file `app.py`, then **Deploy**.
4. Copy the public link into this README and your LinkedIn post.

## Method in short
* **Cleaning:** drop missing Customer IDs, duplicates, cancellations (invoice starts with C), non-positive quantity or price, and non-product codes (postage, manual adjustments).
* **Segmentation:** Recency, Frequency, Monetary per customer, log1p + standard scaling, K-Means++ with k chosen by silhouette, elbow and stability (compared with K-Medoids and Ward hierarchical). Test silhouette about 0.32.
* **Spend forecast:** features from orders before a cut-off date (11 Sep 2010), target = spend in the following 90 days. Random Forest, test MAE about £741 per customer (baseline £1,139).

## Limitations
* One year of data: seasonality (the November peak) cannot be separated from real behaviour change.
* The forecast ranks customers and segments well but is not an exact budget; it under-predicts very large customers and is probably too high outside the Christmas peak.
* Only R, F, M are used; product mix, country and discounts are ignored.
* Cancellations are removed, not subtracted, so spend is gross.

## Re-training
Re-run the notebook, copy the new `.joblib` files into `models/` and `customer_forecast.csv` into `data/`.
If the number of clusters changes, update the playbook at the top of `app.py`.
