import numpy as np
import pandas as pd


def clean_transactions(df, verbose=True):
    """Rule-based cleaning. Nothing is learned from the data, so it is safe to run before the train/test split."""
    steps = []
    d = df.copy()
    steps.append(('raw data', len(d)))
    d = d.dropna(subset=['Customer ID']);                              steps.append(('drop missing Customer ID', len(d)))
    d = d.drop_duplicates();                                           steps.append(('drop duplicate rows', len(d)))
    d = d[~d['Invoice'].astype(str).str.startswith('C')];              steps.append(('drop cancelled invoices (C...)', len(d)))
    d = d[(d['Quantity'] > 0) & (d['Price'] > 0)];                     steps.append(('drop Quantity<=0 or Price<=0', len(d)))
    d = d[d['StockCode'].astype(str).str.match(r'^\d{5}')];           steps.append(('drop non-product codes', len(d)))
    d = d.copy()
    d['Customer ID'] = d['Customer ID'].astype(int)
    d['Spend'] = d['Quantity'] * d['Price']
    if verbose:
        log = pd.DataFrame(steps, columns=['step', 'rows_left'])
        log['rows_removed'] = (-log['rows_left'].diff()).fillna(0).astype(int)
        print(log.to_string(index=False))
    return d


def build_rfm(clean, snapshot_date=None):
    """One row per customer: Recency (days since last order), Frequency (number of orders), Monetary (total spend)."""
    if snapshot_date is None:
        snapshot_date = clean['InvoiceDate'].max().normalize() + pd.Timedelta(days=1)
    g = clean.groupby('Customer ID')
    rfm = pd.DataFrame({
        'Recency': (snapshot_date - g['InvoiceDate'].max()).dt.days,
        'Frequency': g['Invoice'].nunique(),
        'Monetary': g['Spend'].sum(),
    })
    return rfm, snapshot_date
