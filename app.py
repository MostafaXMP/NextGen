"""Customer Segments & 90-Day Spend: Streamlit app for marketing managers."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

BASE = Path(__file__).parent

st.set_page_config(page_title="Customer Segments & 90-Day Spend", page_icon="🛍️", layout="wide")

# ----------------------------------------------------------------------------
# EDIT HERE: names and marketing actions.
# Segments are sorted from most to least valuable (by median past spend), and
# the playbook below is applied in that order, so it survives a re-training
# even if the cluster ids change. Keep as many entries as there are clusters.
# ----------------------------------------------------------------------------
PLAYBOOK = [
    {"name": "VIP Champions",
     "profile": "Order very often, bought very recently, highest spend.",
     "action": "Reward them: exclusive loyalty perks, early access to new products, a personal thank-you.",
     "goal": "Keep them happy and buying."},
    {"name": "Loyal but Cooling Down",
     "profile": "Solid repeat buyers who have not ordered for a couple of months.",
     "action": "Win them back: a 'We miss you' email with a limited-time discount before they drift away.",
     "goal": "Reactivate them before they leave for good."},
    {"name": "New / Low-Frequency Buyers",
     "profile": "Bought recently but only once or twice so far, small baskets.",
     "action": "Nurture them: onboarding emails, product tips and a small incentive for the next order.",
     "goal": "Turn one-time buyers into repeat customers."},
    {"name": "Lost / Inactive",
     "profile": "Bought once, a long time ago, low spend.",
     "action": "Cut the cost: move them to a low-cost automated email list and exclude them from paid ads.",
     "goal": "Do not waste marketing budget on people unlikely to buy."},
]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
@st.cache_data
def load_customers():
    df = pd.read_csv(BASE / "data" / "customer_forecast.csv", index_col="Customer ID")
    info = json.loads((BASE / "data" / "model_info.json").read_text())
    return df, info


@st.cache_resource
def load_models():
    seg = joblib.load(BASE / "models" / "segmentation_pipeline.joblib")
    spend = joblib.load(BASE / "models" / "spend_regression_pipeline.joblib")
    return seg, spend


customers, info = load_customers()
seg_model, spend_model = load_models()

# if info.get("sklearn_version") and info["sklearn_version"] != sklearn.__version__:
#     st.warning(f"The models were saved with scikit-learn {info['sklearn_version']} but this environment has "
#                f"{sklearn.__version__}. Predictions in the 'Try a new customer' tab may be wrong or fail. "
#                f"Fix: pip install scikit-learn=={info['sklearn_version']}")

# map raw cluster id -> playbook entry (most valuable first)
order = customers.groupby("cluster")["Monetary"].median().sort_values(ascending=False).index.tolist()
SEG = {}
for rank, cid in enumerate(order):
    entry = PLAYBOOK[rank] if rank < len(PLAYBOOK) else {
        "name": f"Segment {cid}", "profile": "", "action": "Define an action for this segment.", "goal": ""}
    SEG[cid] = entry
customers["segment"] = customers["cluster"].map(lambda c: SEG[c]["name"])

money = lambda x: f"£{x:,.0f}"

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🛍️ Customer Segments & 90-Day Spend")
st.caption(f"Who to reward, who to win back, and where the next 3 months of revenue will come from. "
           f"Based on orders up to {info['snapshot_date']}.")

tab_dash, tab_lookup, tab_new = st.tabs(["📊 Segment dashboard", "🔎 Customer lookup", "🧪 Try a new customer"])

# ----------------------------------------------------------------------------
# Tab 1: dashboard
# ----------------------------------------------------------------------------
with tab_dash:
    g = customers.groupby("segment")
    prof = g.agg(customers=("Recency", "size"),
                 recency=("Recency", "median"),
                 orders=("Frequency", "median"),
                 spend=("Monetary", "median"),
                 past_revenue=("Monetary", "sum"),
                 pred_total=("predicted_spend_90d", "sum"),
                 pred_avg=("predicted_spend_90d", "mean"))
    prof["pct_customers"] = prof["customers"] / prof["customers"].sum() * 100
    prof["pct_revenue"] = prof["past_revenue"] / prof["past_revenue"].sum() * 100
    prof["pct_pred"] = prof["pred_total"] / prof["pred_total"].sum() * 100
    names_in_order = [SEG[c]["name"] for c in order]
    prof = prof.loc[names_in_order]

    k1, k2, k3 = st.columns(3)
    k1.metric("Customers", f"{len(customers):,}")
    k2.metric("Past revenue (12 months)", money(customers["Monetary"].sum()))
    k3.metric("Predicted revenue, next 90 days", money(customers["predicted_spend_90d"].sum()))

    st.subheader("Segment profiles")
    show = pd.DataFrame({
        "Segment": prof.index,
        "Customers": prof["customers"].map("{:,}".format),
        "% of customers": prof["pct_customers"].map("{:.0f}%".format),
        "Last order (median days ago)": prof["recency"].map("{:.0f}".format),
        "Orders (median)": prof["orders"].map("{:.0f}".format),
        "Spend (median)": prof["spend"].map(money),
        "% of past revenue": prof["pct_revenue"].map("{:.0f}%".format),
        "Predicted next 90 days": prof["pred_total"].map(money),
        "% predicted": prof["pct_pred"].map("{:.0f}%".format),
    })
    st.dataframe(show, hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Customers per segment**")
        st.bar_chart(prof["customers"].rename("Customers").reset_index(), x="segment", y="Customers", sort=False)
    with c2:
        st.markdown("**Predicted revenue next 90 days (£) = revenue at stake**")
        st.bar_chart(prof["pred_total"].rename("Predicted £").reset_index(), x="segment", y="Predicted £", sort=False)

    st.subheader("Recommended action per segment")
    for c in order:
        s = SEG[c]
        with st.expander(f"{s['name']}  ·  {prof.loc[s['name'], 'customers']:,} customers  ·  "
                         f"{money(prof.loc[s['name'], 'pred_total'])} predicted"):
            st.markdown(f"**Who they are:** {s['profile']}")
            st.markdown(f"**Action:** {s['action']}")
            st.markdown(f"**Goal:** {s['goal']}")

# ----------------------------------------------------------------------------
# Tab 2: customer lookup
# ----------------------------------------------------------------------------
with tab_lookup:
    st.subheader("Search a customer")
    cid_text = st.text_input("Customer ID", placeholder="e.g. 12347")
    if cid_text:
        try:
            cid = int(cid_text.strip())
        except ValueError:
            st.error("Customer IDs are numbers, for example 12347.")
            st.stop()
        if cid not in customers.index:
            st.warning(f"Customer {cid} was not found. Only customers with a valid ID and at least one "
                       f"purchase in the data can be looked up.")
        else:
            row = customers.loc[cid]
            s = SEG[row["cluster"]]
            seg_avg = customers.loc[customers["cluster"] == row["cluster"], "predicted_spend_90d"].mean()
            st.markdown(f"### Customer {cid} · {s['name']}")
            a, b, c, d = st.columns(4)
            a.metric("Predicted spend, next 90 days", money(row["predicted_spend_90d"]),
                     delta=f"{money(row['predicted_spend_90d'] - seg_avg)} vs segment average", delta_color="off")
            b.metric("Past spend (12 months)", money(row["Monetary"]))
            c.metric("Last order", f"{int(row['Recency'])} days ago")
            d.metric("Number of orders", f"{int(row['Frequency'])}")
            st.success(f"**Recommended action:** {s['action']}")
            st.caption(f"Goal: {s['goal']}")

# ----------------------------------------------------------------------------
# Tab 3: what-if for a customer who is not in the table
# ----------------------------------------------------------------------------
with tab_new:
    st.subheader("Try a new customer")
    st.write("Type a customer's behaviour and the saved models will give the segment and expected spend.")
    c1, c2, c3 = st.columns(3)
    r = c1.number_input("Days since last order", min_value=0, max_value=730, value=30)
    f = c2.number_input("Number of orders", min_value=1, max_value=500, value=3)
    m = c3.number_input("Total spend so far (£)", min_value=1.0, max_value=1_000_000.0, value=800.0, step=50.0)
    if st.button("Predict", type="primary"):
        x = pd.DataFrame({"Recency": [r], "Frequency": [f], "Monetary": [m]})
        s = SEG[int(seg_model.predict(x)[0])]
        pred = float(np.clip(spend_model.predict(x)[0], 0, None))
        st.markdown(f"### {s['name']}")
        st.metric("Predicted spend, next 90 days", money(pred))
        st.success(f"**Recommended action:** {s['action']}")

st.divider()
st.caption("Forecasts are best used to rank customers and compare segments, not as an exact budget. "
           "They are built from one year of data and may be too high outside the Christmas peak.")
