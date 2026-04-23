import pandas as pd
import numpy as np
from datetime import datetime

def calculate_rfm(df):
    """
    Calculates Recency, Frequency, and Monetary scores for customer segmentation.
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "customer_name",
                "Recency",
                "Frequency",
                "Monetary",
                "R",
                "F",
                "M",
                "RFM_Score",
                "Segment",
            ]
        )

    # Use max date in dataset as 'today' for recency calculation
    today = df['order_date'].max()
    
    rfm = df.groupby('customer_name').agg({
        'order_date': lambda x: (today - x.max()).days,
        'order_id': 'count',
        'sales': 'sum'
    })
    
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    def _score_percentile(series: pd.Series, higher_is_better: bool, n_bins: int = 4) -> pd.Series:
        """
        Robust alternative to qcut for small/low-variance slices.
        Produces integer scores in [1, n_bins] even with many ties.
        """
        s = pd.to_numeric(series, errors="coerce")
        if s.dropna().nunique() <= 1:
            # If everything is identical (or all NaN), put customers in the middle.
            return pd.Series(2, index=series.index, dtype="int64")

        ranked = s.rank(method="average", pct=True)
        if not higher_is_better:
            ranked = (-s).rank(method="average", pct=True)

        scores = np.ceil(ranked * n_bins)
        scores = np.clip(scores, 1, n_bins)
        return pd.Series(scores.astype(int), index=series.index, dtype="int64")

    # Lower recency is better, higher frequency/monetary is better
    rfm["R"] = _score_percentile(rfm["Recency"], higher_is_better=False, n_bins=4)
    rfm["F"] = _score_percentile(rfm["Frequency"], higher_is_better=True, n_bins=4)
    rfm["M"] = _score_percentile(rfm["Monetary"], higher_is_better=True, n_bins=4)
    
    rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis=1)
    
    # Assign Segment Names
    def segment_name(score):
        if score >= 10:
            return 'Champions / Loyal'
        elif score >= 7:
            return 'Potential Loyalists'
        elif score >= 5:
            return 'Promising / At Risk'
        else:
            return 'Lost / Hibernating'
            
    rfm['Segment'] = rfm['RFM_Score'].apply(segment_name)
    
    return rfm.reset_index()

def get_automated_insights(df):
    """
    Generates basic automated insights about the data.
    """
    insights = []
    
    top_region = df.groupby('region')['sales'].sum().idxmax()
    top_cat = df.groupby('category')['sales'].sum().idxmax()
    avg_discount = df['discount'].mean() * 100
    
    insights.append(f"🌟 **{top_region}** is currently your highest performing region in terms of sales volume.")
    insights.append(f"📦 The **{top_cat}** category is the leading driver of revenue across all filtered markets.")
    insights.append(f"📉 Average discount rate is **{avg_discount:.1f}%**. Spikes in discount correlates with volume but shows lower profit margins in some segments.")
    
    return insights
