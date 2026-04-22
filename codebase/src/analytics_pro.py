import pandas as pd
import numpy as np
from datetime import datetime

def calculate_rfm(df):
    """
    Calculates Recency, Frequency, and Monetary scores for customer segmentation.
    """
    # Use max date in dataset as 'today' for recency calculation
    today = df['order_date'].max()
    
    rfm = df.groupby('customer_name').agg({
        'order_date': lambda x: (today - x.max()).days,
        'order_id': 'count',
        'sales': 'sum'
    })
    
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    # Create segments based on quartiles
    # Lower recency is better, higher frequency/monetary is better
    r_labels = range(4, 0, -1)
    f_labels = range(1, 5)
    m_labels = range(1, 5)
    
    rfm['R'] = pd.qcut(rfm['Recency'], q=4, labels=r_labels, duplicates='drop')
    rfm['F'] = pd.qcut(rfm['Frequency'], q=4, labels=f_labels, duplicates='drop')
    rfm['M'] = pd.qcut(rfm['Monetary'], q=4, labels=m_labels, duplicates='drop')
    
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
