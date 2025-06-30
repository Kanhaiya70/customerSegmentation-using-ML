import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import time
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from dotenv import load_dotenv
import os

load_dotenv()

# --- Custom CSS for dark theme and modern look ---
st.markdown('''
    <style>
    html, body, [class*="css"]  {
        background-color: #18191A !important;
        color: #E4E6EB !important;
    }
    .main {
        background-color: #18191A !important;
    }
    .stButton>button {
        color: #fff;
        background: linear-gradient(90deg, #4F8BF9 0%, #235390 100%);
        border: none;
        border-radius: 8px;
        padding: 0.5em 2em;
        font-weight: bold;
        margin-top: 1em;
    }
    .stDownloadButton>button {
        color: #fff;
        background: linear-gradient(90deg, #4F8BF9 0%, #235390 100%);
        border: none;
        border-radius: 8px;
        padding: 0.5em 2em;
        font-weight: bold;
        margin-top: 1em;
    }
    .stTextInput>div>div>input, .stNumberInput>div>input {
        background-color: #242526 !important;
        color: #E4E6EB !important;
        border-radius: 6px;
        border: 1px solid #3A3B3C;
    }
    .stForm {
        background: #23272F;
        border-radius: 12px;
        padding: 2em 2em 1em 2em;
        box-shadow: 0 2px 16px 0 #00000033;
        margin-bottom: 2em;
    }
    hr {
        border: 1px solid #3A3B3C;
        margin: 2em 0;
    }
    .report-section {
        background: #23272F;
        border-radius: 12px;
        padding: 2em;
        margin-top: 2em;
        box-shadow: 0 2px 16px 0 #00000033;
    }
    </style>
''', unsafe_allow_html=True)

# BACKEND_URL = 'http://localhost:5000/predict'
# REPORT_URL = 'http://localhost:5000/generate_report'

BACKEND_URL = os.getenv("URL_PREDICT")
REPORT_URL = os.getenv("URL_REPORT")

SEGMENT_DESCRIPTIONS = {
    0: [
        "Low annual income",
        "Careful spending habits",
        "Seeks value deals"
    ],
    1: [
        "Young demographic",
        "Exploring new products",
        "Potential for loyalty growth"
    ],
    2: [
        "High annual income",
        "High spending habits",
        "Target for premium services"
    ],
    3: [
        "Moderate income",
        "Balanced spending",
        "Occasional splurges"
    ],
    4: [
        "Infrequent shoppers",
        "Low engagement",
        "Opportunity for reactivation"
    ]
}

st.markdown("""
<div style='display: flex; align-items: center; gap: 1em;'>
    <img src='https://cdn-icons-png.flaticon.com/512/12072/12072160.png' width='40' style='margin-bottom:0;'>
    <h1 style='color:#4F8BF9; margin-bottom:0;'>Customer Segmentation & Insights Dashboard</h1>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio('Navigation', ['Prediction & Insights', 'Clustering Analysis', 'Report Generation'])

if 'report_data' not in st.session_state:
    st.session_state['report_data'] = None
if 'pie_chart_img' not in st.session_state:
    st.session_state['pie_chart_img'] = None
if 'scatter_img' not in st.session_state:
    st.session_state['scatter_img'] = None

# Load sample data for scatter plot
@st.cache_data
def load_sample_data():
    df = pd.read_csv('data/enhanced_customers.csv')
    return df

sample_df = load_sample_data()

if page == 'Prediction & Insights':
    st.markdown("""
    <div style='margin-bottom:1em;'><h2>🧾 Customer Input</h2></div>
    """, unsafe_allow_html=True)
    with st.form('input_form'):
        name = st.text_input('Name', placeholder= "Enter your name")        # Trial version
        age = st.number_input('Age', min_value=10, max_value=100, value=25)
        income = st.number_input('Annual Income (k$)', min_value=1, max_value=200, value=60)
        score = st.number_input('Spending Score (1–100)', min_value=1, max_value=100, value=80)
        submitted = st.form_submit_button('Predict Segment')

    if submitted:
        st.audio('audio.mp3', format='audio/mp3', start_time=0, autoplay=True)
        st.markdown("""
            <style>
            audio {
                display: none;
            }
            </style>
        """, unsafe_allow_html=True)
        payload = {'name': name, 'age': age, 'income': income, 'score': score}      # Trial version

        with st.spinner('🔍 Ruko Zara, Sabar karo !.....😂'):
            try:
                time.sleep(0.5)
                response = requests.post(BACKEND_URL, json=payload)
                result = response.json()
                st.markdown(f"""
                <div class='report-section'>
                    <h3 style='color:#4F8BF9;'>🎯 Prediction Result</h3>
                    <p style='font-size:1.2em;'><b>Predicted Segment:</b> <span style='color:#FFD600;'>{result['segment']} - {result['label']}</span></p>
                </div>
                """, unsafe_allow_html=True)

                st.write('Probabilities:', result['probabilities'])
                st.markdown("<hr>", unsafe_allow_html=True)
                # Plotly Pie chart for interactive display
                st.markdown("<h4 style='color:#4F8BF9;'>📊 Segment Probability Breakdown</h4>", unsafe_allow_html=True)
                fig_pie = px.pie(
                    names=list(result['probabilities'].keys()),
                    values=list(result['probabilities'].values()),
                    title='Segment Probability Breakdown',
                    color_discrete_sequence=px.colors.sequential.Blues_r
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown("<hr>", unsafe_allow_html=True)
                
                # Matplotlib Pie chart for PDF
                pie_buf = io.BytesIO()
                plt.figure(figsize=(4,4))
                plt.pie(list(result['probabilities'].values()), labels=list(result['probabilities'].keys()), autopct='%1.1f%%')
                plt.title('Segment Probability Breakdown')
                plt.tight_layout()
                plt.savefig(pie_buf, format='png')
                plt.close()
                pie_b64 = base64.b64encode(pie_buf.getvalue()).decode('utf-8')
                st.session_state['pie_chart_img'] = pie_b64

                # Plotly Scatter plot for interactive display
                st.markdown("<h4 style='color:#4F8BF9;'>🔎 Income vs. Spending Score (Clusters)</h4>", unsafe_allow_html=True)
                fig_scatter = px.scatter(
                    sample_df, x='income', y='score', color='segment',
                    title='Income vs. Spending Score (Clusters)',
                    labels={'income': 'Annual Income (k$)', 'score': 'Spending Score'},
                    color_continuous_scale=px.colors.sequential.Blues_r
                )
                fig_scatter.add_scatter(x=[income], y=[score], mode='markers', marker=dict(size=15, color='Gold', symbol='diamond'), name='Current User')
                st.plotly_chart(fig_scatter, use_container_width=True)

                # Matplotlib Scatter plot for PDF
                scatter_buf = io.BytesIO()
                plt.figure(figsize=(5,4))
                for seg in sample_df['segment'].unique():
                    seg_df = sample_df[sample_df['segment'] == seg]
                    plt.scatter(seg_df['income'], seg_df['score'], label=f'Segment {seg}')
                plt.scatter([income], [score], color='Black', marker='*', s=200, label='Current User')
                plt.xlabel('Annual Income (k$)')
                plt.ylabel('Spending Score')
                plt.title('Income vs. Spending Score (Clusters)')
                plt.legend()
                plt.tight_layout()
                plt.savefig(scatter_buf, format='png')
                plt.close()
                scatter_b64 = base64.b64encode(scatter_buf.getvalue()).decode('utf-8')
                st.session_state['scatter_img'] = scatter_b64

                # Save report data to session state
                st.session_state['report_data'] = {
                    'name': name,           # Trial version
                    'age': age,
                    'income': income,
                    'score': score,
                    'segment': result['segment'],
                    'label': result['label'],
                    'probabilities': result['probabilities'],
                    'pie_chart_img': pie_b64,
                    'scatter_img': scatter_b64
                }

            except Exception as e:
                st.error(f'Prediction failed: {e}')


elif page == 'Report Generation':
    st.markdown("""
    <div style='margin-bottom:1em;'>
        <h2>📝 Report Generation</h2>
    </div>
    """, unsafe_allow_html=True)
    report_data = st.session_state.get('report_data')
    if not report_data:
        st.info('No prediction data available. Please run a prediction first.')
    else:
        seg = int(report_data['segment'])
        seg_desc = SEGMENT_DESCRIPTIONS.get(seg, ["No description available."])
        st.markdown(f"""
        <div class='report-section'>
            <h3 style='color:#4F8BF9;'>Customer Segmentation Report</h3>
            <hr>
            <b>Name:</b> {report_data['name']}  <br>    
            <b>Age:</b> {report_data['age']}  <br>
            <b>Income:</b> {report_data['income']}k  <br>
            <b>Spending Score:</b> {report_data['score']}  <br>
            <br>
            <b>Predicted Segment:</b> <span style='color:#FFD600;'>{seg} - {report_data['label']}</span><br>
            <br>
            <b>Segment Characteristics:</b>
            <ul>
                {''.join(f'<li>{desc}</li>' for desc in seg_desc)}
            </ul>
            <br>
            <b>Included Charts:</b>
            <ul>
                <li>Segment Distribution Bar Chart</li>
                <li>Probability Pie Chart</li>
                <li>Scatter Plot</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        # PDF download logic
        if st.button('🖨️ Print Report / Save as PDF'):
            try:
                with st.spinner('Generating PDF...'):
                    pdf_response = requests.post(REPORT_URL, json=report_data)
                    if pdf_response.status_code == 200:
                        st.download_button(
                            label='Download PDF',
                            data=pdf_response.content,
                            file_name='Customer_Segmentation_Report.pdf',
                            mime='application/pdf'
                        )
                    else:
                        st.error('Failed to generate PDF. Please try again.')
            except Exception as e:
                st.error(f'PDF generation failed: {e}') 

elif page == 'Clustering Analysis':
    st.markdown("""
    <div style='margin-bottom:1em;'><h2>📊 Clustering Insights</h2></div>
    """, unsafe_allow_html=True)

    # Load dataset
    df = sample_df  # already cached as `sample_df`

    # Cluster sizes
    st.subheader("📈 Cluster Sizes")
    cluster_sizes = df['segment'].value_counts().sort_index()
    st.bar_chart(cluster_sizes)

    # Centroids
    st.subheader("📍 Cluster Centroids")
    centroids = df.groupby('segment')[['age', 'income', 'score']].mean().round(2)
    st.dataframe(centroids)

    st.subheader("🌲 Hierarchical Clustering (Dendrogram)")

    # Prepare features for clustering
    X = df[['age', 'income', 'score']].values

    # Plot dendrogram
    fig_dendro = plt.figure(figsize=(8, 5))
    dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    st.pyplot(fig_dendro)

    # Agglomerative Clustering
    hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X)

    # Add cluster labels to dataframe
    df['agglo_cluster'] = y_hc

    # Show cluster sizes
    st.subheader("🧮 Agglomerative Cluster Sizes")
    agglo_sizes = df['agglo_cluster'].value_counts().sort_index()
    st.bar_chart(agglo_sizes)

    # Show cluster centroids (mean of each cluster)
    st.subheader("📍 Agglomerative Cluster Centroids")
    agglo_centroids = df.groupby('agglo_cluster')[['age', 'income', 'score']].mean().round(2)
    st.dataframe(agglo_centroids)

    # Visualize clusters (2D)
    st.subheader("🗺️ Income vs. Score (Agglomerative Clusters)")
    fig_scatter = px.scatter(
        df, x='income', y='score', color='agglo_cluster',
        title='Agglomerative Clusters (Income vs Score)',
        labels={'income': 'Annual Income (k$)', 'score': 'Spending Score'}
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)


    # Scatter plot with centroids
    

    st.subheader("📡 Visualizing Clusters and Centroids")

    # Create base scatter WITHOUT legend by converting px.scatter to go traces
    fig = go.Figure()

    # Manually add each point in df as scatter trace without legend
    fig.add_trace(go.Scatter(
        x=df['income'], y=df['score'],
        mode='markers',
        marker=dict(
            color=df['agglo_cluster'],
            colorscale='Viridis',
            size=8,
            colorbar=dict(title='Cluster'),  # remove this line to also hide colorbar
            showscale=False  # ← disables color bar
        ),
        showlegend=False  # ← disables cluster legend
    ))

    # Plot centroids WITH legend
    for seg, row in centroids.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['income']], y=[row['score']],
            mode='markers+text',
            marker=dict(size=16, color='red', symbol='x'),
            text=[f'Centroid {seg}'],
            name=f'Centroid {seg}',
            showlegend=True  # ← this will appear in legend
        ))

    fig.update_layout(
        title='Clusters with Centroids',
        xaxis_title='Annual Income (k$)',
        yaxis_title='Spending Score',
    )

    st.plotly_chart(fig, use_container_width=True) 