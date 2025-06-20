import streamlit as st
import requests
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64

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

BACKEND_URL = 'https://customersegmentation-2.onrender.com/predict'
REPORT_URL = 'https://customersegmentation-2.onrender.com/generate_report'

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

page = st.sidebar.radio('Navigation', ['Prediction & Insights', 'Report Generation'])

if 'report_data' not in st.session_state:
    st.session_state['report_data'] = None
if 'pie_chart_img' not in st.session_state:
    st.session_state['pie_chart_img'] = None
if 'scatter_img' not in st.session_state:
    st.session_state['scatter_img'] = None

# Load sample data for scatter plot
@st.cache_data
def load_sample_data():
    df = pd.read_csv('data/customers.csv')
    return df

sample_df = load_sample_data()

if page == 'Prediction & Insights':
    st.markdown("""
    <div style='margin-bottom:1em;'><h2>🧾 Customer Input</h2></div>
    """, unsafe_allow_html=True)
    with st.form('input_form'):
        age = st.number_input('Age', min_value=10, max_value=100, value=25)
        income = st.number_input('Annual Income (k$)', min_value=1, max_value=200, value=60)
        score = st.number_input('Spending Score (1–100)', min_value=1, max_value=100, value=80)
        submitted = st.form_submit_button('Predict Segment')

    if submitted:
        payload = {'age': age, 'income': income, 'score': score}
        try:
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
    <div style='margin-bottom:1em;'><h2>📝 Report Generation</h2></div>
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
        <b>Name:</b> Anonymous User  <br>
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