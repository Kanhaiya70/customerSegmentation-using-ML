from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import pdfkit
import tempfile
import os
from io import BytesIO
from dotenv import load_dotenv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

load_dotenv()

app = Flask(__name__)

# Segment labels
SEGMENT_LABELS = {
    0: "Budget-Conscious",
    1: "Young Explorers",
    2: "High Income, High Spend",
    3: "Moderate Spenders",
    4: "Occasional Shoppers"
}

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

# Load model (placeholder, replace with actual model)
try:
    model = joblib.load('segment_classifier.pkl')
    print("Model loaded successfully")
except Exception:
    import traceback
    print("Model loading failed:")
    traceback.print_exc()
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    age = data.get('age')
    income = data.get('income')
    score = data.get('score')
    # Input validation
    if age is None or income is None or score is None:
        return jsonify({'error': 'Missing input'}), 400
    # Dummy prediction if model not loaded
    if model is None:
        segment = 2
        probabilities = {str(i): 0.0 for i in range(5)}
        probabilities['2'] = 1.0
    else:
        X = np.array([[age, income, score]])
        segment = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        probabilities = {str(i): float(proba[i]) for i in range(len(proba))}
    label = SEGMENT_LABELS.get(segment, "Unknown")
    return jsonify({
        'segment': segment,
        'label': label,
        'probabilities': probabilities
    })

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.get_json()
    name = data.get('name')     #Trial version
    age = data.get('age')
    income = data.get('income')
    score = data.get('score')
    segment = int(data.get('segment', 0))
    label = data.get('label', '')
    probabilities = data.get('probabilities', {})
    seg_desc = SEGMENT_DESCRIPTIONS.get(segment, ["No description available."])
    pie_chart_img = data.get('pie_chart_img', None)
    scatter_img = data.get('scatter_img', None)
    # HTML template for the report
    html = f'''
    <html>
        <head>
            <meta charset="utf-8">
            <title>Customer Segmentation Report</title>
        </head>
        <body>
            <h1>Customer Segmentation Report</h1>
            <hr>
            <p>
                <b>Name:</b> {name}<br>      
                <b>Age:</b> {age}<br>
                <b>Income:</b> {income}k$<br>
                <b>Spending Score:</b> {score}
            </p>
            <p>
                <b>Predicted Segment:</b> {segment} - {label}
            </p>
            <p>
                <b>Segment Characteristics:</b>
                <ul>
                    {''.join(f'<li>{desc}</li>' for desc in seg_desc)}
                </ul>
            </p>
            <p>
                <b>Probabilities:</b>
                <ul>
                    {''.join(f'<li>Segment {k}: {v:.2f}</li>' for k, v in probabilities.items())}
                </ul>
            </p>
            <p>
                <b>Included Charts:</b>
                <ul>
                    <li>Probability Pie Chart</li>
                    <li>Scatter Plot</li>
                </ul>
            </p>
    '''
    if pie_chart_img:
        html += f'<h3>Probability Pie Chart</h3><img src="data:image/png;base64,{pie_chart_img}" width="400"/></br></br></br></br>'
    if scatter_img:
        html += f'<div style= "page-break-before: always;"></div>   <h3 style= "margin-bottom: 30px;">Scatter Plot (Income vs. Spending Score)</h3><img src="data:image/png;base64,{scatter_img}" width="400"/>'
    html += '</body></html>'

    # Generate PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmpfile:
        wkhtmltopdf_url = os.getenv("PDF_TOOLKIT")
        # config = pdfkit.configuration(wkhtmltopdf=r'C:\\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')
        config = pdfkit.configuration(wkhtmltopdf= wkhtmltopdf_url)
        pdfkit.from_string(html, tmpfile.name, configuration=config)
        tmpfile.flush()
        tmpfile.seek(0)
        pdf_bytes = tmpfile.read()
    os.unlink(tmpfile.name)
    return send_file(
        BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name='Customer_Segmentation_Report.pdf'
    )

@app.route('/retrain', methods= ['POST'])
def retrain():
    # if 'file' not in request.files:
    #     return jsonify({'error': 'No file uploaded'}), 400

    # file = request.files['file']
    # new_df = pd.read_csv(file)

    # def assign_segment(row):
    #     age, income, score = row['age'], row['income'], row['score']

    #     if income < 40 and score < 40:
    #         return 0  # Budget-Conscious
    #     elif age <= 30 and 40 <= score <= 70:
    #         return 1  # Young Explorers
    #     elif income >= 90 and score >= 70:
    #         return 2  # High Income, High Spend
    #     elif 40 <= income <= 80 and 40 <= score <= 70:
    #         return 3  # Moderate Spenders
    #     elif (income < 40 and score >= 70) or (score > 90):
    #         return 4  # Occasional Shoppers
    #     else:
    #         return 3  # Default to moderate
        
    # new_df['segment'] = new_df.apply(assign_segment, axis= 1)
    # import os

    # # Get the current backend folder path
    # backend_dir = os.path.dirname(__file__)

    # # Save to backend/data/enhanced_customers.csv
    # backend_data_path = os.path.join(backend_dir, 'data')

    # if os.path.exists(backend_data_path):
    #     existing_df = pd.read_csv(backend_data_path)
    #     combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates()
    # else:
    #     combined_df = new_df

    # os.makedirs(backend_data_path, exist_ok=True)
    # backend_data_path = os.path.join(backend_data_path, 'enhanced_customers.csv')
    # combined_df.to_csv(backend_data_path, index=False)

    # # Save to frontend/files/enhanced_customers.csv
    # frontend_data_dir = os.path.abspath(os.path.join(backend_dir, '..', 'frontend', 'data'))
    # os.makedirs(frontend_data_dir, exist_ok=True)
    # frontend_data_path = os.path.join(frontend_data_dir, 'enhanced_customers.csv')
    # combined_df.to_csv(os.path.join(frontend_data_path, 'enhanced_customers.csv'), index=False)

    # # os.makedirs('data', exist_ok=True)
    # # df.to_csv('data/enhanced_customers.csv', index=False)

    # X = new_df[['age', 'income', 'score']]
    # Y = new_df['segment']
    # clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # clf.fit(X, Y)

    # model_path = os.path.abspath(os.path.join(backend_dir, '..', 'segment_classifier.pkl'))
    # joblib.dump(clf, model_path)

    # return jsonify({'message': 'Model retrained successfully'})


    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    new_df = pd.read_csv(file)

    # Segment assignment logic
    def assign_segment(row):
        age, income, score = row['age'], row['income'], row['score']

        if income < 40 and score < 40:
            return 0  # Budget-Conscious
        elif age <= 30 and 40 <= score <= 70:
            return 1  # Young Explorers
        elif income >= 90 and score >= 70:
            return 2  # High Income, High Spend
        elif 40 <= income <= 80 and 40 <= score <= 70:
            return 3  # Moderate Spenders
        elif (income < 40 and score >= 70) or (score > 90):
            return 4  # Occasional Shoppers
        else:
            return 3  # Default to moderate

    new_df['segment'] = new_df.apply(assign_segment, axis=1)

    # Set up backend paths
    backend_dir = os.path.dirname(__file__)
    backend_data_dir = os.path.join(backend_dir, 'data')
    os.makedirs(backend_data_dir, exist_ok=True)
    backend_csv_path = os.path.join(backend_data_dir, 'enhanced_customers.csv')

    # Combine with existing data if available
    if os.path.exists(backend_csv_path):
        existing_df = pd.read_csv(backend_csv_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates()
    else:
        combined_df = new_df

    # Save combined data to backend/data
    combined_df.to_csv(backend_csv_path, index=False)

    # Also save to frontend/data
    frontend_data_dir = os.path.abspath(os.path.join(backend_dir, '..', 'frontend', 'data'))
    os.makedirs(frontend_data_dir, exist_ok=True)
    frontend_csv_path = os.path.join(frontend_data_dir, 'enhanced_customers.csv')
    combined_df.to_csv(frontend_csv_path, index=False)

    # Train model
    X = combined_df[['age', 'income', 'score']]
    y = combined_df['segment']
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # Save model to project root
    model_path = os.path.abspath(os.path.join(backend_dir, '..', 'segment_classifier.pkl'))
    joblib.dump(clf, model_path)

    return jsonify({'message': 'Model retrained successfully with appended data'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)