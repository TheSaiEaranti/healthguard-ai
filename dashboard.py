import dash
from dash import html, dcc, Input, Output, State
import requests

app = dash.Dash(__name__)
server = app.server  # for deployment later if needed

app.layout = html.Div([
    html.H1("🧠 HealthGuard AI - Patient Risk Predictor"),

    html.Div([
        dcc.Input(id='age', type='number', placeholder='Age'),
        dcc.Input(id='blood_pressure', type='number', placeholder='Blood Pressure'),
        dcc.Input(id='cholesterol', type='number', placeholder='Cholesterol'),
        dcc.Input(id='glucose_level', type='number', placeholder='Glucose Level'),
        dcc.Input(id='smoking', type='number', placeholder='Smoking (0 or 1)'),
        dcc.Input(id='bmi', type='number', placeholder='BMI'),
    ], style={'marginBottom': '20px'}),

    html.Button('Predict Risk', id='submit-btn', n_clicks=0),

    html.Div(id='prediction-output', style={'marginTop': '20px', 'fontWeight': 'bold'})
])

@app.callback(
    Output('prediction-output', 'children'),
    Input('submit-btn', 'n_clicks'),
    State('age', 'value'),
    State('blood_pressure', 'value'),
    State('cholesterol', 'value'),
    State('glucose_level', 'value'),
    State('smoking', 'value'),
    State('bmi', 'value'),
)
def predict(n_clicks, age, blood_pressure, cholesterol, glucose_level, smoking, bmi):
    if n_clicks == 0:
        return ""

    # Send POST request to Flask API
    payload = {
        "age": age,
        "blood_pressure": blood_pressure,
        "cholesterol": cholesterol,
        "glucose_level": glucose_level,
        "smoking": smoking,
        "bmi": bmi
    }

    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=payload)
        result = response.json()

        if "error" in result:
            return f"❌ Error: {result['error']}"

        return f"⚠️ Risk Score: {result['risk_score']} — High Risk: {result['high_risk']}"

    except Exception as e:
        return f"❌ API error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

