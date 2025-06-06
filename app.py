from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, template_folder='templates')

# Use relative path so it works locally or on Render
model_path = 'hiv_model_joblib.pkl'
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('hiv.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        Marital_Status = int(request.form['Marital_Status'])
        Education_Level = int(request.form['Education_Level'])
        Perception_Category = float(request.form['Perception_Category'])
        Years_on_Treatment = float(request.form['Years_on_Treatment'])
        Overall_Comfort_Level = float(request.form['Overall_Comfort_Level'])
        Monthly_Income = float(request.form['Monthly_Income'])
        Household_Size = int(request.form['Household_Size'])
        Treatment_Regimen = int(request.form['Treatment_Regimen'])
        Care_Timeliness = int(request.form['Care_Timeliness'])
        Comfort_StaffInteraction = int(request.form['Comfort_StaffInteraction'])

        # Prepare data for model
        features = np.array([[Marital_Status, Education_Level, Perception_Category, Years_on_Treatment,
                              Overall_Comfort_Level, Monthly_Income, Household_Size,
                              Treatment_Regimen, Care_Timeliness, Comfort_StaffInteraction]])

        # Predict
        prediction = model.predict(features)[0]
        predicted_score = round(prediction, 2)

        # Interpretation comment
        if predicted_score == 1:
            experience_comment = "Positive experience."
        elif predicted_score == 2:
            experience_comment = "Negative experience."
        else:
            experience_comment = "Neutral experience."

        return render_template('hiv.html',
                               prediction_text=f"Predicted Experience Score: {predicted_score} - {experience_comment}")

    except Exception as e:
        return render_template('hiv.html', error=f"Error processing input: {str(e)}")

# ✅ Correct main block
if __name__ == '__main__':
 print("✅ Starting Flask server... Visit http://127.0.0.1:5000 in your browser.", flush=True)
 app.run(debug=True, host='127.0.0.1', port=5000)