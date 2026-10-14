
import os
import joblib
import pandas as pd
import gradio as gr

# Load the trained model and supporting files
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("breast_cancer_scaler.pkl")
features = joblib.load("breast_cancer_features.pkl")


# Prediction function
def predict_cancer(*values):
    input_data = pd.DataFrame(
        [values],
        columns=features
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        result = "Model prediction: Malignant"
    else:
        result = "Model prediction: Benign"

    return (
        f"{result}\n"
        f"Estimated malignant probability: {probability:.2%}\n\n"
        "Educational use only — not a medical diagnosis."
    )


# Create input fields
inputs = [
    gr.Number(label=name, value=0)
    for name in features
]


# Create the web interface
app = gr.Interface(
    fn=predict_cancer,
    inputs=inputs,
    outputs=gr.Textbox(label="Prediction Result"),
    title="Breast Cancer Prediction System",
    description="Enter the numeric feature values to get a model prediction."
)


# Start the application on Render
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
