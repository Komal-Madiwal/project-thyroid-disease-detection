import os
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# Use __file__ to get the path of the current script
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))


from src.thyroiddiseasedetect.pipelines.prediction_pipeline import CustomData,PredictPipeline
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    else:
        data = CustomData(
            sex=request.form.get('sex'),
            on_thyroxine=request.form.get('on thyroxine'),
            pregnant=request.form.get('pregnant'),
            query_hypothyroid=request.form.get('query hypothyroid'),
            psych=request.form.get('psych'),
            TSH_measured=request.form.get('TSH measured'),
            TSH=float(request.form.get('TSH')),
            T3=float(request.form.get('T3')),
            TT4=float(request.form.get('TT4')),
            FTI=float(request.form.get('FTI')),
        )

        final_data = data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_data)
        result = round(pred[0], 2)

        return render_template("result.html", final_result=result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

