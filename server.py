from flask import Flask, request, jsonify
from flask_cors import CORS

from symptom import predict_disease

app = Flask(__name__)
CORS(app)


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    symptoms = data.get('symptoms')

    try:
        # 🔥 convert input string → list
        symptoms_list = [s.strip().lower() for s in symptoms.replace(',', ' ').split()]

        # 🔥 ML prediction
        result = predict_disease(symptoms_list)

        # ✅ SIMPLE RESPONSE
        return jsonify({
            "result": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("🔥 Backend running (ML ONLY)")
    app.run(host="127.0.0.1", port=5000, debug=True)