import os
import sys

# Avoid UnicodeEncodeError on Windows cp1252 terminals.
reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
if callable(reconfigure_stdout):
    reconfigure_stdout(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(__file__)

disease_path = os.path.join(BASE_DIR, "DiseaseAndSymptoms.csv")
precaution_path = os.path.join(BASE_DIR, "Disease precaution.csv")

missing_files = [
    path for path in (disease_path, precaution_path) if not os.path.exists(path)
]
if missing_files:
    print("Missing required data file(s):")
    for f in missing_files:
        print(f" - {f}")
    print(
        "\nPlace both CSV files in the same folder as symptom.py and run again."
    )
    sys.exit(1)

# =========================================
# Prediction using Symptom Matching
# =========================================

def predict_disease(user_symptoms):
    # Backwards-compatible wrapper (CLI + any existing callers)
    from symptom_core import load_model, predict_disease as core_predict

    model = load_model(BASE_DIR)
    return core_predict(model, user_symptoms)


# =========================================
# Interactive CLI Version
# =========================================

if __name__ == "__main__":
    cli_symptoms = sys.argv[1:]

    # Support non-interactive runs (for example, Code Runner output panel)
    # by accepting symptoms from command-line args.
    if cli_symptoms:
        user_symptoms = [s.strip().lower() for s in " ".join(cli_symptoms).split(",")]
        user_symptoms = [s for s in user_symptoms if s]
        result = predict_disease(user_symptoms)

        print("\nPredicted Disease:", result["Predicted Disease"])
        if result["Predicted Disease"] != "Unknown":
            confidence = round((result["Match Score"] / len(user_symptoms)) * 100, 2)
            print("Match Score:", result["Match Score"])
            print("Confidence:", f"{confidence}%")

        print("Precautions:")
        for p in result["Precautions"]:
            print("-", p)
        sys.exit(0)

    print("\nWelcome to the Symptom Checker")
    print("Enter symptoms separated by commas (example: fever, cough, headache)")
    print("Type 'exit' to quit.\n")

    while True:

        try:
            user_input = input("Enter your symptoms: ")
        except EOFError:
            print(
                "\nInput is not available in this run mode.\n"
                "Run in a terminal, or pass symptoms as an argument:\n"
                "python symptom.py \"fever, cough, headache\""
            )
            break

        if user_input.lower() == "exit":
            print("\nStay healthy. Goodbye!")
            break

        if not user_input.strip():
            print("Please enter at least one symptom.\n")
            continue

        # Convert string to list
        user_symptoms = [s.strip().lower() for s in user_input.split(",")]

        result = predict_disease(user_symptoms)

        print("\nPredicted Disease:", result["Predicted Disease"])

        if result["Predicted Disease"] != "Unknown":
            confidence = round(
                (result["Match Score"] / len(user_symptoms)) * 100, 2
            )
            print("Match Score:", result["Match Score"])
            print("Confidence:", f"{confidence}%")

        print("Precautions:")
        for p in result["Precautions"]:
            print("-", p)

        print("\n" + "-"*40)
