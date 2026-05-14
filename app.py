from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import os
import json
import random
import subprocess
import logging
import json
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sentence_transformers import SentenceTransformer, util
from models.cnn_model import CNNClassifier

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('app.log')
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)
checkpoint = torch.load("cnn_classifier.pth", map_location=device)

# Fix checkpoint keys
state_dict = checkpoint.get('model') or checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
model_keys = list(model.state_dict().keys())
checkpoint_keys = list(state_dict.keys())

if all(k.startswith('model.') for k in model_keys) and not all(k.startswith('model.') for k in checkpoint_keys):
    fixed_state_dict = {'model.' + k: v for k, v in state_dict.items()}
elif not all(k.startswith('model.') for k in model_keys) and all(k.startswith('model.') for k in checkpoint_keys):
    fixed_state_dict = {k[len('model.'):]: v for k, v in state_dict.items()}
else:
    fixed_state_dict = state_dict

model.load_state_dict(fixed_state_dict)
model.eval()

# Transform definition
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load chatbot intents
with open("intents.json") as file:
    data = json.load(file)

# Load sentence transformer model
nlp_model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute all pattern embeddings
intent_embeddings = []
for intent in data["intents"]:
    pattern_embeddings = [
        nlp_model.encode(pattern, convert_to_tensor=True)
        for pattern in intent["patterns"]
    ]
    intent_embeddings.append({
        "tag": intent["tag"],
        "responses": intent["responses"],
        "embeddings": pattern_embeddings
    })

# Rule-based fallback for vague inputs
def rule_based_intent(user_input):
    text = user_input.lower()
    if "type" in text:
        return "types_thyroid_cancer"
    elif "reduce" in text or "prevent" in text:
        return "prevention"
    elif "explain" in text or "define" in text or "what is" in text:
        return "definition_thyroid_cancer"
    elif "symptom" in text or "sign" in text:
        return "symptoms"
    elif "treatment" in text or "therapy" in text or "cure" in text:
        return "treatment_options"
    elif "cause" in text or "trigger" in text or "reason" in text:
        return "risk_factors"
    return None

# Enhanced chatbot response function
def get_response(user_input):
    user_input = user_input.strip().lower()
    user_embedding = nlp_model.encode(user_input, convert_to_tensor=True)

    best_score = 0.0
    best_response = None

    for intent in intent_embeddings:
        max_similarity = max([
            util.cos_sim(user_embedding, emb).item()
            for emb in intent["embeddings"]
        ])
        if max_similarity > best_score:
            best_score = max_similarity
            best_response = random.choice(intent["responses"])

    if best_score > 0.45:
        return best_response

    # Fallback using rule-based logic
    fallback_tag = rule_based_intent(user_input)
    if fallback_tag:
        for intent in data["intents"]:
            if intent["tag"] == fallback_tag:
                return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."

# Routes



@app.route('/')
def home():
    return render_template('index.html')
@app.route('/about.html')
def about_html():
    return render_template('about.html')

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    true_label_str = request.form.get("true_label")
    true_label_map = {"Benign": 0, "Malignant": 1}

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = Image.open(filepath).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            prob = torch.sigmoid(output).item()

        pred_label = 1 if prob > 0.6 else 0
        pred_str = "Malignant" if pred_label == 1 else "Benign"
        confidence = prob if pred_label == 1 else 1 - prob

        response = {
            "prediction": pred_str,
            "accuracy": round(confidence * 100, 2),
            "filename": filename
        }

        if true_label_str in true_label_map:
            true_label = true_label_map[true_label_str]
            y_true = [true_label]
            y_pred = [pred_label]

            response.update({
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "accuracy_full": accuracy_score(y_true, y_pred)
            })

        logger.info(f'Prediction made for file {filename}')
        return jsonify(response)

    except Exception as e:
        logger.error(f'Error making prediction: {str(e)}')
        return jsonify({"error": str(e)}), 500

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/train", methods=["POST"])
def train_model():
    try:
        subprocess.run(["python", "classifier/train_classifier.py"])
        logger.info('Training complete!')
        return jsonify({"message": "Training complete!"})
    except Exception as e:
        logger.error(f'Error training model: {str(e)}')
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate", methods=["GET"])
def evaluate():
    try:
        test_dataset = ImageFolder("data/cls/val", transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        y_true, y_pred = [], []

        with torch.no_grad():
            for img, label in test_loader:
                img = img.to(device)
                output = model(img)
                prob = torch.sigmoid(output).item()
                pred = 1 if prob > 0.6 else 0

                y_true.append(label.item())
                y_pred.append(pred)

        logger.info('Evaluation complete!')
        return jsonify({
            "precision": round(precision_score(y_true, y_pred), 4),
            "recall": round(recall_score(y_true, y_pred), 4),
            "f1_score": round(f1_score(y_true, y_pred), 4)
        })

    except Exception as e:
        logger.error(f'Error evaluating model: {str(e)}')
        return jsonify({"error": str(e)}), 500

@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        user_input = request.json.get("message", "")
        response = get_response(user_input)
        logger.info(f'Chatbot response: {response}')
        return jsonify({"reply": response})
    except Exception as e:
        logger.error(f'Error getting chatbot response: {str(e)}')
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
