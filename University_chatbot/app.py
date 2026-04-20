from flask import Flask, request, jsonify
import pickle
import json
import re
import random
import os

app = Flask(__name__)

# ==============================
# LOAD MODEL
# ==============================
# FIX 1: Use os.path for robust path resolution regardless of working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "chatbot_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# ==============================
# LOAD RESPONSES (UTF-8 SAFE)
# ==============================
intents_path = os.path.join(BASE_DIR, "intents.json")
with open(intents_path, encoding="utf-8") as f:
    responses = json.load(f)

# ==============================
# CHAT MEMORY
# ==============================
chat_history = []

# ==============================
# TEXT CLEANING FUNCTION
# ==============================
def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text.strip()

# ==============================
# SMART RESPONSE HANDLER
# ==============================
def get_response(intent):
    reply = responses.get(intent, responses["fallback"])

    # FIX 2: Always resolve list to string BEFORE appending extras
    if isinstance(reply, list):
        reply = random.choice(reply)

    # Extra context enhancements
    if intent == "fees":
        reply += "\n💡 You can pay via challan or online banking."
    elif intent == "admissions":
        reply += "\n📌 Apply early to secure your seat."
    elif intent == "greeting":
        reply += "\n😊 How can I help you today?"
    elif intent == "scholarships":
        reply += "\n🎓 Maintain your CGPA to continue receiving benefits."

    return reply

# ==============================
# HOME ROUTE
# ==============================
@app.route('/')
def home():
    # FIX 3: Use os.path so it works regardless of CWD
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(template_path, encoding="utf-8") as f:
        return f.read()

# ==============================
# CHAT API
# ==============================
@app.route("/chat", methods=["POST"])
def chat():
    global chat_history

    try:
        user_input = request.json.get("message", "")

        if not user_input.strip():
            return jsonify({
                "response": "Please type a message.",
                "intent": "empty",
                "confidence": 0.0
            })

        # Clean input
        user_input_clean = clean(user_input)

        # Predict intent
        probs = model.predict_proba([user_input_clean])[0]
        intent = model.classes_[probs.argmax()]
        confidence = max(probs)

        # FIX 4: Fallback also returns a string (random.choice), not raw list
        if confidence < 0.30:
            reply = random.choice(responses["fallback"])
        else:
            reply = get_response(intent)

        # Save chat history
        chat_history.append({
            "user": user_input,
            "bot": reply,
            "intent": intent,
            "confidence": round(float(confidence), 2)
        })

        return jsonify({
            "response": str(reply),
            "intent": str(intent),
            "confidence": round(float(confidence), 2)
        })

    except Exception as e:
        return jsonify({
            "response": "Something went wrong. Please try again.",
            "error": str(e)
        }), 500

# ==============================
# HISTORY ROUTE (Bonus)
# ==============================
@app.route("/history", methods=["GET"])
def history():
    return jsonify(chat_history[-20:])  # Last 20 messages

# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":
    app.run(debug=True)