from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import requests
import base64
import time

app = Flask(__name__, template_folder='templates')

# Load pre-trained MobileNet model
model = MobileNet(weights='imagenet')

GEMINI_API_KEY = "AIzaSyDv-fom0vvzI8iqG_E4bChCuZtpgyl5aHM"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

last_pet_interact_time = 0
PET_INTERACT_COOLDOWN = 10  # seconds

# In-memory chat history store
chat_history = []

@app.route('/')
def index():
    return render_template('index_chatgpt.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        img_bytes = file.read()
        img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        results = decode_predictions(preds, top=3)[0]
        predictions = [{'label': label, 'description': desc, 'probability': float(prob)} for (label, desc, prob) in results]
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/gemini-predict', methods=['POST'])
def gemini_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        img_bytes = file.read()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        payload = {
            "contents": [{
                "parts": [{"text": f"Recognize this image: data:image/jpeg;base64,{img_base64}"}]
            }]
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
        if response.status_code != 200:
            return jsonify({'error': f'Gemini API error: {response.status_code}'}), 500
        data = response.json()
        content = data.get('candidates', [{}])[0].get('content', '')
        return jsonify({'predictions': [{'description': content, 'probability': 1.0}]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/gemini-chat', methods=['POST'])
def gemini_chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    message = data['message']
    try:
        payload = {
            "contents": [{
                "parts": [{"text": message}]
            }]
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
        if response.status_code != 200:
            return jsonify({'error': f'Gemini API error: {response.status_code}'}), 500
        resp_data = response.json()
        # Extract text from parts inside content
        content_obj = resp_data.get('candidates', [{}])[0].get('content', {})
        parts = content_obj.get('parts', [{}])
        content_text = parts[0].get('text', '') if parts else ''
        # Validate content_text length
        if not content_text or len(content_text.strip()) < 5:
            content_text = "Xin lỗi, em không hiểu ý anh lắm. 😅 Anh có thể nói rõ hơn không?"
        return jsonify({'response': content_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/pet-interact', methods=['POST'])
def pet_interact():
    global last_pet_interact_time
    current_time = time.time()
    if current_time - last_pet_interact_time < PET_INTERACT_COOLDOWN:
        return jsonify({'response': "Xin lỗi, bạn gái đang nghỉ ngơi. Vui lòng thử lại sau vài giây nhé!"})
    last_pet_interact_time = current_time

    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    message = data['message']
    personality = data.get('personality', '')

    # Save user message to chat history
    chat_history.append({'sender': 'user', 'message': message})

    try:
        # Prepare conversation history for prompt (last 6 messages)
        history_messages = chat_history[-6:]
        conversation = ""
        for msg in history_messages:
            speaker = "Bạn" if msg['sender'] == 'user' else "Cô gái"
            conversation += f"{speaker}: {msg['message']}\n"

        prompt = f"Bạn là một cô gái thân thiện với các đặc điểm sau: {personality}. Dưới đây là cuộc trò chuyện trước đó:\n{conversation}Hãy trả lời tin nhắn này một cách dễ thương và vui nhộn bằng tiếng Việt:\n{message}"
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
        if response.status_code != 200:
            return jsonify({'error': f'Gemini API error: {response.status_code}'}), 500
        resp_data = response.json()
        content = resp_data.get('candidates', [{}])[0].get('content', '')

        # Save AI response to chat history
        chat_history.append({'sender': 'girl', 'message': content})

        return jsonify({'response': content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat-history', methods=['GET'])
def get_chat_history():
    return jsonify({'history': chat_history})

@app.route('/clear-chat', methods=['POST'])
def clear_chat():
    global chat_history
    chat_history = []
    return jsonify({'message': 'Chat history cleared'})

if __name__ == '__main__':
    app.run(debug=True)
