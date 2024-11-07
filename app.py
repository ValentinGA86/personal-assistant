from flask import Flask, request, render_template
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json

app = Flask(__name__)
CORS(app)

# Bot setup

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def handle_prompt():   
# Handle incoming data
    
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']
    if input_text.lower() == "reset":
        conversation_history.clear()
        return "Conversation history has been reset."


# Create conversation history string

    conversation_history.append(input_text)

# Tokenization of user prompt and chat history
    
    inputs = tokenizer(" ".join(conversation_history), return_tensors="pt")

# Generate output from the model

    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Update conversation history with the model's response

    conversation_history.append(response)
    
    return response


if __name__ == '__main__':
    app.run()