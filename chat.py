from flask import Flask, request, jsonify
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

no_pattern_match_responses = [
    "What? If you can't speak clearly, at least tell me how you're feeling today.",
    "I don't get your gibberish. Try again, maybe start with how much you hate your job?",
    "Your babbling is tiresome. How about you just tell me what's wrong with your life?",
    "Can't understand a word. Why not just whine about your boss like everyone else?",
    "This is nonsense. Focus! What's annoying you today? Work? Love life?",
    "Make sense, please. Start with how miserable your job makes you feel.",
    "That's just noise. How about sharing something real, like your daily frustrations?",
    "Lost in translation. Try expressing your feelings about your career or relationships.",
    "What's that? Just tell me how much you despise your daily routine.",
    "I'm not getting it. Maybe describe how you're coping with your so-called life."
]

app = Flask(__name__)

# Load the model and intents just once when the Lambda function is initialized
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE, map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['user_input']
    response = get_response(user_input)
    return jsonify({'response': response})

# Function to get a response from the model
def get_response(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    probability = probs[0][predicted.item()]
    if probability.item() > 0.1:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return random.choice(no_pattern_match_responses)
# AWS Lambda handler function
def lambda_handler(event, context):
    user_input = event["user_input"]
    response = get_response(user_input)
    return {
        'statusCode': 200,
        'body': json.dumps({'response': response})
    }

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
