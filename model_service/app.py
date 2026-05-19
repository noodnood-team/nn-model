from flask import Flask, request, jsonify
import torch
from PIL import Image
import numpy as np
import os
from torchvision import transforms

from load_model import build_model, get_champion_model, get_model_weight_from_pipeline, get_model_evaluation_from_pipeline

# Initialize Flask app
app = Flask(__name__)
app.json.sort_keys = False

model_name, pipeline_id, model_mse = get_champion_model()
model, device = build_model(model_name)
model_weight_path = get_model_weight_from_pipeline(pipeline_id=pipeline_id)
evaluation_result = get_model_evaluation_from_pipeline(pipeline_id=pipeline_id)

state_dict = torch.load(model_weight_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()
model = model.to(device)

# create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# make sure to use the same transformations as during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
])

# Function to predict calories from an image
def predict(image_path, model):
    
    # load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # เพิ่ม batch dimension
    
    # predict calories
    with torch.no_grad():
        output = model(img_tensor)
        calories = np.expm1(output.cpu().numpy()[0])
    
    return calories

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/health', methods=['GET'])
def health_check():
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    print(f"Health check request received from IP: {client_ip}")
    return jsonify({'status': 'ok'})

@app.route('/model_info', methods=['GET'])
def model_info():
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    print(f"Model info request received from IP: {client_ip}")
    model_performance = {
        "overall_mae": evaluation_result["overall_mae"],
        "overall_rmse": evaluation_result["overall_rmse"],
        "overall_mse": evaluation_result["overall_mse"],
        "mse_log_loss": evaluation_result["mse_log_loss"],
        "targets": {
            "calories": {
                "mae": evaluation_result["calories_mae"],
                "rmse": evaluation_result["calories_rmse"]
            },
            "protein": {
                "mae": evaluation_result["protein_mae"],
                "rmse": evaluation_result["protein_rmse"]
            },
            "carbs": {
                "mae": evaluation_result["carbs_mae"],
                "rmse": evaluation_result["carbs_rmse"]
            },
            "fat": {
                "mae": evaluation_result["fat_mae"],
                "rmse": evaluation_result["fat_rmse"]
            }
        }
    }
    
    return jsonify({
        'model_name': model_name,
        'pipeline_id': pipeline_id,
        'model_performance': model_performance
    })

# Route for handling prediction requests
@app.route('/predict', methods=['POST'])
def get_prediction():
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    print(f"Prediction request received from IP: {client_ip}")
    try:
        # receive the image file from the request
        image_file = request.files['image']
        image_path = os.path.join('uploads', image_file.filename)
        image_file.save(image_path)

        # predict calories using the model
        predicted_calories = predict(image_path, model)
        result = {
            'calories': float(predicted_calories[0]),
            'protein': float(predicted_calories[1]),
            'carbs': float(predicted_calories[2]),
            'fat': float(predicted_calories[3])
        }
        # remove the uploaded image after prediction
        os.remove(image_path)
        # return the prediction result as JSON
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8080, use_reloader=False)