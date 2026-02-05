import os
import sys
sys.path.append('C:\\Projects\\mlproject')
from flask import Flask, request, jsonify
from src.pipeline.predict_pipeline import PredictionPipeline
from src.logger import logging
from src.exception import CustomException

app = Flask(__name__)

# Initialize prediction pipeline
predictor = None

def initialize_predictor():
    """Initialize the prediction pipeline"""
    global predictor
    try:
        logging.info("Initializing prediction pipeline...")
        predictor = PredictionPipeline()
        logging.info("Prediction pipeline initialized successfully")
        print("✓ Prediction pipeline initialized successfully!")
        return True
    except Exception as e:
        logging.error(f"Error initializing predictor: {str(e)}")
        print(f"✗ Error initializing predictor: {str(e)}")
        raise CustomException(e, sys)

@app.route('/', methods=['GET'])
def home():
    """Home page - simple prediction form"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Student Math Score Prediction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                color: #555;
                font-weight: bold;
            }
            input, select {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                width: 100%;
                font-size: 16px;
            }
            button:hover {
                background-color: #45a049;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                background-color: #e8f5e9;
                border-left: 4px solid #4CAF50;
                display: none;
            }
            .error {
                background-color: #ffebee;
                border-left-color: #f44336;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📚 Student Math Score Prediction</h1>
            <form id="predictionForm">
                <div class="form-group">
                    <label for="gender">Gender:</label>
                    <select id="gender" name="gender" required>
                        <option value="">Select Gender</option>
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="race_ethnicity">Race/Ethnicity:</label>
                    <select id="race_ethnicity" name="race/ethnicity" required>
                        <option value="">Select Race/Ethnicity</option>
                        <option value="group A">Group A</option>
                        <option value="group B">Group B</option>
                        <option value="group C">Group C</option>
                        <option value="group D">Group D</option>
                        <option value="group E">Group E</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="parental_level_of_education">Parental Level of Education:</label>
                    <select id="parental_level_of_education" name="parental level of education" required>
                        <option value="">Select Education Level</option>
                        <option value="some high school">Some High School</option>
                        <option value="high school">High School</option>
                        <option value="some college">Some College</option>
                        <option value="associate's degree">Associate's Degree</option>
                        <option value="bachelor's degree">Bachelor's Degree</option>
                        <option value="master's degree">Master's Degree</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="lunch">Lunch Type:</label>
                    <select id="lunch" name="lunch" required>
                        <option value="">Select Lunch Type</option>
                        <option value="standard">Standard</option>
                        <option value="free/reduced">Free/Reduced</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="test_preparation_course">Test Preparation Course:</label>
                    <select id="test_preparation_course" name="test preparation course" required>
                        <option value="">Select Course</option>
                        <option value="none">None</option>
                        <option value="completed">Completed</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="reading_score">Reading Score (0-100):</label>
                    <input type="number" id="reading_score" name="reading score" min="0" max="100" required>
                </div>

                <div class="form-group">
                    <label for="writing_score">Writing Score (0-100):</label>
                    <input type="number" id="writing_score" name="writing score" min="0" max="100" required>
                </div>
                
                <button type="submit">Predict Math Score</button>
            </form>
            
            <div id="result" class="result"></div>
        </div>
        
        <script>
            document.getElementById('predictionForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData(document.getElementById('predictionForm'));
                const data = Object.fromEntries(formData);
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('result');
                    
                    if (response.ok) {
                        resultDiv.innerHTML = `<strong>Predicted Math Score: ${result.prediction.toFixed(2)}</strong>`;
                        resultDiv.classList.remove('error');
                    } else {
                        resultDiv.innerHTML = `<strong>Error: ${result.error}</strong>`;
                        resultDiv.classList.add('error');
                    }
                    resultDiv.style.display = 'block';
                } catch (error) {
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = `<strong>Error: ${error.message}</strong>`;
                    resultDiv.classList.add('error');
                    resultDiv.style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for making predictions
    Expects JSON data with student features
    """
    try:
        logging.info("Prediction request received")
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        logging.info(f"Input data received: {list(data.keys())}")
        
        # Make prediction using pipeline
        prediction = predictor.predict_single(data)
        
        logging.info(f"Prediction: {prediction}")
        
        return jsonify({
            'prediction': prediction,
            'message': 'Prediction successful'
        }), 200
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    Expects JSON array of student records
    """
    try:
        logging.info("Batch prediction request received")
        
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Expected JSON array of records'}), 400
        
        # Make batch predictions using pipeline
        predictions = predictor.predict_batch(data)
        
        logging.info(f"Batch predictions completed: {len(predictions)} records")
        
        return jsonify({
            'predictions': predictions,
            'count': len(predictions),
            'message': 'Batch prediction successful'
        }), 200
        
    except Exception as e:
        logging.error(f"Error during batch prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'preprocessor_loaded': predictor.preprocessor is not None
    }), 200

@app.route('/info', methods=['GET'])
def info():
    """Get model information"""
    return jsonify(predictor.get_model_info()), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    logging.error(f"Server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("="*70)
    print("LOADING ML MODEL FOR STUDENT MATH SCORE PREDICTION")
    print("="*70)
    
    # Initialize prediction pipeline
    initialize_predictor()
    
    print("\n" + "="*70)
    print("STARTING FLASK SERVER")
    print("="*70)
    print("✓ Server running at: http://localhost:5000")
    print("✓ Home page: http://localhost:5000")
    print("✓ API prediction: POST http://localhost:5000/predict")
    print("✓ Batch prediction: POST http://localhost:5000/predict_batch")
    print("✓ Health check: GET http://localhost:5000/health")
    print("✓ Model info: GET http://localhost:5000/info")
    print("="*70 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0')
