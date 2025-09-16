from flask import Flask, jsonify, request
import logging
# Import the primary function from our professional analyzer script
from gold_analyzer import run_analysis_for_api

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

@app.route('/analyze', methods=['GET'])
def analyze_gold_endpoint():
    """
    API endpoint to trigger a new gold analysis.
    Accepts a 'period' query parameter (e.g., /analyze?period=2y).
    Defaults to the configuration in the analyzer script if not provided.
    """
    logging.info("API request received at /analyze")
    # Get 'period' from URL query parameters, with a default of None
    period = request.args.get('period', None)
    
    # Call the main function from your script. 
    # It's designed to run the analysis and return the report dictionary.
    # We pass the period if it exists, otherwise the function uses its own default.
    report = run_analysis_for_api(period=period)
    
    if "error" in report:
        logging.error(f"Analysis resulted in an error: {report['error']}")
        return jsonify(report), 500
    
    logging.info("Successfully generated analysis report.")
    return jsonify(report)

if __name__ == '__main__':
    # Run the Flask server, making it accessible on your local network
    # Use debug=False in a production environment
    app.run(host='0.0.0.0', port=5000, debug=True)
