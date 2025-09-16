from flask import Flask, jsonify, request
import logging
from gold_analyzer import run_analysis_for_api, to_compact_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/analyze', methods=['GET'])
def analyze_gold_endpoint():
    logging.info("API request received at /analyze")
    period = request.args.get('period', None)
    report = run_analysis_for_api(period=period)
    if "error" in report:
        logging.error(f"Analysis resulted in an error: {report['error']}")
        code = 503 if "unavailable" in str(report["error"]).lower() else 500
        return jsonify(report), code
    logging.info("Successfully generated analysis report.")
    return jsonify(report), 200

@app.route('/analyze/compact', methods=['GET'])
def analyze_compact_endpoint():
    logging.info("API request received at /analyze/compact")
    period = request.args.get('period', None)
    report = run_analysis_for_api(period=period)
    if "error" in report:
        logging.error(f"Analysis resulted in an error: {report['error']}")
        code = 503 if "unavailable" in str(report["error"]).lower() else 500
        return jsonify(report), code
    return jsonify(to_compact_report(report)), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
