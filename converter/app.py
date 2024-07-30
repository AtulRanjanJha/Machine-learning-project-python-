from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

# Replace with your actual API key from ExchangeRate-API
API_KEY = '0e51f3e308b5b0e909b2ecba'

# Function to get exchange rates
def get_exchange_rate(base_currency, target_currency):
    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/{base_currency}"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        return data['conversion_rates'][target_currency]
    else:
        return None

# Route to serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint for currency conversion
@app.route('/convert', methods=['GET'])
def convert_currency():
    base_currency = request.args.get('base_currency')
    target_currency = request.args.get('target_currency')
    amount = float(request.args.get('amount'))
    rate = get_exchange_rate(base_currency, target_currency)
    if rate:
        converted_amount = amount * rate
        return jsonify({'converted_amount': converted_amount})
    else:
        return jsonify({'error': 'Invalid currency code or API error'}), 400

if __name__ == '__main__':
    app.run(debug=True)
