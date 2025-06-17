import json
import requests

def lambda_handler(event, context):
    try:
        # Parse incoming data
        body = json.loads(event['body'])

        # Prepare input for FastAPI
        data = {
            "cycle": body.get('cycle'),
            "ambient_temperature": body.get('ambient_temperature'),
            "capacity": body.get('capacity'),
            "voltage_measured": body.get('voltage_measured'),
            "current_measured": body.get('current_measured'),
            "temperature_measured": body.get('temperature'),
            "current_load": body.get('load_current'),
            "voltage_load": body.get('load_voltage'),
            "time": body.get('time')
        }

        # FastAPI URL (Your ML API URL)
        fastapi_url = "https://your-ml-api-url.com/predict"

        response = requests.post(fastapi_url, json=data)
        result = response.json()

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Prediction Successful',
                'prediction': result
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
