import requests
import json
import os
import uuid
from datetime import datetime  

from pymongo import MongoClient
from urllib.parse import quote_plus

APP_ID = '5396649c7f6a4001b88e78d51f243d93'
FILE_LOCATION = r'C:\Users\afshe\OneDrive - Kumaraguru College of Technology\Desktop\API\data.json'
OUTPUT_FILE = r'C:\Users\afshe\OneDrive - Kumaraguru College of Technology\Desktop\API\output.json'

name = 'afsheenzaahrah25' 
pass_word = '9io1i0I1SlZvIZwZ'
username = quote_plus(name)
password = quote_plus(pass_word)
cluster = 'cluster0.jwxhgrj.mongodb.net'
database_name = 'demo_database'

MONGO_URL = f'mongodb+srv://{username}:{password}@{cluster}/{database_name}?retryWrites=true'

try:
    mongo_client = MongoClient(MONGO_URL)
    db = mongo_client[database_name]
    collection_name = "demo_data"
    collection = db[collection_name]

except Exception as e:
    print(f'Error connecting to MongoDB: {str(e)}')

class CurrencyConversion():
    def __init__(self):
        self.rates = None
        self.timestamp = None
        self.base = None

    def read_file(self):
        if os.path.exists(FILE_LOCATION):
            with open(FILE_LOCATION, 'r') as f:
                data = json.load(f)
                self.rates = data.get('rates', {})
                self.base = data.get('base', None)
                self.timestamp = data.get('timestamp', None)
        else:
            self.rates = None

    def call_api(self):
        url = f'https://openexchangerates.org/api/latest.json?app_id={APP_ID}'
        response = requests.get(url)
        return response.json()

    def save_data(self, api_call_response):
        with open(FILE_LOCATION, 'w') as f:
            json.dump(api_call_response, f, indent=4)

    def update_rates(self):
        self.read_file()
        if self.rates is None:
            api_call_response = self.call_api()
            if api_call_response:
                self.rates = api_call_response.get('rates', {})
                self.timestamp = api_call_response.get('timestamp', None)
                self.base = api_call_response.get('base', None)
                self.save_data(api_call_response)

    def currency_conversion(self, value=100, base_currency='INR'):
        if self.rates is not None:
            base_rate = self.rates.get(base_currency, None)
            if base_rate is not None:
                new_rates = {currency: rate / base_rate * value for currency, rate in self.rates.items()}
                print(f"Converted rates for {base_currency}:")
                print(json.dumps(new_rates, indent=4))
                print()
                return new_rates
            else:
                print(f"Base currency '{base_currency}' not found in rates data.")
        else:
            print("Rates not available.")

    def save_as_dict(self, file_location):
        if self.rates is not None:
            data_to_save = {
                'base': self.base,
                'timestamp': self.timestamp,
                'rates': self.rates
            }
            with open(file_location, 'w') as f:
                json.dump(data_to_save, f, indent=4)
                print(f"Data saved to {file_location}")
        else:
            print("Cannot save data as rates are not available.")

    def save_to_mongodb(self, base_currency, new_rates, timestamp):
        if new_rates:
            documents = []
            for currency, conversion_rate in new_rates.items():
                document = {
                    'id': str(uuid.uuid4()),  
                    'base_currency': base_currency,
                    'timestamp': timestamp,
                    'new_rates':new_rates  
                }
                documents.append(document)

            collection.insert_many(documents)
            print(f"Successfully inserted data into MongoDB for {base_currency}")
        else:
            print(f"No data to insert into MongoDB for {base_currency}")

if __name__ == "__main__":
    cc_instance = CurrencyConversion()

    if cc_instance.rates is None:
        cc_instance.update_rates()

    base_currencies = list(cc_instance.rates.keys())
    conversion_value = 100

    for base_currency in base_currencies:
        new_rates = cc_instance.currency_conversion(value=conversion_value, base_currency=base_currency)
        timestamp = datetime.now().isoformat() 
        cc_instance.save_to_mongodb(base_currency,new_rates,timestamp)

    cc_instance.save_as_dict(OUTPUT_FILE)