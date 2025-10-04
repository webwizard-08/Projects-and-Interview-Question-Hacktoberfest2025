import requests

def convert_currency(amount, from_currency, to_currency):
    # Force uppercase so case doesn't matter
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_currency}&to={to_currency}"
    response = requests.get(url)
    data = response.json()

    if "rates" in data and to_currency in data["rates"]:
        return data["rates"][to_currency]
    return None

print("💱 Currency Converter")
amount = float(input("Enter amount: "))
from_currency = input("From currency (e.g., USD, EUR, INR): ").strip()
to_currency = input("To currency (e.g., USD, EUR, INR): ").strip()

result = convert_currency(amount, from_currency, to_currency)
if result:
    print(f"{amount} {from_currency.upper()} = {result:.2f} {to_currency.upper()}")
else:
    print("❌ Conversion failed. Please check currency codes or internet connection.")
