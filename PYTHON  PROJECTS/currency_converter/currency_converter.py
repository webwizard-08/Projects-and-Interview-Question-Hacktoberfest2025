# currency_converter.py
# Author: <your-name>
# Description: Simple currency converter using live exchange rates (fallback static rates).

import requests

def convert_currency(amount, from_currency="USD", to_currency="INR"):
    """Convert currency using an online API. Falls back to static rates if offline."""
    url = f"https://api.exchangerate.host/convert?from={from_currency}&to={to_currency}&amount={amount}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        result = data["result"]
        print(f"💰 {amount} {from_currency} = {result:.2f} {to_currency}")
    except Exception:
        print("⚠️ Could not fetch live rates. Using fallback static rate (1 USD = 83 INR).")
        result = amount * 83
        print(f"💰 {amount} {from_currency} = {result:.2f} {to_currency}")
    return result

if __name__ == "__main__":
    print("\n--- Currency Converter ---")
    amount = float(input("Enter amount: "))
    from_curr = input("From currency (e.g., USD): ").upper()
    to_curr = input("To currency (e.g., INR): ").upper()
    convert_currency(amount, from_curr, to_curr)
