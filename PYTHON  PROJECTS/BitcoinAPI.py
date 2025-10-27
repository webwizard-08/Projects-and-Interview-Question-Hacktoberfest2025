import requests
import json
import sys

try:
    btc_amount = input("Enter the amount of Bitcoin: ")

    try:
        btc_amount = float(btc_amount)

        url = "https://rest.coincap.io/v3/assets/bitcoin"
        key = "25f3e1978b3b91e7d32b9db4d9950bfe9bd30cedba1e6bddaf3a87e94cd285ec"
        headers = {
            "Authorization": f"Bearer {key}"
        }
        response = requests.get(url, headers=headers)
        data = response.json()
        price = float(data['data']['priceUsd'])
        multiple = price * btc_amount
        formatted_price = f"{multiple:,.4f}"
        print(f"${formatted_price}")

    except ValueError:
        print("Input is not a number")
        exit(1)

except requests.RequestException:
    print("Error fetching Bitcoin price")
    exit(1)
