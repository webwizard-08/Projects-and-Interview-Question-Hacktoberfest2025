import requests
import sys

def get_bitcoin_price(api_key):
    """Fetch current Bitcoin price from CoinCap API."""
    url = "https://api.coincap.io/v2/assets/bitcoin"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data['data']['priceUsd'])
    except requests.RequestException as e:
        print(f"Error fetching Bitcoin price: {e}", file=sys.stderr)
        sys.exit(1)
    except (KeyError, ValueError) as e:
        print(f"Error parsing API response: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    # WARNING: Never hardcode API keys in production code
    # Use environment variables: api_key = os.environ.get('COINCAP_API_KEY')
    api_key = "25f3e1978b3b91e7d32b9db4d9950bfe9bd30cedba1e6bddaf3a87e94cd285ec"
    
    try:
        btc_amount = float(input("Enter the amount of Bitcoin: "))
        
        if btc_amount < 0:
            print("Error: Amount cannot be negative", file=sys.stderr)
            sys.exit(1)
        
        price = get_bitcoin_price(api_key)
        total_value = price * btc_amount
        
        print(f"${total_value:,.2f}")
        
    except ValueError:
        print("Error: Input must be a valid number", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled", file=sys.stderr)
        sys.exit(0)

if __name__ == "__main__":
    main()