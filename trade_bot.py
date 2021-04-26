import alpaca_trade_api as tradeapi
import json
from datetime import datetime
from timeframe import TimeFrame

# authentication and connection details
api_key = 'PKVRNIRLRDOJKDR79WDE'
api_secret = '44DdwNppDhZChsM1q4QaeXHpZ9vpexyXg0wyYDmC'
base_url = 'https://paper-api.alpaca.markets'
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
# Get our account information.

account = api.get_account()
bank = account.buying_power
asset = 0

# Check if our account is restricted from trading.
if account.trading_blocked:
    print('Account is currently restricted from trading.')

# Check how much money we can use to open new positions.
print(account)
