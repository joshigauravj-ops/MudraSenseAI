import unittest
from unittest.mock import patch

import pandas as pd

from market_tools import fetch_price_history


class PriceHistoryTests(unittest.TestCase):
    def test_fetch_price_history_normalizes_ticker_and_returns_ohlcv_bars(self):
        history = pd.DataFrame(
            {
                "Open": [100.0, 102.0],
                "High": [105.0, 107.0],
                "Low": [99.0, 101.0],
                "Close": [103.0, 106.0],
                "Volume": [1500, 1800],
            },
            index=pd.to_datetime(["2026-09-25", "2026-09-28"]),
        )
        with patch("market_tools.yf.Ticker") as ticker_factory:
            ticker_factory.return_value.history.return_value = history

            bars = fetch_price_history("reliance", period="5d", interval="1d")

        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0]["close"], 103.0)
        self.assertEqual(bars[1]["volume"], 1800)
        ticker_factory.assert_called_once_with("RELIANCE.NS")

    def test_fetch_price_history_rejects_unsupported_period(self):
        with patch("market_tools.yf.Ticker") as ticker_factory:
            result = fetch_price_history("RELIANCE", period="10y")

        self.assertIsInstance(result, str)
        ticker_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()