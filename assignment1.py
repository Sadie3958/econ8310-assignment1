import numpy as np
import pandas as pd

def forecast(y, h):

    # Convert to numpy array 
    y = np.asarray(y, dtype=float)

    # Monthly seasonality 
    season = 12

    # If not enough data, fall back to last observation
    if len(y) < season:
        return np.repeat(y[-1], h)

    # Seasonal naive forecast
    forecasts = []
    for i in range(h):
        forecasts.append(y[-season + (i % season)])

    return np.array(forecasts)
