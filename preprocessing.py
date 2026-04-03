import urllib.request
import numpy as np

def load_from_url(url):
    response = urllib.request.urlopen(url)
    data = response.read().decode('utf-8').strip().split('\n')[1:]
    values = [float(row.split(',')[1]) for row in data]
    return np.array(values).reshape(-1, 1)

def compute_statistics(data):
    mu = np.mean(data)
    sigma = np.std(data)
    if sigma == 0:
        sigma = 1e-8
    return mu, sigma

def standardize(data, mu, sigma):
    return (data - mu) / sigma

def create_windows(data, T):
    N = len(data)
    windows = np.array([data[i:i+T] for i in range(N - T + 1)])
    return windows