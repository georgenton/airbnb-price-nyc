import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    p = np.pi / 180
    a = 0.5 - np.cos((lat2 - lat1) * p) / 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 2 * R * np.arcsin(np.sqrt(a))

TIMES_SQUARE = (40.7580, -73.9855)
WALL_ST = (40.7060, -74.0086)

def distance_to_times_sq(lat, lon):
    return haversine(lat, lon, *TIMES_SQUARE)

def distance_to_wall_st(lat, lon):
    return haversine(lat, lon, *WALL_ST)
