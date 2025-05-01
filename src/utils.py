import numpy as np

def get_rainbow_color(t, speed=2):
    r = int((np.sin(t * speed + 0) + 1) * 127.5)
    g = int((np.sin(t * speed + 2) + 1) * 127.5)
    b = int((np.sin(t * speed + 4) + 1) * 127.5)
    return (b, g, r)
