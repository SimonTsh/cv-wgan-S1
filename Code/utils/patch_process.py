import numpy as np

def sinfunc(t, A, w, p, c):
    return A * np.sin(w * t + p) + c

def sinc_1d(x, x0, dx):
    return np.sinc((x - x0) / dx)

def gaussian(x, a, b, c, d):
    return a * np.exp(-((x - b) / c)**2 / 2) + d

def impulse_response(x, a, b, c):
    return a * np.exp(-b * x) + c

def double_exponential(x, a1, b1, a2, b2, c):
    return a1 * np.exp(-b1 * x) + a2 * np.exp(-b2 * x) + c

def lorentzian(x, a, b, c, d):
    return a / (1 + ((x - b) / c)**2) + d


def win_patch(image, window_size, center_x, center_y):
    # Calculate the start and end coordinates for the window
    start_x = center_x - window_size // 2
    start_y = center_y - window_size // 2
    end_x = start_x + window_size
    end_y = start_y + window_size

    # Ensure the window is within the image bounds
    start_x = int(max(0, start_x))
    start_y = int(max(0, start_y))
    end_x = int(min(image.shape[1], end_x))
    end_y = int(min(image.shape[0], end_y))

    # Extract the window from the image
    window = image[start_y:end_y, start_x:end_x]

    return window

def filter_2d(data_fft, radius, type): #boundary): # sigma_s):
    ny, nx = data_fft.shape
    freq_x = np.fft.fftfreq(nx)
    freq_y = np.fft.fftfreq(ny)
    freq_x, freq_y = np.meshgrid(np.fft.fftshift(freq_x), np.fft.fftshift(freq_y))
    
    if type == 'circle':
        # circular filter
        mask = np.sqrt(freq_x**2 + freq_y**2) <= radius
    elif type == 'rect':
        # rectangular filter
        center_x = nx // 2
        center_y = ny // 2
        mask = (np.abs(freq_x - freq_x[center_x, center_y]) * nx <= radius / 2) & (np.abs(freq_y - freq_y[center_x, center_y]) * ny <= radius / 2)
    elif type == 'gaussian':
        # gaussian filter
        sigma_s = radius / 4
        sigma_f = 1 / (2 * np.pi * sigma_s)
        mask = np.exp(-((freq_x**2 + freq_y**2) / (2 * sigma_f**2)))    
    else:
        KeyError('Not known crop function')

    filtered_fft = data_fft * mask
    return filtered_fft, mask