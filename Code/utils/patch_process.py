import numpy as np
from skimage import exposure
from scipy import fftpack

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


def display_img(tiff_image):
    tiny_e = 1e-15
    tiff_image_abs = np.abs(tiff_image)
    tiff_image_norm = (tiff_image_abs - tiff_image_abs.min()) / (tiff_image_abs.max() - tiff_image_abs.min())  # rescale between 0 and 1
    p2, p98 = np.percentile(tiff_image_norm, (2, 98))
    tiff_image_rescale = exposure.rescale_intensity(tiff_image_norm, in_range=(p2, p98))
    rescale_img = 10*np.log10(tiff_image_rescale + tiny_e) # (tiff_image_rescale * 255).astype(np.uint8)

    return rescale_img
    
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

def grid_patch_extraction(image, patch_size, overlap=0):
    h, w = image.shape[:2]
    patch_h, patch_w = patch_size
    stride_h = patch_h - overlap
    stride_w = patch_w - overlap
    patches = []
    for y in range(0, h - patch_h + 1, stride_h):
        for x in range(0, w - patch_w + 1, stride_w):
            patch = image[y:y+patch_h, x:x+patch_w]
            patches.append(patch)
    return patches

def filter_2d(data_fft, radius, type): #boundary): # sigma_s):
    num, ny, nx = data_fft.shape
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

    filtered_fft = np.zeros([num, ny, nx], dtype=np.complex64)
    for i in range(num):
        filtered_fft[i] = data_fft[i] * mask

    return filtered_fft, mask

def downsample_sar(image, scale_factor):
    # Compute the 2D FFT of the image
    fft_image = fftpack.fft2(image)
    
    # Shift the zero-frequency component to the center
    fft_shifted = fftpack.fftshift(fft_image)
    
    # Calculate new dimensions
    batch, rows, cols = fft_shifted.shape
    new_rows, new_cols = int(rows / scale_factor), int(cols / scale_factor)
    
    # Crop the frequency domain
    crop_rows = slice(rows//2 - new_rows//2, rows//2 + new_rows//2)
    crop_cols = slice(cols//2 - new_cols//2, cols//2 + new_cols//2)
    fft_cropped = fft_shifted[:, crop_rows, crop_cols]
    
    # Shift back
    fft_cropped_shifted = fftpack.ifftshift(fft_cropped)
    
    # Inverse FFT
    downsampled_image = np.abs(fftpack.ifft2(fft_cropped_shifted))
    
    return downsampled_image