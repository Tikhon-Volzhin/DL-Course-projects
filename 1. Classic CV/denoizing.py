import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from skimage.io import imshow, imsave, imread

def gaussian_kernel(size, sigma):
    limit = (size - 1) / (2 * sigma)
    spread = np.linspace(-limit, limit, size)
    gkern1d = np.exp(-0.5 * spread * spread)
    print(gkern1d[None, :].shape, gkern1d[:, None].shape)
    gkern2d = gkern1d[None, :] * gkern1d[:, None]
    gkern2d /= gkern2d.sum()
    return gkern2d


def fourier_transform(kernel, shape):
    w_img, h_img = shape
    w_kern, h_kern = kernel.shape
    w_diff, h_diff = w_img - w_kern, h_img - h_kern
    pad = [((w_diff + 1)//2, w_diff//2), ((h_diff+1)//2, h_diff//2)]
    padded_kernel = np.pad(kernel, pad, mode='constant', constant_values=(0,))
    return np.fft.fft2(np.fft.ifftshift(padded_kernel))


def inverse_kernel(H, threshold=1e-10):
    threshold_matrix = (np.abs(H) > threshold)
    return 1. / (threshold_matrix * H + (~threshold_matrix)) - (~threshold_matrix)


def inverse_filtering(blurred_img, kernel, threshold=1e-10):
    fft_img = np.fft.fft2(blurred_img)
    fourier_kernel = fourier_transform(kernel, blurred_img.shape)
    F_approx = fft_img * inverse_kernel(fourier_kernel, threshold)
    f_approx = np.fft.ifft2(F_approx)
    return np.abs(f_approx)


def wiener_filtering(blurred_img, kernel, K=0):
    if K == 0:
        return inverse_filtering(blurred_img, kernel)
    fft_img = np.fft.fft2(blurred_img)
    fft_kernel = fourier_transform(kernel, blurred_img.shape)
    F_approx = fft_img * np.conj(fft_kernel)/ (np.abs(fft_kernel) ** 2 + K)
    f_approx = np.fft.ifft2(F_approx)
    return np.abs(f_approx)


def compute_psnr(img1, img2):
    mse_img = np.mean((img1 - img2)**2)
    return 20./np.log(10)*np.log(255./np.sqrt(mse_img))
