import numpy as np
from math import ceil
import time
from skimage.io import imread, imshow, imsave
from matplotlib import pyplot as plt


def get_optimal_shift(color_img, green_img_fft):
    n_row, n_col = color_img.shape
    color_img_fft = np.fft.fft2(color_img)
    C_mtx = np.fft.ifft2(color_img_fft * green_img_fft).real
    indx_y, indx_x = np.unravel_index(np.argmax(C_mtx), C_mtx.shape)

    if (indx_y > n_row//2):
        indx_y = indx_y - n_row
    if (indx_x > n_col//2):
        indx_x = indx_x - n_col

    return indx_y, indx_x


def align(raw_img, g_coords):
    H_max, W_max = raw_img.shape
    H_split_size = H_max // 3
    H_border_cut, W_border_cut = H_split_size // 10, W_max // 10

    B_slice = raw_img[H_border_cut : H_split_size - H_border_cut][:, W_border_cut : W_max - W_border_cut]
    G_slice = raw_img[H_split_size + H_border_cut : 2 * H_split_size - H_border_cut][:, W_border_cut : W_max - W_border_cut]
    R_slice = raw_img[2 * H_split_size + H_border_cut : 3 * H_split_size - H_border_cut][:, W_border_cut : W_max - W_border_cut]

    Green_fft = np.conj(np.fft.fft2(G_slice))
    u_blue, v_blue = get_optimal_shift(B_slice, Green_fft)
    u_red, v_red = get_optimal_shift(R_slice, Green_fft)
    g_y, g_x = g_coords
    b_coords = ((g_y % H_split_size) + u_blue, g_x + v_blue)
    r_coords = ((g_y % H_split_size) + 2*H_split_size + u_red, g_x + v_red)
    alinged_img = np.dstack([np.roll(R_slice, (-v_red,-u_red), axis = (1,0)), G_slice, np.roll(B_slice, (-v_blue, -u_blue), axis = (1,0))])
    
    return alinged_img, b_coords, r_coords


def get_bayer_masks(n_rows, n_cols):

    n_rows_period, n_cols_period = ceil(n_rows/2), ceil(n_cols/2)
    red_mask_base = np.array([[False, True], [False, False]])
    green_mask_base = np.array([[True, False], [False, True]])
    blue_mask_base = np.array([[False, False], [True, False]])

    full_base = np.dstack([red_mask_base, green_mask_base, blue_mask_base])
    
    return np.tile(full_base, (n_rows_period, n_cols_period, 1)) [: n_rows, : n_cols]


def get_colored_img(raw_img):
    mask = get_bayer_masks(*raw_img.shape)
    return mask * raw_img[... , np.newaxis]


def bilinear_interpolation(colored_img):
    bayer_mask_pad = np.pad(get_bayer_masks(*colored_img.shape[:2]), ((1,1), (1,1), (0,0)), constant_values= (True,) )
    colored_img_pad = np.pad(colored_img, ((1,1), (1,1), (0,0)), constant_values= (0, ))
    result = np.copy(colored_img_pad)
    for ind, elem in np.ndenumerate(bayer_mask_pad):
        if not elem:
            W, H, C = ind
            bayer_local_mask = bayer_mask_pad[W - 1: W + 2, H - 1: H + 2, C]
            colored_img_local = colored_img_pad[W - 1: W + 2, H - 1: H + 2, C]
            result[W][H][C] = np.sum(colored_img_local) / np.sum(bayer_local_mask)
    return result[1 : -1, 1 : -1]