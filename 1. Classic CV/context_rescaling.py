import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave


def remove_columns(image, mask, mode):
    n_row, n_col = image.shape[:2]
    if mode == 'vertical shrink':
        res_shape = (n_row - 1, n_col)
        if image.ndim == 3:
            result_R, result_G, result_B = np.empty(res_shape), np.empty(res_shape), np.empty(res_shape)
            for W_idx in range(n_col):
                result_R[:, W_idx] = image[:, W_idx, 0][~mask[:, W_idx]]
                result_G[:, W_idx] = image[:, W_idx, 1][~mask[:, W_idx]]
                result_B[:, W_idx] = image[:, W_idx, 2][~mask[:, W_idx]]
        elif image.ndim == 2:
            result = np.empty(res_shape)
            for W_idx in range(n_col):
                result[:, W_idx] = image[:, W_idx][~mask[:, W_idx]]
            return result
    else:
        res_shape = (n_row, n_col - 1)
        if image.ndim == 3:
            result_R, result_G, result_B = np.empty(res_shape), np.empty(res_shape), np.empty(res_shape)
            for H_idx in range(n_row):
                result_R[H_idx] = image[H_idx, :, 0][~mask[H_idx]]
                result_G[H_idx] = image[H_idx, :, 1][~mask[H_idx]]
                result_B[H_idx] = image[H_idx, :, 2][~mask[H_idx]]
        elif image.ndim == 2:
            result = np.empty(res_shape)
            for H_idx in range(n_row):
                result[H_idx] = image[H_idx, :][~mask[H_idx]]
            return result
    return np.dstack([result_R, result_G, result_B])


def compute_energy(image):
    Y_matrix = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2] 
    Y_gradient_x, Y_gradient_y = np.gradient(Y_matrix)
    Y_gradient =  np.sqrt(Y_gradient_x * Y_gradient_x + Y_gradient_y * Y_gradient_y)
    return Y_gradient


def compute_seam_matrix(energy, mode, mask = None):
    if mask is not None:
        energy +=  mask.astype(np.float64) * (256. * energy.shape[0] * energy.shape[1])

    if mode == 'vertical shrink' or mode == 'vertical':
        energy = energy.T

    n_row, n_col = energy.shape
    seam_matrix = energy.copy()
    seam_buff_line = np.full(n_col + 2, (256000000. * energy.shape[0] * energy.shape[1]))
    for idx_H in range(1, n_row):
        seam_buff_line[1: -1] = seam_matrix[idx_H - 1]
        min_neighbour_matrix = np.vstack([np.roll(seam_buff_line, 1), seam_buff_line, np.roll(seam_buff_line, -1)])
        min_neighbour_line = np.argmin(min_neighbour_matrix, axis= 0)[1: -1] - 1 + np.arange(n_col)
        seam_matrix[idx_H] += seam_matrix[idx_H - 1][min_neighbour_line]

    if mode == 'vertical shrink' or mode == 'vertical':
        seam_matrix = seam_matrix.T
    return seam_matrix


def remove_minimal_seam(image, seam_matrix, mode, mask = None):
    if mode == 'vertical shrink':
        seam_matrix = seam_matrix.T
    image = image.astype(np.float64)
    n_row = seam_matrix.shape[0]
    shrink_mask = np.full(seam_matrix.shape, False)

    W_idx = np.argmin(seam_matrix[-1])
    for H_idx in range(n_row - 1, 0, -1):
        shrink_mask[H_idx][W_idx] = True
        if W_idx == 0:
            W_idx = np.argmin(seam_matrix[H_idx - 1][0 : 2])
        else:
            W_idx += np.argmin(seam_matrix[H_idx - 1][W_idx - 1 : W_idx + 2]) - 1

    shrink_mask[0][W_idx] = True

    if mode == 'vertical shrink':
        shrink_mask = shrink_mask.T

    if mask is not None:
        cutted_mask = remove_columns(mask, shrink_mask, mode)
    else:
        cutted_mask = None

    return remove_columns(image, shrink_mask, mode).astype(np.uint8), cutted_mask


def create_empty_mask(image, image_name, format):
    res = np.dstack([np.zeros(image.shape[:2]), np.zeros(image.shape[:2]), image[...,2]])
    imsave(image_name + "_mask." + format, res.astype(np.uint8))


def read_mask(image_name, format):
    '''
    Green is +
    Red is - 
    '''
    mask_readed = imread(image_name + "_mask." + format)
    return (mask_readed[..., 1] > 0.01).astype(np.float64) - (mask_readed[..., 0] > 80).astype(np.float64)


def seam_carve(image, mode, mask = None):
    energy = compute_energy(image)
    seam_matrix = compute_seam_matrix(energy, mode, mask)
    return remove_minimal_seam(image, seam_matrix, mode, mask)


def image_compression(n_times, image, mode, mask = None):
    if mask is not None:
        mask_ = mask.copy()
    else:
        mask_ = mask

    image_ = image.copy()
    for i in range(n_times):
        image_, mask_  = seam_carve(image_, mode , mask_)

    return image_


image_name = "image"
format = "jpg"
img = imread(image_name + "." + format)

mask = None

print("Выберете режим: c/r:")
mask_mode = input()
if mask_mode == "c":
    create_empty_mask(img, image_name, format)
elif mask_mode == "r":
    mask = read_mask(image_name, format)
    print("Выберете направление и величину сжатия h/v (размер: " + str(img.shape[:2]) + ")")
    mode_size_input = input()
    mode_input, compression_size = mode_size_input.split()
    if mode_input == "h":
        mode = 'horizontal shrink'
    elif mode_input == "v":
        mode = 'vertical shrink'

    plt.imshow(image_compression(int(compression_size), img, mode, mask))
    plt.show()


