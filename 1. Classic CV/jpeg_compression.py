import os
from statistics import covariance
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio


def pca_compression(matrix, p):
    """ 
    Сжатие изображения с помощью PCA
    """

    centered_matrix = matrix - np.mean(matrix, axis = 1)[:, None]
    eigh_val, eigh_vec = np.linalg.eigh(centered_matrix @ centered_matrix.T)
    sorted_idx = np.argsort(eigh_val)[::-1]
    eigh_vec_sorted = eigh_vec[:, sorted_idx]
    needed_vec = eigh_vec_sorted[:, :p]    
    projection = np.dot(needed_vec.T, centered_matrix)
    return needed_vec, projection, np.mean(matrix, axis = 1)

def pca_decompression(compressed):
    """ 
    Разжатие изображения
    """
    result_img = []
    for i, comp in enumerate(compressed):
        eigh_vecs, projection, mtx_mean = comp
        result_img.append(eigh_vecs @ projection + mtx_mean[:,None])
    return np.clip(np.array(result_img).transpose(1,2,0), 0, 255)



def rgb2ycbcr(img):
    """ 
    Переход из пр-ва RGB в пр-во YCbCr
    """
    mtx = np.array([[0.299, 0.587, 0.114], [-0.1687, -0.3313, 0.5], [0.5, -0.4187, -0.0813]])
    bias = np.array([0,128,128])
    return np.dot(img, mtx.T) + bias

def ycbcr2rgb(img):
    """ 
    Переход из пр-ва YCbCr в пр-во RGB
    """
    mtx = np.array([[1, 0, 1.402], [1, -0.34414, -0.71414], [1, 1.77, 0]])
    bias = np.array([0, 128, 128])
    img_bias = img - bias
    return np.dot(img_bias, mtx.T)



def downsampling(component):
    """
    Уменьшаем цветовые компоненты в 2 раза
    """
    blurred = gaussian_filter(component, sigma=10)
    return blurred[::2,::2]


def dct(block):
    """
    Дискретное косинусное преобразование
    """
    X = (2*np.tile(np.arange(8).reshape(-1,1), (1,8)) + 1)/16 * np.pi
    Y = (2*np.tile(np.arange(8), (8, 1)) + 1)/16 * np.pi
    G = np.empty(block.shape)
    for u in range(8):
        for v in range(8):
            a_u = 1 if u else 1/np.sqrt(2)
            a_v = 1 if v else 1/np.sqrt(2)
            G[u][v] = a_u * a_v * np.sum(block * np.cos(u * X) * np.cos(v * Y))/4
    return G



# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block, quantization_matrix):
    """
    Квантование
    """
    return np.round(block/quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """
    Генерация матрицы квантования по Quality Factor
    """

    assert 1 <= q <= 100
    if q < 50:
        scale = 5000 / q 
    elif q <= 99:
        scale = 200 - 2 * q
    else:
        scale = 1

    res_quant_mtx = np.floor((50 + scale * default_quantization_matrix)/ 100)
    res = np.where(res_quant_mtx > 0, res_quant_mtx, 1)
    return res


def zigzag(block):

    """
    Зигзаг-сканирование
    """

    reversed_view = block[:, ::-1]
    zigzag_list = []
    dir_way = 1
    for i in range(7, -8, -1):
        zigzag_list = np.concatenate([zigzag_list, np.diag(reversed_view, k = i)[::(dir_way := -dir_way)]])
    return zigzag_list


def compression(zigzag_list):
    """
    Сжатие последовательности после зигзаг-сканирования
    """
    zigzag_list_np = np.array(zigzag_list)
    nonzero_idx_conc = np.nonzero(np.concatenate([[1], zigzag_list_np, [1]]))
    diff = (np.diff(nonzero_idx_conc) - 1)[0]
    idx = np.nonzero(diff)[0]
    zeros_num_arr = diff[idx]
    res = np.insert(zigzag_list_np[zigzag_list_np.nonzero()], np.tile(idx,(2)), np.concatenate([0 * zeros_num_arr, zeros_num_arr]))
    return res


def inverse_compression(compressed_list):
    """
    Разжатие последовательности
    """
    compressed_list_np = np.array(compressed_list)
    zeros_idx_prev = np.nonzero(compressed_list_np == 0)[0]
    zeros_vals = (compressed_list_np[zeros_idx_prev + 1] - 1).astype(np.uint8)
    compressed_new = np.delete(compressed_list_np, zeros_idx_prev + 1)
    zeros_idx_new = np.nonzero(compressed_new == 0)[0] + 1
    zeros_insert = np.zeros(np.sum(zeros_vals))
    zeros_insert_idx = np.repeat(zeros_idx_new, zeros_vals)
    result = np.insert(compressed_new, zeros_insert_idx, zeros_insert)
    return result


def inverse_zigzag(input):
    """
    Обратное зигзаг-сканирование
    """

    input_np = np.array(input)
    zigzag_matrix = np.zeros((8,8))
    dir_way = -1
    ptr = 0
    for i in range(-7, 8):
        k = 8 - abs(i)
        zigzag_matrix += np.diagflat(input_np[ptr : (ptr:= ptr + k)][::(dir_way:= -dir_way)], i)
    print(zigzag_matrix[::-1])
    return zigzag_matrix[::-1]





def inverse_quantization(block, quantization_matrix):
    """
    Обратное квантование
    """
    return block * quantization_matrix


def inverse_dct(block):
    """
    Обратное дискретное косинусное преобразование
    """
    U = np.tile(np.arange(8).reshape(-1,1), (1,8))
    V = np.tile(np.arange(8), (8, 1))

    a_U = np.full((8,8), 1.)
    a_V = np.full((8,8), 1.)

    a_U[0] = 1 / np.sqrt(2)
    a_V[:,0] = 1 /np.sqrt(2)

    f = np.empty((8,8))
    for x in range(8):
        for y in range(8):
            f[x][y] = np.sum(a_V * a_U * block * np.cos((2*x + 1) * U*np.pi/16) * np.cos((2*y + 1) * V * np.pi/ 16))/4
    return np.round(f)


def upsampling(component):
    """
    Увеличиваем цветовые компоненты в 2 раза
    """
    n_row, n_col = component.shape
    first_processing = np.repeat(component, np.ones(n_col, dtype=np.int64) * 2, axis=1)
    upscaled = np.repeat(first_processing, np.ones(n_row, dtype=np.int64) * 2, axis=0)
    return upscaled

def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
    ycbcr_img = rgb2ycbcr(rgb_img)
    ycbcr_img [..., 1] = gaussian_filter(ycbcr_img [..., 1], 10)
    ycbcr_img [..., 2] = gaussian_filter(ycbcr_img [..., 2], 10)
    plt.imshow(np.clip(ycbcr2rgb(ycbcr_img), 0, 255).astype(np.uint8))
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    ycbcr_img = rgb2ycbcr(rgb_img)
    ycbcr_img [..., 0] = gaussian_filter(ycbcr_img [..., 1], 10)
    plt.imshow(np.clip(ycbcr2rgb(ycbcr_img), 0, 255).astype(np.uint8))
    plt.savefig("gauss_2.png")

get_gauss_1()
get_gauss_2()