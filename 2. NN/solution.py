from platform import release
from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            return parameter - self.lr * parameter_grad

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            new_param =  parameter - self.lr* (self.momentum * updater.inertia + parameter_grad)
            updater.inertia = parameter_grad
            return new_param

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        return np.clip(inputs, 0, None)

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        layer_grads =  grad_outputs * np.heaviside(self.forward_inputs, 1.)
        return layer_grads


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        bounded_input = inputs - np.max(inputs, axis = 1)[:, None]
        inp_exp = np.exp(bounded_input)
        return inp_exp/inp_exp.sum(axis = 1)[:, None]

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        return grad_outputs * self.forward_outputs - np.diag(grad_outputs @ self.forward_outputs.T)[:, None] * self.forward_outputs

# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units = self.output_units

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, c)), output values

                n - batch size
                d - number of input units
                c - number of output units
        """
        return inputs @ self.weights + self.biases

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of input units
                c - number of output units
        """
        self.biases_grad = grad_outputs.sum(axis = 0)
        for i, j in zip(self.forward_inputs, grad_outputs):
            self.weights_grad += i[:, None] * j
        return grad_outputs @ self.weights.T


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((1,)), mean Loss scalar for batch

                n - batch size
                d - number of units
        """
        return -np.array([(y_gt * np.log(np.where(y_pred > eps, y_pred, eps))).sum()/ y_gt.shape[0]])

    def gradient_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n, d)), dLoss/dY_pred

                n - batch size
                d - number of units
        """
        return - y_gt/ np.where(y_pred > eps, y_pred, eps)/y_gt.shape[0]


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    model = Model(CategoricalCrossentropy(), SGDMomentum(0.001, momentum=0.9))
    model.add(Dense(100, input_shape = (x_train.shape[1], )))
    model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())
    model.fit(x_train, y_train, 256, 3, x_valid=x_valid, y_valid=y_valid)
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get('USE_FAST_CONVOLVE', False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    n_batch = inputs.shape[0]
    kernels = np.flip(kernels, axis=(2,3))
    c_size = kernels.shape[0]
    oh, ow = inputs.shape[2] + 2 * padding - kernels.shape[2] + 1, inputs.shape[3] + 2 * padding - kernels.shape[3] + 1
    padded_inputs = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values = (0,))
    result = np.empty((n_batch, c_size, oh, ow))
    for i in range(oh):
        for j in range(ow):
            result[..., i, j] = (padded_inputs[..., None,: ,  i : i + kernels.shape[2] , j : j + kernels.shape[3]] * kernels[None, ...]).sum(axis = (-3, -2, -1))
    return result


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name='kernels',
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_channels,),
            initializer=np.zeros
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, c, h, w)), output values

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        padding = (self.kernels.shape[-1] - 1)//2
        return convolve(inputs, self.kernels, padding) + self.biases[None, :, None, None]

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        self.biases_grad = grad_outputs.sum(axis = (0, 2, 3))
        self.kernels_grad = np.flip(convolve(self.forward_inputs.transpose(1, 0, 2, 3), np.flip(grad_outputs.transpose(1, 0, 2, 3), axis=(2,3)), 
                                             (self.kernels.shape[-1] - 1)//2).transpose(1, 0, 2, 3), axis=(2,3))
        return convolve(grad_outputs, np.flip(self.kernels.transpose(1, 0, 2, 3), axis=(2,3)), (self.kernels.shape[-1] - 1)//2)


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {'avg', 'max'}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)


    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, ih, iw)), input values

            :return: np.array((n, d, oh, ow)), output values

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        n, d, ih, iw = inputs.shape
        pool = self.pool_size
        oh, ow = ih // pool, iw // pool
        result = np.empty((n, d, oh, ow))
        pooling_split = inputs.reshape(n, d, oh, pool, ow, pool)
        
        if self.pool_mode == 'max':
            flatten_pool_split = pooling_split.transpose(0, 1, 2, 4, 3, 5). reshape(n, d, -1, pool * pool)
            self.max_idx = np.argmax(flatten_pool_split, axis = -1, keepdims=True).reshape(n, d, oh, ow, -1)
            result = np.max(pooling_split, axis = (3, 5))
        elif self.pool_mode == 'avg':
            result = np.mean(pooling_split, axis = (3, 5))
        return result


    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

            :return: np.array((n, d, ih, iw)), dLoss/dInputs

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        pool = self.pool_size
        n, d, oh, ow = grad_outputs.shape
        if self.pool_mode == 'max':
            result = np.zeros((n, d, oh, ow, pool * pool))
            np.put_along_axis(result, self.max_idx, grad_outputs[..., None], axis=-1)
            result = result.reshape(n, d, oh, ow, pool, pool).transpose(0, 1, 2, 4, 3, 5).reshape(self.forward_inputs.shape)
        elif self.pool_mode == 'avg':
            grad_outputs_res = grad_outputs.reshape(n, d, oh, 1, ow, 1)/(pool * pool)
            result = np.ones((n, d, oh, pool, ow, pool)) * grad_outputs_res
            result = result.reshape(self.forward_inputs.shape)
        return result


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name='beta',
            shape=(input_channels,),
            initializer=np.zeros
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name='gamma',
            shape=(input_channels,),
            initializer=np.ones
        )

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, d, h, w)), output values

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        n, d, h, w = inputs.shape
        if self.is_training:
            batch_mean = np.mean(inputs, axis = (0, 2, 3))
            batch_var = np.var(inputs, axis = (0, 2, 3))

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            batch_mean = batch_mean.reshape(1, -1, 1, 1)
            batch_var = batch_var.reshape(1, -1, 1, 1)

            self.input_centering = (inputs - batch_mean)
            self.input_stand = 1./ np.sqrt(batch_var + eps)
            self.X_norm = self.input_centering * self.input_stand
        else:
            self.X_norm = (inputs - self.running_mean.reshape(1, -1, 1, 1))/ np.sqrt(self.running_var.reshape(1, -1, 1, 1) + eps)
        return self.gamma.reshape(1, -1, 1, 1) * self.X_norm + self.beta.reshape(1, -1, 1, 1)

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        n, d, h, w = grad_outputs.shape
        self.gamma_grad = (grad_outputs * self.X_norm).sum(axis = (0, 2, 3))
        self.beta_grad = grad_outputs.sum(axis = (0, 2, 3))
        
        grad_outputs = self.gamma.reshape(1, -1, 1, 1) * grad_outputs
        var_grad = -0.5 * (grad_outputs * self.input_centering).sum(axis = (0, 2, 3)).reshape(1, -1, 1, 1) * np.power(self.input_stand, 3)
        mean_grad = -(grad_outputs * self.input_stand).sum(axis = (0, 2, 3)).reshape(1, -1, 1, 1) - 2. * var_grad * (self.input_centering).sum(axis = (0, 2, 3)).reshape(1, -1, 1, 1)/ (n * h * w)
        
        return grad_outputs * self.input_stand + 2. * var_grad * self.input_centering/ (n * h * w) + mean_grad/ (n * h * w)


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (np.prod(self.input_shape),)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, (d * h * w))), output values

                n - batch size
                d - number of input channels
                (h, w) - image shape
        """
        return inputs.reshape(inputs.shape[0], -1)

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of units
                (h, w) - input image shape
        """
        n, d, h, w = self.forward_inputs.shape
        return grad_outputs.reshape(n, d, h, w)


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        if self.is_training:
            mask = (np.random.uniform(size = (inputs.shape)) > self.p)
            self.forward_mask = mask
            Y = inputs * mask
        else:
            Y = (1. - self.p) * inputs
        return Y

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        return grad_outputs * self.forward_mask


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    
    model = Model(CategoricalCrossentropy(), SGDMomentum(0.006, momentum=0.9))

    model.add(Conv2D(16, kernel_size = 3 , input_shape = (3, 32, 32)))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D())

    model.add(Conv2D(32, kernel_size = 3))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Conv2D(32, kernel_size = 3))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D())

    model.add(Dropout(0.2))

    model.add(Conv2D(32, kernel_size = 3))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D())


    model.add(Conv2D(32, kernel_size = 3))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D())

    model.add(Conv2D(16, kernel_size = 3))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D())

    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Softmax())

    print(model)
    model.fit(x_train, y_train, 48, 14, x_valid=x_valid, y_valid=y_valid)
    return model

# ============================================================================