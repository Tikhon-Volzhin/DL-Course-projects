import torch
import numpy as np


class Normalize(torch.nn.Module):
    """
    Normalization layer as described in ParseNet paper.
    """
    def __init__(self, scale, input_shape, name, **kwargs):
        super(Normalize, self).__init__(**kwargs)
        self.scale = scale
        self.name = '{}_gamma'.format(name)

        shape = (1, input_shape[1],) + tuple([1] * (len(input_shape) - 2))
        init_gamma = torch.tensor(self.scale * np.ones(shape, dtype=np.float32))
        self.register_parameter(self.name, torch.nn.Parameter(init_gamma))

    def forward(self, x):
        output = x / torch.norm(x, dim=1, keepdim=True)

        gamma = getattr(self, self.name)

        output *= gamma
        return output

class PriorBox(torch.nn.Module):
    """
    Generate the prior boxes of designated sizes and aspect ratios.
    """

    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, variances=None, **kwargs):
        super(PriorBox, self).__init__(**kwargs)

        self.waxis = 3
        self.haxis = 2
        self.img_size = img_size
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]
        if max_size:
            self.aspect_ratios.append(1.0)
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar  in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)

        if variances is None:
            variances = [0.1]
        self.variances = variances
        self.clip = True

    def get_output_shape_for(self, input_shape):
        num_priors_ = len(self.aspect_ratios)
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        num_boxes = num_priors_ * layer_width * layer_height
        return input_shape[0], num_boxes, 8

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)

    def forward(self, x):
        input_shape = x.shape

        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]

        img_width = self.img_size[0]
        img_height = self.img_size[1]

        box_widths = []
        box_heights = []

        for ar in self.aspect_ratios:
            if ar == 1 and not box_widths:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and box_widths:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))

        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        step_x = img_width / layer_width
        step_y = img_height / layer_height

        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)

        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1,1)


        num_priors_ = len(self.aspect_ratios)


        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors_))

        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4)

        if self.clip:
            prior_boxes = np.clip(prior_boxes, 0.0, 1.0)


        num_boxes = len(prior_boxes)
        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        prior_boxes = np.concatenate((prior_boxes, variances), axis=1)
        prior_boxes_tensor = torch.tensor(prior_boxes, device=x.device).unsqueeze(0)

        pattern = [input_shape[0], 1, 1]
        prior_boxes_tensor = torch.tile(prior_boxes_tensor, pattern)

        return prior_boxes_tensor


if __name__ == '__main__':
    obj = PriorBox((300, 300, 3), 1.0, 1.0)