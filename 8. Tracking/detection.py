import os

import numpy as np
import torch
from config import VOC_CLASSES, bbox_util, model
from PIL import Image
from skimage import io
from skimage.transform import resize
from utils import get_color


def detection_cast(detections):
    return np.array(detections, dtype=np.int32).reshape((-1, 5))


def rectangle(shape, ll, rr, line_width=5):
    ll = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(ll, 0))
    rr = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(rr, 0))
    result = []

    for c in range(line_width):
        for i in range(ll[0] + c, rr[0] - c + 1):
            result.append((i, ll[1] + c))
            result.append((i, rr[1] - c))
        for j in range(ll[1] + c + 1, rr[1] - c):
            result.append((ll[0] + c, j))
            result.append((rr[0] - c, j))

    return tuple(zip(*result))


IMAGENET_MEAN = np.array([103.939, 116.779, 123.68]).reshape(1, 1, 3)


def image2tensor(image):
    image = image.astype(np.float32)  
    image = resize(image, output_shape=(300, 300, 3))  
    image = image[..., [2, 1, 0]] 
    image = image - IMAGENET_MEAN
    image = image.transpose([2, 0, 1])  
    tensor = torch.tensor(image.copy()).unsqueeze(0).float()
    # tensor.shape == (1, channels, height, width)
    return tensor


@torch.no_grad()
def extract_detections(frame, min_confidence=0.6, labels=None):
    """
    Extract detections from frame.
    """

    W, H = frame.shape[:-1]
    scale_vector = np.array([H, W, H, W]).reshape(1, 4)
    input_tensor = image2tensor(frame)

    results = bbox_util.detection_out(model(input_tensor).numpy(), confidence_threshold=min_confidence)

    if labels is not None:
        result_labels = results[:, 0].astype(np.int32)
        indices = [
            index
            for index, label in enumerate(result_labels)
            if VOC_CLASSES[label - 1] in labels
        ]
        results = results[indices]
    results = np.array(results).reshape(-1, 6)
    results = results[:, [0, 2, 3, 4, 5]]
    results[:, [1, 2, 3, 4]] = (results[:, [1, 2, 3, 4]] - 0.5) * scale_vector + scale_vector/2
    results = results[results[:, 3] - results[:, 1] < H//2]
    return detection_cast(results)


def draw_detections(frame, detections):
    """
    Draw detections on frame.
    """
    frame = frame.copy()

    for detection in detections:
        label_id, xmin, ymin, xmax, ymax = detection
        color = get_color(label_id)
        rr, cc = rectangle(frame.shape, (ymin, xmin), (ymax, xmax))
        frame[rr, cc] = color

    return frame


def main():
    dirname = os.path.dirname(__file__)
    frame = Image.open(os.path.join(dirname, "tests", "02_unittest_tracker_input", "frames", "000000.jpg"))
    frame = np.array(frame)

    detections = extract_detections(frame)
    frame = draw_detections(frame, detections)

    io.imshow(frame)
    io.show()


if __name__ == "__main__":
    main()
