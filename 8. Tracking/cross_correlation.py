import os

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import match_template
from skimage import io
from detection import detection_cast, draw_detections, extract_detections
from tracker import Tracker


def gaussian(shape, x, y, dx, dy):
    """
    Return gaussian for tracking.
    """
    Y, X = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.exp(-((X - x) ** 2) / dx**2 - (Y - y) ** 2 / dy**2)


class CorrelationTracker(Tracker):
    """Generate detections and building tracklets."""

    def __init__(self, detection_rate=12, **kwargs):
        super().__init__(**kwargs)
        self.detection_rate = detection_rate  
        self.prev_frame = None  

    def build_tracklet(self, frame):
        detections = []
        gr_frame, gr_prev_frame = rgb2gray(frame), rgb2gray(self.prev_frame)
        # print(self.detection_history[-1])
        for label, xmin, ymin, xmax, ymax in self.detection_history[-1]:
            temp_corr_ = match_template(gr_frame, gr_prev_frame[ymin:ymax, xmin:xmax], pad_input=True) 
            dx, dy = (xmax - xmin), (ymax - ymin)
            gauss_corr = gaussian(gr_frame.shape, (xmax + xmin)/2, (ymax + ymin)/2, dx, dy)
            center = np.unravel_index(np.argmax(temp_corr_*gauss_corr), gauss_corr.shape)
            x_center, y_center = center[::-1]
            detections.append([label, max(x_center - dx//2, 0), max(y_center - dy//2, 0), min(x_center + dx//2, gr_frame.shape[1] - 1), min(y_center + dy//2, gr_frame.shape[0] - 1)])
        return detection_cast(detections)

    def update_frame(self, frame):
        if not self.frame_index:
            detections = self.init_tracklet(frame)
            self.save_detections(detections)
        elif self.frame_index % self.detection_rate == 0:
            detections = extract_detections(frame, min_confidence=0.37, labels=self.labels)
            detections = detections[detections[:, 3] - detections[:, 1] < 2 * frame.shape[1]/5]
            detections = self.bind_tracklet(detections)

            self.save_detections(detections)
        else:
            detections = self.build_tracklet(frame)

        self.detection_history.append(detections)
        self.prev_frame = frame
        self.frame_index += 1

        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    from moviepy.editor import VideoFileClip

    dirname = os.path.dirname(__file__)
    # input_clip = VideoFileClip(os.path.join(dirname, "data", "test.mp4"))
    # input_clip = VideoFileClip(os.path.join(dirname, "data", "test2.mp4"))
    input_clip = VideoFileClip(os.path.join(dirname, "data", "test1.mp4"))


    tracker = CorrelationTracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == "__main__":
    main()
