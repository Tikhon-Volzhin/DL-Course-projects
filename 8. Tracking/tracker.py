import os

import numpy as np

from detection import detection_cast, draw_detections, extract_detections
from metrics import iou_score


class Tracker:
    """Generate detections and build tracklets."""

    def __init__(self, return_images=True, lookup_tail_size=80, labels=None):
        self.return_images = return_images 
        self.frame_index = 0
        self.labels = labels 
        self.detection_history = [] 
        self.last_detected = {}
        self.tracklet_count = 0 

        self.lookup_tail_size = lookup_tail_size

    def new_label(self):
        self.tracklet_count += 1
        return self.tracklet_count - 1

    def init_tracklet(self, frame):
        detections = extract_detections(frame, labels=self.labels)
        detections[:, 0] = np.array([self.new_label() for _ in range(detections.shape[0])])
        
        return detections

    @property
    def prev_detections(self):
        detections = []
        chosen_id = set()
        for detect_ in self.detection_history[-1: -self.lookup_tail_size - 1:-1]:
            for detect__ in detect_[::-1]:
                if detect__[0] in chosen_id:
                    continue
                detections.append(detect__)
                chosen_id.add(detect__[0])
        return detection_cast(detections)

    def bind_tracklet(self, detections):
        """
        Set id at first detection column.
        Find best fit between detections and previous detections.
        """
        detections = detections.copy()
        prev_detections = self.prev_detections
        IOU_MIN = 0.4
        iou_corr = []
        idx_to_id = {}
        chosen_ids = set()
        for idx_, detection_ in enumerate(detections):
            for prev_detection_ in prev_detections:
                iou_ = iou_score(prev_detection_[1:], detection_[1:])
                if iou_ > IOU_MIN:
                    iou_corr.append([iou_, idx_, prev_detection_[0]])
        for iou_, idx_, id_ in sorted(iou_corr, key=lambda x: x[0], reverse=True):
            if id_ in chosen_ids or idx_ in idx_to_id:
                continue
            idx_to_id[idx_] = id_
            chosen_ids.add(id_)
        for i in range(detections.shape[0]):
            if i in idx_to_id:
                detections[i, 0] = idx_to_id[i]
            else:
                detections[i, 0] = self.new_label()
        return detection_cast(detections)

    def save_detections(self, detections):
        for label in detections[:, 0]:
            self.last_detected[label] = self.frame_index

    def update_frame(self, frame):
        if not self.frame_index:
            detections = self.init_tracklet(frame)
        else:
            detections = extract_detections(frame, labels=self.labels)
            detections = self.bind_tracklet(detections)

        self.save_detections(detections)
        self.detection_history.append(detections)
        self.frame_index += 1

        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    from moviepy.editor import VideoFileClip

    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, "data", "test.mp4"))

    tracker = Tracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == "__main__":
    main()
