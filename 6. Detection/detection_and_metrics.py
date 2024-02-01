# ============================== 1 Classifier model ============================

from torch import nn

class SimpleCNNModel(nn.Sequential):
        def __init__(self, input_shape):
            super(SimpleCNNModel, self).__init__()
            H, W, C = input_shape
            self.H_out, self.W_out = (((H - 2)//2 - 7)//2 - 2), (((W - 2)//2 - 7)//2 - 3)
            self.input_shape = input_shape
            self.conv1 = nn.Conv2d(C, 32, 3)
            self.relu1 = nn.ReLU()
            self.conv1_bn = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.drop1 = nn.Dropout(0.2)

            self.conv2 = nn.Conv2d(32, 64, 4)
            self.relu2 = nn.ReLU()
            self.conv2_bn = nn.BatchNorm2d(64)

            self.conv3 = nn.Conv2d(64, 64, 3)
            self.relu3 = nn.ReLU()
            self.conv3_bn = nn.BatchNorm2d(64)
            
            self.conv4 = nn.Conv2d(64, 128, 3)
            self.relu4 = nn.ReLU()
            self.conv4_bn = nn.BatchNorm2d(128)
            self.pool4 = nn.MaxPool2d(2, 2)
            self.drop4 = nn.Dropout(0.2)

            self.conv5 = nn.Conv2d(128, 128, (3, 4))
            self.relu5 = nn.ReLU()
            self.conv5_bn = nn.BatchNorm2d(128)

            self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
            self.fc1 = nn.Linear(128 * self.H_out * self.W_out, 64)
            self.fc1_bn = nn.BatchNorm1d(64)
            self.relu_fc1 = nn.ReLU()
            self.fc2 = nn.Linear(64, 16)
            self.fc2_bn = nn.BatchNorm1d(16)
            self.relu_fc2 = nn.ReLU()
            self.fc3 = nn.Linear(16, 2)
            self.softmax = nn.Softmax(dim=1)

def get_cls_model(input_shape): 
    return SimpleCNNModel(input_shape)


def fit_cls_model(X, y):
    import torch
    import numpy as np
    from  torchvision import transforms 
    model = get_cls_model((40, 100, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epoch = 12
    batch_size = 64
    transform_ = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(12)])
    split_idx = [x for x in range(batch_size, X.shape[0], batch_size)]
    for i in range(epoch):
        shuffle_idx = np.random.permutation(X.shape[0])
        X_splitted, y_splitted = np.split(X[shuffle_idx], split_idx), np.split(y[shuffle_idx], split_idx)
        for X_batch_, y_batch_ in zip(X_splitted, y_splitted):
            X_batch_ = transform_(X_batch_)
            optimizer.zero_grad()
            y_pred_proba = model(X_batch_)
            loss = torch.nn.functional.cross_entropy(y_pred_proba, y_batch_)
            loss.backward()
            optimizer.step()
    return model

# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    from torch import nn
    cls_ = SimpleCNNModel((40, 100, 1))
    cls_.load_state_dict(cls_model)
    class DetectionModule(nn.Module):
        def __init__(self, cls_model):
            super(DetectionModule, self).__init__()
            H_out, W_out = cls_model.H_out, cls_model.W_out
            from collections import OrderedDict
            self.cls_model_cnn = nn.Sequential(*list(cls_model.children())[:19])
            self.conv1 = nn.Conv2d(128, 64, (H_out, W_out))
            conv1_state = OrderedDict()
            conv1_state['weight'] = cls_model.fc1.state_dict()['weight'].reshape(64, 128, H_out, W_out)
            conv1_state['bias'] = cls_model.fc1.state_dict()['bias']
            self.conv1.load_state_dict(conv1_state)

            self.bn1 = nn.BatchNorm2d(64) 
            self.bn1.load_state_dict(cls_model.fc1_bn.state_dict())

            self.relu1 = nn.ReLU()

            self.conv2 = nn.Conv2d(64, 16, 1)
            conv2_state = OrderedDict()
            conv2_state['weight'] = cls_model.fc2.state_dict()['weight'].reshape(16, 64, 1, 1)
            conv2_state['bias'] = cls_model.fc2.state_dict()['bias']
            self.conv2.load_state_dict(conv2_state)

            self.bn2 = nn.BatchNorm2d(16)
            self.bn2.load_state_dict(cls_model.fc2_bn.state_dict())

            self.relu2 = nn.ReLU()

            self.conv3 = nn.Conv2d(16, 2, 1)
            conv3_state = OrderedDict()
            conv3_state['weight'] = cls_model.fc3.state_dict()['weight'].reshape(2, 16, 1, 1)
            conv3_state['bias'] = cls_model.fc3.state_dict()['bias']
            self.conv3.load_state_dict(conv3_state)

            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.cls_model_cnn(x)
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.softmax(self.conv3(x))
            return x
    

    return DetectionModule(cls_)
                                    
# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    import torch
    import numpy as np
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    detected = {}
    transform = A.Compose([A.ToFloat(), ToTensorV2()])
    with torch.no_grad():
        detection_model.eval()
        for filename, np_img in dictionary_of_images.items():
            H_max, W_max = (np_img.shape[0] - 40)//4 + 1, (np_img.shape[1] - 100)//4 + 1
            img_ = transform(image = np_img)['image']
            heat_map_ = detection_model(img_[None, ...])
            heat_map_cut = heat_map_.squeeze(0)[1, :H_max, :W_max]
            res = []
            for i in range(H_max):
                for j in range(W_max):
                    res.append([i * 4, j * 4, 40, 100, heat_map_cut[i][j].item()])
            detected[filename] = res
    return detected


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    if first_bbox[0] == second_bbox[0] and first_bbox[1] == second_bbox[1]:
        return 1.
    
    first_area, second_area = first_bbox[2] * first_bbox[3], second_bbox[2] * second_bbox[3]
    if first_bbox[0] > second_bbox[0]:
        first_bbox, second_bbox = second_bbox, first_bbox

    if first_bbox[1] < second_bbox[1]:
        intersection =  max(first_bbox[0] + first_bbox[2] - second_bbox[0], 0.) * max(first_bbox[1] + first_bbox[3] - second_bbox[1], 0.)
    else:
        intersection =  max(first_bbox[0] + first_bbox[2] - second_bbox[0], 0.) * max(second_bbox[1] + second_bbox[3] - first_bbox[1], 0.)

    return intersection/(first_area + second_area - intersection)


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    import numpy as np
    iou_thr = 0.5
    confidence = []
    tp_fp = []
    gt_len = 0
    for filename in pred_bboxes:
        pred_bboxes_sorted = sorted(pred_bboxes[filename], key=lambda x: x[4], reverse=True)
        gt_list = gt_bboxes[filename]
        gt_selected = set()
        gt_len += len(gt_list)
        for bbox_ in pred_bboxes_sorted:
            bbox_rect = [*bbox_[:-1]]
            confidence.append(bbox_[-1])
            gt_max_idx = -1
            max_iou = 0.
            for idx, gt_ in enumerate(gt_list):
                if idx in gt_selected:
                    continue
                iou_ = calc_iou(gt_, bbox_rect)
                if iou_ > iou_thr and iou_ > max_iou:
                    gt_max_idx = idx
                    max_iou = iou_
            if gt_max_idx == -1:
                tp_fp.append(0)
            else:
                tp_fp.append(1)
                gt_selected.add(gt_max_idx)

    confidence_st, tp_fp_st = np.array(confidence), np.array(tp_fp)
    sort_idxs = np.argsort(confidence)
    confidence_st, tp_fp_st = confidence_st[sort_idxs], tp_fp_st[sort_idxs]

    _, indx = np.unique(confidence_st, return_index=True)
    TP = np.cumsum(tp_fp_st[::-1])[::-1][indx]
    cummulative_range = np.arange(len(tp_fp_st), 0, -1)[indx]
    presicion, recall = np.concatenate([TP/ cummulative_range, [1.]]), np.concatenate([TP/ gt_len, [0.]])
    return -np.trapz(x = recall, y = presicion)


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr = 0.6):
    result = {}
    for filename in detections_dictionary:
        pred_bboxes_sorted = sorted(detections_dictionary[filename], key=lambda x: x[4], reverse=True)
        chosen_bboxes = []
        deleted_idxs = set()
        for idx_, bbox_ in enumerate(pred_bboxes_sorted):
            if idx_ in deleted_idxs:
                continue
            chosen_bboxes.append(bbox_)
            bbox_rect_ = [*bbox_[:-1]]
            if idx_ == len(pred_bboxes_sorted) - 1:
                break
            for lower_idx__, lower_bbox__ in enumerate(pred_bboxes_sorted[idx_ + 1:], idx_ + 1):
                if lower_idx__ in deleted_idxs:
                    continue
                if calc_iou( [*lower_bbox__[:-1]], bbox_rect_) > iou_thr:
                    deleted_idxs.add(lower_idx__)
        result[filename] = chosen_bboxes
    return result



def vis(img_path, gt_, model):
    import torch
    from PIL import Image
    import matplotlib.pyplot as plt 
    import albumentations as A
    import os
    import numpy as np
    import matplotlib.patches as patches
    import cv2
    for path in os.listdir(img_path):
            theres = 0.95
            img_ = np.array(Image.open(os.path.join(img_path, path)))
            d_ = {path: img_}
            pred_ = get_detections(model, d_)[path]
            fig, ax = plt.subplots()
            ax.imshow(img_)
            for elem in pred_:
                if theres < elem[4]:
                    print(elem)
                    rect = patches.Rectangle((elem[1]/1., elem[0]/1.), elem[3]/1., elem[2]/1.,  linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)        
            plt.show()
    
# import torch
# model = get_cls_model((40, 100, 1))
# model.load_state_dict(torch.load('classifier_model.pth', map_location='cpu'))
# detection_m = get_detection_model(model.state_dict())
# vis('./tests/04_unittest_detector_input/test_imgs', './tests/04_unittest_detector_input/true_detections.json', detection_m)