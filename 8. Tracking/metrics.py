def iou_score(bbox1, bbox2):
    """
    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4
    
    first_area, second_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]), (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    intersection =  max(min(bbox1[2], bbox2[2])  - max(bbox2[0], bbox1[0]) , 0.) * max(min(bbox1[3], bbox2[3])  - max(bbox2[1], bbox1[1]), 0.)
    return intersection/(first_area + second_area - intersection)

def motp(obj, hyp, threshold=0.5):
    """
    Calculate MOTP
    """

    dist_sum = 0  
    match_count = 0

    matches = {}  

    for frame_obj, frame_hyp in zip(obj, hyp):
        chosen_ids_obj_ = set()
        chosen_ids_hyp_ = set()
        obj_dict_, hyp_dict_ = {frame_[0]: frame_[1:] for frame_ in frame_obj}, {frame_[0]: frame_[1:] for frame_ in frame_hyp}
        matches_ = {}
        for obj_id_, hyp_id_ in matches.items():
            if obj_id_ in obj_dict_ and hyp_id_ in hyp_dict_ and (iou_ := iou_score(obj_dict_[obj_id_], hyp_dict_[hyp_id_])) > threshold:
                dist_sum += iou_
                match_count +=1
                matches_[obj_id_] = hyp_id_
                chosen_ids_obj_.add(obj_id_)
                chosen_ids_hyp_.add(hyp_id_)

        pairwise_det_ = []
        for obj_id_, fr_obj_ in obj_dict_.items():
            if obj_id_ in chosen_ids_obj_:
                continue
            for hyp_id_, fr_hyp_ in hyp_dict_.items():
                if hyp_id_ in chosen_ids_hyp_:
                    continue
                iou_ = iou_score(fr_obj_, fr_hyp_)
                if iou_ > threshold:
                    pairwise_det_.append([iou_, obj_id_, hyp_id_])
        for _, obj_id_, hyp_id_ in sorted(pairwise_det_, key=lambda x: x[0], reverse=True):
            if obj_id_ in chosen_ids_obj_ or hyp_id_ in chosen_ids_hyp_:
                continue
            dist_sum += iou_
            match_count +=1
            matches_[obj_id_] = hyp_id_
            chosen_ids_obj_.add(obj_id_)
            chosen_ids_hyp_.add(hyp_id_)

        for key, val in matches_.items():
            matches[key] = val
    MOTP = dist_sum/match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.56):
    """
    Calculate MOTP/MOTA
    """
    # print(obj, hyp)
    dist_sum = 0  
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0

    obj_count = 0
    matches = {}  


    for frame_obj, frame_hyp in zip(obj, hyp):
        chosen_ids_obj_ = set()
        chosen_ids_hyp_ = set()
        obj_dict_, hyp_dict_ = {frame_[0]: frame_[1:] for frame_ in frame_obj}, {frame_[0]: frame_[1:] for frame_ in frame_hyp}
        obj_count += len(obj_dict_)

 
        matches_ = {}
        for obj_id_, hyp_id_ in matches.items():
            if obj_id_ in obj_dict_ and hyp_id_ in hyp_dict_ and (iou_ := iou_score(obj_dict_[obj_id_], hyp_dict_[hyp_id_])) > threshold:
                dist_sum += iou_
                match_count +=1
                matches_[obj_id_] = hyp_id_
                chosen_ids_obj_.add(obj_id_)
                chosen_ids_hyp_.add(hyp_id_)
   
        pairwise_det_ = []
        for obj_id_, fr_obj_ in obj_dict_.items():
            if obj_id_ in chosen_ids_obj_:
                continue
            for hyp_id_, fr_hyp_ in hyp_dict_.items():
                if hyp_id_ in chosen_ids_hyp_:
                    continue
                iou_ = iou_score(fr_obj_, fr_hyp_)
                if iou_ > threshold:
                    pairwise_det_.append([iou_, obj_id_, hyp_id_])
        
        for _, obj_id_, hyp_id_ in sorted(pairwise_det_, key=lambda x: x[0], reverse=True):
            if obj_id_ in chosen_ids_obj_ or hyp_id_ in chosen_ids_hyp_:
                continue
            dist_sum += iou_
            match_count +=1
            matches_[obj_id_] = hyp_id_
            chosen_ids_obj_.add(obj_id_)
            chosen_ids_hyp_.add(hyp_id_)

        for obj_id_ in matches_:
            if obj_id_ in matches and matches_[obj_id_] != matches[obj_id_]:
                mismatch_error +=1
                chosen_ids_obj_.add(obj_id_)
                chosen_ids_hyp_.add(matches_[obj_id_])

        for key, val in matches_.items():
            matches[key] = val
  
        missed_count += len(obj_dict_) - len(chosen_ids_obj_)
        false_positive += len(hyp_dict_) - len(chosen_ids_hyp_)
    MOTP = dist_sum/match_count
    MOTA = 1 - (false_positive + mismatch_error + missed_count)/(obj_count)
    # raise BaseException((false_positive)/obj_count, (mismatch_error)/obj_count, (missed_count)/obj_count)
    return MOTP, MOTA
