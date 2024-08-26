# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/8/24 14:58
@desc: 
"""
import torch
import torchvision

def nms(points, scores, threshold=0.5):
    if len(scores) == 0:
        return torch.zeros((0,), dtype=torch.int32)

    x1, y1, x2, y2 = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    _, inds = torch.sort(scores, descending=True)
    kepp = []
    while inds.numel() > 0:
        point = points[inds[0]]
        kepp.append(point.item())

        if len(inds) == 1:
            break

        xx1 = torch.maximum(x1[inds[1:]], x1[inds[0]])
        yy1 = torch.maximum(y1[inds[1:]], y1[inds[0]])
        xx2 = torch.minimum(x2[inds[1:]], x2[inds[0]])
        yy2 = torch.minimum(y2[inds[1:]], y2[inds[0]])

        width = torch.clamp(xx2 - xx1, min=0)
        height = torch.clamp(yy2 - yy1, min=0)
        overlap = width * height

        iou = overlap / (areas[inds[1:]] + areas[inds[0]] - overlap)

        inds = inds[1:][iou <= threshold]
    return torch.tensor(kepp, dtype=torch.int32)


def iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    width = (x2 - x1).clamp(min=0)
    heigth = (y2 - y1).clamp(min=0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = width * heigth / (box1_area + box2_area - width * heigth)
    return iou