# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/8/24 14:58
@desc: 
"""
from sys import platform

import torch
import torchvision
from PyInstaller.compat import check_requirements
from jupyter_server.utils import check_version
from onnxruntime.transformers.large_model_exporter import export_onnx
from ultralytics.utils import colorstr, LOGGER


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


import torch
import torch.nn.functional as F


def roi_pooling(feature_map, rois, output_size):
    """
    参数:
    - feature_map: (N, C, H, W) 特征图张量
    - rois: (num_rois, 5) RoI张量，每个RoI由 (batch_index, x1, y1, x2, y2) 表示
    - output_size: (output_height, output_width) 输出特征图的大小
    返回:
    - pooled_features: (num_rois, C, output_height, output_width) 池化后的特征图
    """
    num_rois = rois.size(0)
    num_channels = feature_map.size(1)
    output_height, output_width = output_size

    # 初始化池化后的输出张量
    pooled_features = torch.zeros((num_rois, num_channels, output_height, output_width))

    for i in range(num_rois):
        roi = rois[i]
        batch_index = int(roi[0].item())
        x1, y1, x2, y2 = roi[1:].int()

        roi_width = max(x2 - x1 + 1, 1)
        roi_height = max(y2 - y1 + 1, 1)

        # 计算每个池化区域的大小
        bin_size_w = roi_width / output_width
        bin_size_h = roi_height / output_height

        for ph in range(output_height):
            for pw in range(output_width):
                # 计算当前bin的边界
                h_start = int(y1 + ph * bin_size_h)
                h_end = int(y1 + (ph + 1) * bin_size_h)
                w_start = int(x1 + pw * bin_size_w)
                w_end = int(x1 + (pw + 1) * bin_size_w)

                # 边界限制在特征图范围内
                h_start = min(max(h_start, 0), feature_map.size(2))
                h_end = min(max(h_end, 0), feature_map.size(2))
                w_start = min(max(w_start, 0), feature_map.size(3))
                w_end = min(max(w_end, 0), feature_map.size(3))

                # 如果区域大小为0，则跳过
                if h_end <= h_start or w_end <= w_start:
                    continue

                # 从特征图中提取当前bin的子区域
                roi_bin = feature_map[batch_index, :, h_start:h_end, w_start:w_end]

                # 在该区域上执行最大池化
                pooled_features[i, :, ph, pw] = torch.max(roi_bin.view(num_channels, -1), dim=1)[0]

    return pooled_features


# 示例使用
feature_map = torch.randn(1, 256, 50, 50)  # 例如 (1, 256, 50, 50)
rois = torch.tensor([[0, 10, 10, 30, 30],  # 第一张图中的一个RoI
                     [0, 20, 20, 40, 40]], dtype=torch.float32)  # 第一张图中的另一个RoI

output_size = (7, 7)  # 输出的特征图大小
pooled_features = roi_pooling(feature_map, rois, output_size)

print(pooled_features.shape)  # 输出: torch.Size([2, 256, 7, 7])
def cross_entropy(predict, target):
    """
    :param predict: [bs, num_classes]
    :param target: [bs, ]   值为[0, num_classes-1]
    :return:
    """
    probs = torch.exp(predict) / torch.exp(predict).sum(dim=-1, keepdims=True)
    batch_size = predict.shape[0]
    target_one_hot = torch.zeros_like(probs)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    loss = -torch.sum(target_one_hot * torch.log(probs + 1e-10)) / batch_size
    return loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class DFLLoss(nn.Module):
    def __init__(self, reg_max=16):
        super(DFLLoss, self).__init__()
        self.reg_max = reg_max  # Number of bins used for the distribution

    def forward(self, pred, target):
        # Reshape the predictions to (batch_size, 4, reg_max + 1, H, W)
        pred = pred.view(-1, 4, self.reg_max + 1, *pred.shape[2:])

        # Split into x, y, w, h predictions
        pred_x = pred[:, 0, :, :, :]
        pred_y = pred[:, 1, :, :, :]
        pred_w = pred[:, 2, :, :, :]
        pred_h = pred[:, 3, :, :, :]

        # Split target into x, y, w, h
        target_x = target[:, 0, :, :]
        target_y = target[:, 1, :, :]
        target_w = target[:, 2, :, :]
        target_h = target[:, 3, :, :]

        # Compute the loss for each coordinate
        loss_x = self._df_loss(pred_x, target_x)
        loss_y = self._df_loss(pred_y, target_y)
        loss_w = self._df_loss(pred_w, target_w)
        loss_h = self._df_loss(pred_h, target_h)

        # Total loss is the sum of all coordinate losses
        loss = loss_x + loss_y + loss_w + loss_h
        return loss

    def _df_loss(self, pred, target):
        # Convert target into two indices (lower and upper bounds) and weights (linear interpolation)
        t = target.long()
        w = target - t  # fractional part (interpolation weight)

        # Define lower and upper bounds based on target
        t0 = torch.clamp(t, 0, self.reg_max)
        t1 = torch.clamp(t + 1, 0, self.reg_max)

        # Gather the predicted logits for the lower and upper bins
        p0 = pred.gather(1, t0.unsqueeze(1))  # (batch_size, 1, H, W)
        p1 = pred.gather(1, t1.unsqueeze(1))  # (batch_size, 1, H, W)

        # Compute the loss as a weighted combination of the two bins
        loss = F.binary_cross_entropy_with_logits(p0, (1 - w).unsqueeze(1), reduction='none') + \
               F.binary_cross_entropy_with_logits(p1, w.unsqueeze(1), reduction='none')

        return loss.mean()


# Example usage:
# Assume we have a predicted output and a ground truth target tensor
# pred: (batch_size, 4 * (reg_max + 1), H, W), target: (batch_size, 4, H, W)
batch_size = 2
H, W = 80, 80
reg_max = 16
pred = torch.randn(batch_size, 4 * (reg_max + 1), H, W)
target = torch.rand(batch_size, 4, H, W) * reg_max  # Random target values in the range [0, reg_max]

# Initialize DFL loss
dfl_loss = DFLLoss(reg_max=reg_max)

# Compute the loss
loss_value = dfl_loss(pred, target)
print("DFL Loss:", loss_value.item())


class quantization():
    def __init__(self, num_bits=8):
        self.num_bits = num_bits

    def export_engine(model, im, file, half, dynamic, simplify, workspace=4, verbose=False,
                      prefix=colorstr("TensorRT:")):
        """
        Export a YOLOv5 model to TensorRT engine format, requiring GPU and TensorRT>=7.0.0.

        Args:
            model (torch.nn.Module): YOLOv5 model to be exported.
            im (torch.Tensor): Input tensor of shape (B, C, H, W).
            file (pathlib.Path): Path to save the exported model.
            half (bool): Set to True to export with FP16 precision.
            dynamic (bool): Set to True to enable dynamic input shapes.
            simplify (bool): Set to True to simplify the model during export.
            workspace (int): Workspace size in GB (default is 4).
            verbose (bool): Set to True for verbose logging output.
            prefix (str): Log message prefix.

        Returns:
            (pathlib.Path, None): Tuple containing the path to the exported model and None.

        Raises:
            AssertionError: If executed on CPU instead of GPU.
            RuntimeError: If there is a failure in parsing the ONNX file.

        Example:
            ```python
            from ultralytics import YOLOv5
            import torch
            from pathlib import Path

            model = YOLOv5('yolov5s.pt')  # Load a pre-trained YOLOv5 model
            input_tensor = torch.randn(1, 3, 640, 640).cuda()  # example input tensor on GPU
            export_path = Path('yolov5s.engine')  # export destination

            export_engine(model.model, input_tensor, export_path, half=True, dynamic=True, simplify=True, workspace=8, verbose=True)
            ```
        """
        assert im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. `python export.py --device 0`"
        try:
            import tensorrt as trt
        except Exception:
            if platform.system() == "Linux":
                check_requirements("nvidia-tensorrt", cmds="-U --index-url https://pypi.ngc.nvidia.com")
            import tensorrt as trt

        if trt.__version__[0] == "7":  # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
            grid = model.model[-1].anchor_grid
            model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
            export_onnx(model, im, file, 12, dynamic, simplify)  # opset 12
            model.model[-1].anchor_grid = grid
        else:  # TensorRT >= 8
            check_version(trt.__version__, "8.0.0", hard=True)  # require tensorrt>=8.0.0
            export_onnx(model, im, file, 12, dynamic, simplify)  # opset 12
        onnx = file.with_suffix(".onnx")

        LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
        is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # is TensorRT >= 10
        assert onnx.exists(), f"failed to export ONNX file: {onnx}"
        f = file.with_suffix(".engine")  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        if is_trt10:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)
        else:  # TensorRT versions 7, 8
            config.max_workspace_size = workspace * 1 << 30
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f"failed to load ONNX file: {onnx}")

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

        if dynamic:
            if im.shape[0] <= 1:
                LOGGER.warning(f"{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument")
            profile = builder.create_optimization_profile()
            for inp in inputs:
                profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
            config.add_optimization_profile(profile)

        LOGGER.info(f"{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}")
        if builder.platform_has_fast_fp16 and half:
            config.set_flag(trt.BuilderFlag.FP16)

        build = builder.build_serialized_network if is_trt10 else builder.build_engine
        with build(network, config) as engine, open(f, "wb") as t:
            t.write(engine if is_trt10 else engine.serialize())
        return f, None
