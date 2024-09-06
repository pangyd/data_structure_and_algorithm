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


def crossentropy(predict, target):
    softmax = torch.exp(predict) / torch.exp(predict).sum(dim=-1, keepdims=True)
    probs = softmax(predict)
    log_probs = torch.log(probs)
    target_log_probs = log_probs[range(target.shape[0]), target]
    loss = -target_log_probs.mean()
    return loss


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
