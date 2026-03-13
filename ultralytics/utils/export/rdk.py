# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
D-Robotics RDK X5 BPU Export logic.
"""

import os
import shutil
import subprocess
import yaml
import numpy as np
import torch
from pathlib import Path

from ultralytics.utils import LOGGER, colorstr, LINUX, ARM64
from ultralytics.utils.checks import check_requirements, check_version

def bpu_detect_forward(self, x):
    """YOLO Detect Head Modified for D-Robotics BPU."""
    res = []
    for i in range(self.nl):
        res.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())  # cls
        res.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())  # bbox
    return res

def bpu_v10_detect_forward(self, x):
    """YOLOv10 Detect Head Modified for D-Robotics BPU."""
    res = []
    for i in range(self.nl):
        res.append(self.one2one_cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())  # cls
        res.append(self.one2one_cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())  # bbox
    return res

def bpu_segment_forward(self, x):
    """YOLO Segment Head Modified for D-Robotics BPU."""
    res = []
    for i in range(self.nl):
        res.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())  # cls
        res.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())  # bbox
        res.append(self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous())  # proto weights
    res.append(self.proto(x[0]).permute(0, 2, 3, 1).contiguous())       # proto mask
    return res

def bpu_pose_forward(self, x):
    """YOLO Pose Head Modified for D-Robotics BPU."""
    res = []
    for i in range(self.nl):
        res.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())  # cls
        res.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())  # bbox
        res.append(self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous())  # kpts
    return res

def bpu_obb_forward(self, x):
    """YOLO OBB Head Modified for D-Robotics BPU."""
    res = []
    for i in range(self.nl):
        res.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())  # cls
        res.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())  # bbox
        res.append(self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous())  # theta logits
    return res

def bpu_classify_forward(self, x):
    """YOLO Classify Head Modified for BPU. Replaces Linear with Conv2d 1x1."""
    if x.ndim == 4:
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        if hasattr(self, 'conv_linear'):
            x = self.conv_linear(x)
        else:
            x = torch.flatten(x, 1)
            x = self.linear(x)
            return x
        return x.flatten(1)
    return self.linear(torch.flatten(self.pool(self.conv(x)), 1))

def apply_rdk_patches(model):
    """Applies BPU-specific monkey patches to the model heads."""
    from ultralytics.nn.modules import Detect, Classify, Segment, Pose, OBB
    from ultralytics.nn.modules.head import v10Detect
    
    for m in model.modules():
        if isinstance(m, Detect) and not isinstance(m, (Segment, Pose, OBB)):
            m.forward = bpu_detect_forward.__get__(m, Detect)
            LOGGER.info(f"{colorstr('D-Robotics:')} Patched Detect head for BPU.")
        elif isinstance(m, v10Detect):
            m.forward = bpu_v10_detect_forward.__get__(m, v10Detect)
            LOGGER.info(f"{colorstr('D-Robotics:')} Patched YOLOv10 Detect head for BPU.")
        elif isinstance(m, Segment):
            m.forward = bpu_segment_forward.__get__(m, Segment)
            LOGGER.info(f"{colorstr('D-Robotics:')} Patched Segment head for BPU.")
        elif isinstance(m, Pose):
            m.forward = bpu_pose_forward.__get__(m, Pose)
            LOGGER.info(f"{colorstr('D-Robotics:')} Patched Pose head for BPU.")
        elif isinstance(m, OBB):
            m.forward = bpu_obb_forward.__get__(m, OBB)
            LOGGER.info(f"{colorstr('D-Robotics:')} Patched OBB head for BPU.")
        elif isinstance(m, Classify):
            # Convert Linear to Conv2d 1x1 for Classify
            in_features = m.linear.in_features
            out_features = m.linear.out_features
            m.conv_linear = torch.nn.Conv2d(in_features, out_features, 1)
            m.conv_linear.weight.data = m.linear.weight.data.view(out_features, in_features, 1, 1)
            m.conv_linear.bias.data = m.linear.bias.data
            m.forward = bpu_classify_forward.__get__(m, Classify)
            LOGGER.info(f"{colorstr('D-Robotics:')} Patched Classify head for BPU.")

def export_rdk(model, args, onnx_path=None):
    """Export Ultralytics YOLO model to D-Robotics BPU .bin format using hb_mapper."""
    prefix = colorstr("D-Robotics:")
    if ARM64:
        raise RuntimeError(f"{prefix} Export is only supported on x86_64 Linux with hb_mapper toolchain.")

    if onnx_path is None:
        # Fallback if no onnx_path provided, though preferred to be passed from Exporter
        imgsz = args.imgsz
        if isinstance(imgsz, int):
            imgsz = (imgsz, imgsz)
        
        onnx_path = Path(args.model).with_suffix(".onnx")
        from . import torch2onnx
        torch2onnx(model, torch.zeros(1, 3, *imgsz).to(next(model.parameters()).device), str(onnx_path), opset=11)
    
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"{prefix} Intermediate ONNX file not found at {onnx_path}")

    save_dir = getattr(args, "save_dir", onnx_path.parent) or onnx_path.parent
    save_dir = Path(save_dir)
    config_path = save_dir / "hb_mapper_config.yaml"
    bin_path = onnx_path.with_suffix(".bin")
    
    config = {
        "model_parameters": {
            "onnx_model": str(onnx_path),
            "output_model_file_prefix": onnx_path.stem,
            "working_dir": str(save_dir / "hb_work_dir"),
        },
        "input_parameters": {
            "input_name": "images",
            "input_type_rt": "nv12",
            "input_space_and_range": "BGR",
            "input_layout_rt": "NHWC",
        },
        "calibration_parameters": {
            "cal_data_dir": str(args.data or "cal_images"),
            "quant_config_strategy": "weights_equalization",
        },
        "compiler_parameters": {
            "optimize_level": "O3",
        }
    }
    
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    LOGGER.info(f"{prefix} Starting BPU compilation with hb_mapper...")
    try:
        subprocess.run(["hb_mapper", "makertbin", "--config", str(config_path)], check=True)
        LOGGER.info(f"{prefix} Export success: {bin_path}")
    except Exception as e:
        LOGGER.error(f"{prefix} BPU compilation failed: {e}")
        return None

    return str(bin_path)

def decode_rdk(outputs, imgsz, score_thres=0.25, nc=80):
    """Decodes D-Robotics BPU outputs into a single YOLO-format tensor."""
    strides = [8, 16, 32]
    all_preds = []
    
    safe_thres = max(min(score_thres, 1.0 - 1e-6), 1e-6)
    logit_thres = -np.log(1.0 / safe_thres - 1.0)
    
    for i, stride in enumerate(strides):
        cls_feat = outputs[i * 2]      # (1, H, W, nc)
        box_feat = outputs[i * 2 + 1]  # (1, H, W, 4)
        
        max_logits = np.max(cls_feat[0], axis=-1)
        mask = max_logits >= logit_thres
        
        if not np.any(mask):
            continue
            
        valid_cls_logits = cls_feat[0][mask]
        valid_box_raw = box_feat[0][mask]
        
        h, w = cls_feat.shape[1:3]
        grid_y, grid_x = np.indices((h, w))
        valid_grid_x = grid_x[mask] + 0.5
        valid_grid_y = grid_y[mask] + 0.5
        
        x1 = (valid_grid_x - valid_box_raw[:, 0]) * stride
        y1 = (valid_grid_y - valid_box_raw[:, 1]) * stride
        x2 = (valid_grid_x + valid_box_raw[:, 2]) * stride
        y2 = (valid_grid_y + valid_box_raw[:, 3]) * stride
        
        valid_cls_scores = 1.0 / (1.0 + np.exp(-valid_cls_logits))
        
        layer_pred = np.concatenate([
            x1[:, None], y1[:, None], x2[:, None], y2[:, None], 
            valid_cls_scores
        ], axis=-1)
        
        all_preds.append(layer_pred)
        
    if not all_preds:
        return torch.zeros((1, 4 + nc, 0))
        
    final_pred = np.concatenate(all_preds, axis=0)
    return torch.from_numpy(final_pred).permute(1, 0).unsqueeze(0).float()
