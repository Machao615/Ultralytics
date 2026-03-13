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
import cv2
from pathlib import Path

from ultralytics.utils import LOGGER, colorstr, LINUX, ARM64
from ultralytics.utils.checks import check_requirements, check_version
from ultralytics.data.utils import check_det_dataset

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

def _prepare_calibration_data(args, cal_data_dir, imgsz):
    """Generates calibration data using image dataset referenced in args.data."""
    if not args.data:
        raise ValueError("Missing 'data' argument for D-Robotics INT8 calibration. E.g., data=coco8.yaml")
    
    os.makedirs(cal_data_dir, exist_ok=True)
    try:
        data = check_det_dataset(args.data)
        # Strictly use 'train' for calibration, as it provides better diversity
        train_path = data.get("train", "")
        if not train_path:
            raise ValueError(f"No 'train' split found in {args.data}. Calibration requires training data.")
    except Exception as e:
        raise ValueError(f"Could not parse data YAML {args.data} for calibration: {e}")
        
    if isinstance(train_path, list):
        train_path = train_path[0]
        
    img_dir = Path(train_path)
    if img_dir.is_file() and img_dir.suffix == ".txt":
        with open(img_dir, "r") as f:
            lines = [x.strip() for x in f.read().splitlines() if x.strip()]
            img_paths = [Path(x) for x in lines]
    else:
        img_paths = list(img_dir.rglob("*.*"))
        
    img_paths = [p for p in img_paths if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
    
    if not img_paths:
        raise ValueError(f"No images found for calibration in {train_path}")
        
    import random
    sample_num = min(args.batch if hasattr(args, 'batch') and args.batch > 1 else 20, len(img_paths))
    # Randomly select calibration images to ensure better representation
    img_paths = random.sample(img_paths, sample_num)
    
    width, height = (imgsz, imgsz) if isinstance(imgsz, int) else (imgsz[1], imgsz[0])
    
    LOGGER.info(f"Preparing {len(img_paths)} random calibration images from train split for BPU...")
    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        input_tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = cv2.resize(input_tensor, (width, height))
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        
        dst_path = os.path.join(cal_data_dir, f"cal_{idx}.rgbchw")
        input_tensor.tofile(dst_path)

def export_rdk(model, args, onnx_path=None):
    """Export Ultralytics YOLO model to D-Robotics BPU .bin format using hb_mapper."""
    prefix = colorstr("D-Robotics:")
    if ARM64:
        raise RuntimeError(f"{prefix} Export is only supported on x86_64 Linux with hb_mapper toolchain.")

    imgsz = args.imgsz
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)

    if onnx_path is None:
        onnx_path = Path(args.model).with_suffix(".onnx")
        from . import torch2onnx
        torch2onnx(model, torch.zeros(1, 3, *imgsz).to(next(model.parameters()).device), str(onnx_path), opset=11)
    
    onnx_path = Path(onnx_path).resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"{prefix} Intermediate ONNX file not found at {onnx_path}")

    save_dir = getattr(args, "save_dir", onnx_path.parent) or onnx_path.parent
    save_dir = Path(save_dir).resolve()
    ws_dir = save_dir / ".temporary_workspace"
    cal_data_dir = ws_dir / ".calibration_data"
    bpu_output_dir = ws_dir / "bpu_model_output"
    
    # Clean workspace if exists
    if ws_dir.exists():
        shutil.rmtree(ws_dir)
    ws_dir.mkdir(parents=True)
    bpu_output_dir.mkdir(parents=True)
    
    # Prepare calibration
    _prepare_calibration_data(args, str(cal_data_dir), imgsz)

    model_name = onnx_path.stem
    output_model_prefix = f"{model_name}_bayese_{imgsz[1]}x{imgsz[0]}_nv12"
    bin_path = onnx_path.with_suffix(".bin")
    
    yaml_content = f'''model_parameters:
  onnx_model: '{str(onnx_path)}'
  march: "bayes-e"
  layer_out_dump: False
  working_dir: '{str(bpu_output_dir)}'
  output_model_file_prefix: '{output_model_prefix}'
input_parameters:
  input_name: ""
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  norm_type: 'data_scale'
  scale_value: 0.003921568627451
calibration_parameters:
  cal_data_dir: '{str(cal_data_dir)}'
  cal_data_type: 'float32'
  calibration_type: 'default'
  optimization: set_Softmax_input_int8,set_Softmax_output_int8
compiler_parameters:
  jobs: 16
  compile_mode: 'latency'
  debug: true
  optimize_level: 'O3'
'''

    config_path = ws_dir / "hb_mapper_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    
    LOGGER.info(f"{prefix} Starting BPU compilation with hb_mapper...")
    original_cwd = os.getcwd()
    try:
        os.chdir(ws_dir)
        subprocess.run(["hb_mapper", "makertbin", "--config", "hb_mapper_config.yaml", "--model-type", "onnx"], check=True)
        
        # Find the output bin file
        compiled_bin = next(bpu_output_dir.rglob("*.bin"), None)
        if compiled_bin:
            shutil.copy(compiled_bin, bin_path)
            LOGGER.info(f"{prefix} Export success: {bin_path}")
        else:
            LOGGER.error(f"{prefix} Compilation finished but .bin file not found.")
            return None
    except Exception as e:
        LOGGER.error(f"{prefix} BPU compilation failed: {e}")
        return None
    finally:
        os.chdir(original_cwd)

    return str(bin_path)

def decode_rdk(outputs, imgsz, score_thres=0.25, nc=80):
    """Decodes D-Robotics BPU outputs into a single YOLO-format tensor."""
    strides = [8, 16, 32]
    all_preds = []
    
    safe_thres = max(min(score_thres, 1.0 - 1e-6), 1e-6)
    logit_thres = -np.log(1.0 / safe_thres - 1.0)
    
    # Infer DFL REG size from the first box output shape (e.g. if 64, REG=16)
    reg = outputs[1].shape[-1] // 4
    weights_static = np.arange(reg, dtype=np.float32)
    
    for i, stride in enumerate(strides):
        cls_feat = outputs[i * 2]      # (1, H, W, nc)
        box_feat = outputs[i * 2 + 1]  # (1, H, W, 4 * reg)
        
        max_logits = np.max(cls_feat[0], axis=-1)
        mask = max_logits >= logit_thres
        
        if not np.any(mask):
            continue
            
        valid_cls_logits = cls_feat[0][mask]
        valid_box_raw = box_feat[0][mask] # (N, 4 * reg)
        
        # --- DFL Decoding ---
        # Reshape to (N, 4, reg)
        box_reshaped = valid_box_raw.reshape(-1, 4, reg)
        # Apply Softmax over the 'reg' dimension. Prevent overflow by subtracting max.
        max_vals = np.max(box_reshaped, axis=-1, keepdims=True)
        exp_vals = np.exp(box_reshaped - max_vals)
        box_softmax = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        # Expected value (weighted sum) to get l, t, r, b
        ltrb = np.sum(box_softmax * weights_static, axis=-1) # (N, 4)
        
        h, w = cls_feat.shape[1:3]
        grid_y, grid_x = np.indices((h, w))
        valid_grid_x = grid_x[mask] + 0.5
        valid_grid_y = grid_y[mask] + 0.5
        
        # Calculate pixel coordinates
        x1 = (valid_grid_x - ltrb[:, 0]) * stride
        y1 = (valid_grid_y - ltrb[:, 1]) * stride
        x2 = (valid_grid_x + ltrb[:, 2]) * stride
        y2 = (valid_grid_y + ltrb[:, 3]) * stride

        # Convert to cx, cy, w, h (expected by Ultralytics NMS)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w_box = x2 - x1
        h_box = y2 - y1

        valid_cls_scores = 1.0 / (1.0 + np.exp(-valid_cls_logits))

        layer_pred = np.concatenate([
            cx[:, None], cy[:, None], w_box[:, None], h_box[:, None],
            valid_cls_scores
        ], axis=-1)        
        all_preds.append(layer_pred)
        
    if not all_preds:
        return torch.zeros((1, 4 + nc, 0))
        
    final_pred = np.concatenate(all_preds, axis=0)
    return torch.from_numpy(final_pred).permute(1, 0).unsqueeze(0).float()
