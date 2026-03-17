# RDK X5 Modifications Summary

This document summarizes the main changes currently present in `ultralytics_x5` compared with the original workspace copy in `ultralytics`.

## Scope

The fork is no longer a pure upstream mirror. It contains:

- RDK X5 export and runtime support
- Runtime/backend integration changes
- Training/validation/predict path adjustments across several YOLO tasks
- A small set of documentation and test additions related to RDK

This summary focuses on source changes under `ultralytics_x5` and excludes `.git` and `__pycache__` files.

## Core RDK X5 Integration

Files:

- `ultralytics/engine/exporter.py`
- `ultralytics/utils/export/rdk.py`
- `ultralytics/utils/checks.py`
- `ultralytics/nn/autobackend.py`
- `ultralytics/nn/backends/__init__.py`
- `ultralytics/nn/backends/rdk.py`

What changed:

- Added a new export format entry: `format="rdk"` with `*_rdk_model/` output directories.
- Added RDK-specific export flow that:
  - patches model heads for BPU-friendly ONNX layout
  - exports an intermediate ONNX model
  - generates X5-oriented `hb_mapper` config
  - compiles the final `.bin` model and bundles `metadata.yaml`
- Added environment checks for:
  - x86_64 Linux export toolchain installation
  - ARM64 RDK board runtime availability
- Reworked inference integration so RDK is handled as a modular backend, closer to the upstream RKNN style.
- Current runtime path is intentionally narrowed to X5-only behavior:
  - single NV12 input tensor
  - no S100 dual-input Y/UV runtime path

## Runtime / Backend Refactor

Files:

- `ultralytics/nn/autobackend.py`
- `ultralytics/nn/backends/*`

What changed:

- The original `autobackend.py` in this fork had diverged heavily from upstream.
- It has now been rebased back toward the current upstream structure.
- The fork adds only a thin RDK registration layer in `autobackend.py`.
- A new `ultralytics/nn/backends/` package was introduced to match upstream backend organization and make future syncs easier.

## Task Path Changes

Files:

- `ultralytics/engine/predictor.py`
- `ultralytics/engine/trainer.py`
- `ultralytics/engine/tuner.py`
- `ultralytics/engine/validator.py`
- `ultralytics/models/yolo/classify/{predict,train,val}.py`
- `ultralytics/models/yolo/detect/{train,val}.py`
- `ultralytics/models/yolo/obb/{predict,train,val}.py`
- `ultralytics/models/yolo/pose/{predict,train,val}.py`
- `ultralytics/models/yolo/segment/{predict,train,val}.py`
- `ultralytics/models/yolo/world/{train,train_world}.py`
- `ultralytics/models/yolo/yoloe/{train,train_seg}.py`
- `ultralytics/models/fastsam/{predict,val}.py`
- `ultralytics/models/sam/predict.py`

What changed:

- These files differ from the original repo and likely contain compatibility work for the custom export/runtime path.
- Several of them appear related to:
  - end-to-end model behavior
  - postprocess flow alignment
  - training/validation compatibility for custom heads or export targets

## Utility and Data Changes

Files:

- `ultralytics/__init__.py`
- `ultralytics/data/converter.py`
- `ultralytics/hub/auth.py`
- `ultralytics/solutions/solutions.py`

What changed:

- These files differ from the original repo and appear to include project-specific compatibility or behavior changes beyond pure RDK runtime support.

## Documentation Changes

Files:

- `docs/en/guides/rdk-deployment.md`
- `docs/en/integrations/tensorrt.md`
- `docs/en/reference/data/converter.md`
- `docs/overrides/main.html`

What changed:

- Added or updated RDK deployment documentation.
- Adjusted docs/custom overrides to reflect fork-specific functionality.

## Tests Added

Files:

- `tests/test_rdk.py`

What changed:

- Added a small RDK-focused regression test file for requirement checks and unsupported-platform warnings.

## Current Status Notes

- `autobackend.py` is now much closer to upstream than it was before this cleanup.
- The biggest fork-specific logic for RDK is concentrated in:
  - `ultralytics/utils/export/rdk.py`
  - `ultralytics/nn/backends/rdk.py`
  - `ultralytics/utils/checks.py`
- If you want future syncs to stay manageable, these three areas should remain the primary extension points.
