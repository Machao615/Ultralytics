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
- Fixed a runtime circular import issue by making calibration dataset loading and runtime decode imports lazy.
- Added lightweight P0/P1 runtime optimization work in the X5 backend:
  - optional stage-level profiling for preprocess / infer / decode
  - direct RGB-to-I420 conversion without an intermediate BGR pass
  - reusable NV12 scratch buffer allocation
  - batch=1 fast path without `np.stack()`

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

## Chinese Version

本文档用于说明 `ultralytics_x5` 相比工作区中的原始 `ultralytics` 做了哪些主要修改。

### 范围

当前 fork 已不再是纯上游镜像，主要包含：

- RDK X5 导出与运行时支持
- 推理后端结构调整
- 多个 YOLO 任务在 train / val / predict 链路上的兼容性修改
- 少量和 RDK 相关的文档与测试补充

### RDK X5 核心集成

涉及文件：

- `ultralytics/engine/exporter.py`
- `ultralytics/utils/export/rdk.py`
- `ultralytics/utils/checks.py`
- `ultralytics/nn/autobackend.py`
- `ultralytics/nn/backends/__init__.py`
- `ultralytics/nn/backends/rdk.py`

主要改动：

- 新增 `format="rdk"` 导出格式，输出目录为 `*_rdk_model/`
- 新增 RDK 导出流程：
  - 对检测头做 BPU 友好的导出补丁
  - 先导出中间 ONNX
  - 生成面向 X5 的 `hb_mapper` 配置
  - 编译 `.bin` 并同时保存 `metadata.yaml`
- 新增 RDK 环境检查：
  - x86_64 Linux 上的导出工具链检查
  - ARM64 板端运行时检查
- 运行时已明确收口为 X5-only：
  - 仅支持单输入 NV12
  - 不再支持 S100 的双输入 Y / UV 模式
- 修复了一个运行时循环导入问题：
  - 将校准数据集加载改为延迟导入
  - 将 `decode_rdk` 改为运行时延迟导入
- 已完成一轮 P0 / P1 优化：
  - 增加 preprocess / infer / decode 分阶段 profiling
  - 用 RGB 直接转 I420，去掉中间 BGR 转换
  - 复用 NV12 临时 buffer，减少重复分配
  - 针对 batch=1 增加不走 `np.stack()` 的快路径

### 运行时 / 后端结构调整

涉及文件：

- `ultralytics/nn/autobackend.py`
- `ultralytics/nn/backends/*`

主要改动：

- 之前 fork 中的 `autobackend.py` 与上游差异过大
- 现在已回收成更接近上游的结构
- `autobackend.py` 只保留一层很薄的 RDK 接线
- 新增 `ultralytics/nn/backends/` 目录，使后续同步主线更容易

### 任务链路差异

涉及文件：

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

这些文件与原始仓库存在差异，主要和以下内容相关：

- end2end 模型行为兼容
- 自定义导出目标后的 postprocess 对齐
- 自定义头部或运行时路径下的训练 / 验证兼容

### 其他工具与数据层改动

涉及文件：

- `ultralytics/__init__.py`
- `ultralytics/data/converter.py`
- `ultralytics/hub/auth.py`
- `ultralytics/solutions/solutions.py`

这些改动不完全属于 RDK 运行时本身，也包含项目侧兼容性调整。

### 文档与测试

涉及文件：

- `docs/en/guides/rdk-deployment.md`
- `docs/en/integrations/tensorrt.md`
- `docs/en/reference/data/converter.md`
- `docs/overrides/main.html`
- `tests/test_rdk.py`

主要改动：

- 增加或更新了 RDK 部署文档
- 增加了少量 RDK 回归测试

### 当前建议

- 目前 RDK 相关的主要扩展点集中在：
  - `ultralytics/utils/export/rdk.py`
  - `ultralytics/nn/backends/rdk.py`
  - `ultralytics/utils/checks.py`
- 如果后续还要持续同步上游，建议继续把 RDK 逻辑收敛在这三处附近，而不要再次把大量实现塞回 `autobackend.py`
## Recent Update: RDK RGB Export Option

- Added an RDK export argument `rdk_input_type`.
- Default value remains `nv12` to preserve current X5 export behavior.
- When `rdk_input_type=rgb` is passed during export, the generated `hb_mapper` YAML now writes `input_type_rt: 'rgb'`.
- The generated output model prefix now also reflects the selected runtime input type.

## 最近更新：RDK RGB 导出参数

- 新增 RDK 导出参数 `rdk_input_type`。
- 默认值仍为 `nv12`，保持现有 X5 导出行为不变。
- 导出时传入 `rdk_input_type=rgb` 后，生成的 `hb_mapper` YAML 会写入 `input_type_rt: 'rgb'`。
- 生成的输出模型前缀也会随运行时输入类型一并变化。
