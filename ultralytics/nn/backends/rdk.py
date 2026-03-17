from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_rdk_requirements
from ultralytics.utils.export.rdk import decode_rdk

from .base import BaseBackend


class RDKBackend(BaseBackend):
    """D-Robotics RDK X5 inference backend for .bin models compiled by hb_mapper."""

    def load_model(self, weight: str | Path) -> None:
        """Load an RDK X5 model directory or compiled .bin file."""
        check_rdk_requirements()

        import hbm_runtime

        w = Path(weight)
        if not w.is_file():
            w = next(w.rglob("*.bin"))

        LOGGER.info(f"Loading {w} for D-Robotics inference...")
        self.model = hbm_runtime.HB_HBMRuntime(str(w))
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.nhwc = True
        if len(self.input_names) != 1:
            raise RuntimeError(
                f"RDK X5 runtime expects a single NV12 input tensor, but model '{w.name}' exposes "
                f"{len(self.input_names)} inputs."
            )

        metadata_file = w.parent / "metadata.yaml"
        if metadata_file.exists():
            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Run inference on an RDK model."""
        im = im.cpu().numpy()
        if im.dtype != np.uint8:
            im = (im * 255.0).clip(0, 255).astype(np.uint8)

        input_shape = getattr(self.model, "input_shapes", {}).get(self.model_name, {}).get(self.input_names[0])
        single_nv12s = []

        for img in im:
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            h, w = bgr_img.shape[:2]
            yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((h * w * 3 // 2,))
            y_arr = yuv420p[: h * w].reshape((h, w, 1))
            u_arr = yuv420p[h * w : h * w + h * w // 4].reshape((h // 2, w // 2))
            v_arr = yuv420p[h * w + h * w // 4 :].reshape((h // 2, w // 2))
            uv_arr = np.stack((u_arr, v_arr), axis=-1)

            nv12_flat = np.empty((h * w * 3 // 2,), dtype=np.uint8)
            nv12_flat[: h * w] = y_arr.reshape(-1)
            nv12_flat[h * w :] = uv_arr.reshape(-1)
            if input_shape is not None and np.prod(input_shape) == len(nv12_flat):
                single_nv12s.append(nv12_flat.reshape(input_shape[1:]))
            else:
                single_nv12s.append(nv12_flat.reshape(int(h * 1.5), w, 1))

        input_tensor = {self.model_name: {self.input_names[0]: np.stack(single_nv12s, axis=0)}}

        outputs = self.model.run(input_tensor)[self.model_name]
        y = [outputs[name] for name in self.output_names]

        if self.task == "classify":
            return y[0]

        return decode_rdk(y, self.imgsz, nc=len(self.names))
