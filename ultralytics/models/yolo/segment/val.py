# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# SACR species-tree validation version
#
# 作用：
#   用于“主分支 tree-only + 辅助 species head”的 SACR 模型验证。
#
# 为什么要改：
#   你的训练逻辑是：
#       - 主 YOLO 分支保持 tree-only 分割；
#       - 辅助 species 分支学习 tp1/tp2；
#       - species 特征只用于修正 mask coefficients。
#
#   因此官方 val 阶段不能再按 tp1/tp2 评价主分支分类，
#   否则 class=1 的 GT 会被当成类别错误，导致 P/R/mAP 虚低。
#
# 这个文件做的事情：
#   1. GT 类别统一折叠为 tree=0；
#   2. prediction 类别统一折叠为 tree=0；
#   3. mask 处理、box/mask IoU、mAP 计算仍沿用官方流程；
#   4. 树种分类精度不要看官方 val，要用你自己的 matched-tree species 评价脚本。

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, mask_iou


class SegmentationValidator(DetectionValidator):
    """Tree-only validator for SACR species-guided segmentation models."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.process = None
        self.args.task = "segment"
        self.metrics = SegmentMetrics()

    @staticmethod
    def _force_tree_cls_tensor(x: torch.Tensor) -> torch.Tensor:
        """Fold any class tensor to tree class 0, preserving shape/device/dtype."""
        if x is None:
            return x
        return torch.zeros_like(x)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch and fold GT classes to tree=0 for validation."""
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].float()

        # SACR tree-only validation:
        # keep original masks/boxes, but fold tp1/tp2 GT classes into one tree class.
        if "cls" in batch and batch["cls"] is not None:
            batch["cls"] = self._force_tree_cls_tensor(batch["cls"])

        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics and select mask processing function."""
        super().init_metrics(model)
        if self.args.save_json:
            check_requirements("faster-coco-eval>=1.6.7")

        # More accurate vs faster
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask

        # 显示名改成 tree，避免官方 val 表里继续显示 tp1/tp2。
        try:
            self.names = {0: "tree"}
        except Exception:
            pass

    def get_desc(self) -> str:
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Post-process YOLO predictions and force predicted classes to tree=0."""
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        preds = super().postprocess(preds[0])
        imgsz = [4 * x for x in proto.shape[2:]]  # get image size from proto

        for i, pred in enumerate(preds):
            # Fold predicted classes to tree=0.
            # This prevents official validator from treating tp1/tp2 as main-task classes.
            if "cls" in pred and pred["cls"] is not None and len(pred["cls"]):
                pred["cls"] = torch.zeros_like(pred["cls"])

            coefficient = pred.pop("extra")
            pred["masks"] = (
                self.process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
                if coefficient.shape[0]
                else torch.zeros(
                    (0, *(imgsz if self.process is ops.process_mask_native else proto.shape[2:])),
                    dtype=torch.uint8,
                    device=pred["bboxes"].device,
                )
            )

        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare one image's GT masks and fold GT classes to tree=0."""
        prepared_batch = super()._prepare_batch(si, batch)

        if "cls" in prepared_batch and prepared_batch["cls"] is not None:
            prepared_batch["cls"] = self._force_tree_cls_tensor(prepared_batch["cls"])

        nl = prepared_batch["cls"].shape[0]
        if self.args.overlap_mask:
            masks = batch["masks"][si]
            index = torch.arange(1, nl + 1, device=masks.device).view(nl, 1, 1)
            masks = (masks == index).float()
        else:
            masks = batch["masks"][batch["batch_idx"] == si]

        if nl:
            mask_size = [s if self.process is ops.process_mask_native else s // 4 for s in prepared_batch["imgsz"]]
            if masks.shape[1:] != mask_size:
                masks = F.interpolate(masks[None], mask_size, mode="bilinear", align_corners=False)[0]
            masks = masks.gt_(0.5)

        prepared_batch["masks"] = masks
        return prepared_batch

    def gather_stats(self) -> None:
        """Gather stats from all GPUs."""
        super().gather_stats()
        self._gather_image_metrics(self.metrics.seg)

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Compute correct prediction matrix for boxes and masks under tree-only class matching."""
        # Double safety: force both predicted and GT classes to tree=0 before matching.
        if "cls" in preds and preds["cls"] is not None:
            preds["cls"] = self._force_tree_cls_tensor(preds["cls"])
        if "cls" in batch and batch["cls"] is not None:
            batch["cls"] = self._force_tree_cls_tensor(batch["cls"])

        tp = super()._process_batch(preds, batch)

        gt_cls = batch["cls"]
        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp_m = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        else:
            iou = mask_iou(batch["masks"].flatten(1), preds["masks"].flatten(1).float())
            tp_m = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()

        tp.update({"tp_m": tp_m})
        return tp

    def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
        """Plot batch predictions with masks and bounding boxes."""
        for p in preds:
            masks = p["masks"]
            if masks.shape[0] > self.args.max_det:
                LOGGER.warning(f"Limiting validation plots to 'max_det={self.args.max_det}' items.")
            p["masks"] = torch.as_tensor(masks[: self.args.max_det], dtype=torch.uint8).cpu()
            if "cls" in p and p["cls"] is not None:
                p["cls"] = self._force_tree_cls_tensor(p["cls"])
        super().plot_predictions(batch, preds, ni, max_det=self.args.max_det)

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO detections to txt file."""
        from ultralytics.engine.results import Results

        if "cls" in predn and predn["cls"] is not None:
            predn["cls"] = self._force_tree_cls_tensor(predn["cls"])

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            masks=torch.as_tensor(predn["masks"], dtype=torch.uint8),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Save one JSON result for COCO evaluation."""
        if "cls" in predn and predn["cls"] is not None:
            predn["cls"] = self._force_tree_cls_tensor(predn["cls"])

        def to_string(counts: list[int]) -> str:
            result = []
            for i in range(len(counts)):
                x = int(counts[i])
                if i > 2:
                    x -= int(counts[i - 2])
                while True:
                    c = x & 0x1F
                    x >>= 5
                    more = (x != -1) if (c & 0x10) else (x != 0)
                    if more:
                        c |= 0x20
                    c += 48
                    result.append(chr(c))
                    if not more:
                        break
            return "".join(result)

        def multi_encode(pixels: torch.Tensor) -> list[list[int]]:
            transitions = pixels[:, 1:] != pixels[:, :-1]
            row_idx, col_idx = torch.where(transitions)
            col_idx = col_idx + 1
            counts = []
            for i in range(pixels.shape[0]):
                positions = col_idx[row_idx == i]
                if len(positions):
                    count = torch.diff(positions).tolist()
                    count.insert(0, positions[0].item())
                    count.append(len(pixels[i]) - positions[-1].item())
                else:
                    count = [len(pixels[i])]
                if pixels[i][0].item() == 1:
                    count = [0, *count]
                counts.append(count)
            return counts

        pred_masks = predn["masks"].transpose(2, 1).contiguous().view(len(predn["masks"]), -1)
        h, w = predn["masks"].shape[1:3]
        counts = multi_encode(pred_masks)
        rles = [{"size": [h, w], "counts": to_string(c)} for c in counts]

        super().pred_to_json(predn, pbatch)
        for i, r in enumerate(rles):
            self.jdict[-len(rles) + i]["segmentation"] = r

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scale predictions to original image size."""
        if "cls" in predn and predn["cls"] is not None:
            predn["cls"] = self._force_tree_cls_tensor(predn["cls"])

        return {
            **super().scale_preds(predn, pbatch),
            "masks": ops.scale_masks(predn["masks"][None], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])[
                0
            ].byte(),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Return COCO-style instance segmentation evaluation metrics."""
        pred_json = self.save_dir / "predictions.json"
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "segm"], suffix=["Box", "Mask"])
