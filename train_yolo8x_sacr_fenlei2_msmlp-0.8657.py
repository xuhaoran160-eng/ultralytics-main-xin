# -*- coding: utf-8 -*-
"""
Train YOLOv8x-SACR-fenlei2-MSMLP.

This version keeps SACR mask refinement local, but moves final species prediction to a detached
multi-scale MLP classifier. Species classification starts after 20 epochs by default.

Replace files:
  1) ultralytics/nn/modules/head.py
  2) ultralytics/nn/tasks.py
  3) ultralytics/nn/modules/__init__.py   (only if SACRSpeciesSegment is not imported/exported)
  4) use yolov8x-sacr-fenlei2-msmlp.yaml
  5) run this train file

  Python train_yolo8x_sacr_fenlei2_msmlp-0.8657.py

"""
from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

CFG = {
    "name": "SACR_fenlei2_MSMLP_start20_detached",
    # SACR local mask correction, keep small and stable.
    # Target alpha. Actual training alpha schedule:
    #   epoch < 20:      effective_refine_alpha = 0
    #   epoch 20 ~ 40:   linear warm-up 0 -> 0.03
    #   epoch >= 40:     effective_refine_alpha = 0.03
    "refine_alpha": 0.03,
    "refine_start_epoch": 20,
    "refine_warmup_epochs": 20,
    # Species MLP losses. Both start after species_start_epoch.
    "species_gain": 0.10,          # anchor/location MLP species CE
    "object_species_gain": 0.20,   # GT-mask object-level MLP species CE
    "species_start_epoch": 20,
    # Critical gradient blockers.
    "species_cls_detach": True,    # anchor/location MLP cannot update YOLO backbone/neck
    "object_feature_detach": True, # object MLP cannot update YOLO backbone/neck
    "anchor_species_loss": True,
    "object_species_loss": True,
    # MLP details.
    "species_mlp_hidden": 256,
    "species_mlp_dropout": 0.10,
    "local_gap_kernel": 3,
}


def find_last_head(obj, depth: int = 0):
    if obj is None or depth > 12:
        return None
    if isinstance(obj, (list, tuple)):
        return obj[-1] if len(obj) else None
    if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        try:
            return obj[-1] if len(obj) else None
        except Exception:
            pass
    inner = getattr(obj, "model", None)
    if inner is not None and inner is not obj:
        return find_last_head(inner, depth + 1)
    return None


def apply_cfg(head, cfg=CFG, reset_debug=True):
    if head is None or head.__class__.__name__ != "SACRSpeciesSegment":
        raise RuntimeError(f"最后一层不是 SACRSpeciesSegment，当前为: {type(head)}")
    for k, v in cfg.items():
        if k == "name":
            continue
        if hasattr(head, k):
            setattr(head, k, v)
    if reset_debug and hasattr(head, "_msmlp_debug_printed"):
        try:
            delattr(head, "_msmlp_debug_printed")
        except Exception:
            pass
    return head


def set_epoch(head, epoch: int):
    if head is not None and head.__class__.__name__ == "SACRSpeciesSegment":
        setattr(head, "_current_epoch", int(epoch))


def print_check(head, title):
    print("=" * 80)
    print(title)
    print("head:", head.__class__.__name__)
    keys = [
        "tree_nc", "species_nc", "nm", "npr", "refine_alpha",
        "refine_start_epoch", "refine_warmup_epochs",
        "species_gain", "object_species_gain", "species_start_epoch",
        "species_cls_detach", "object_feature_detach",
        "anchor_species_loss", "object_species_loss",
        "species_mlp_hidden", "species_mlp_dropout", "local_gap_kernel",
    ]
    for k in keys:
        print(f"{k}:", getattr(head, k, None))
    print("=" * 80)


def make_train_start_callback():
    def _cb(trainer):
        head = find_last_head(getattr(trainer, "model", None))
        apply_cfg(head, CFG, reset_debug=True)
        set_epoch(head, int(getattr(trainer, "epoch", 0) or 0))
        print_check(head, "[SACR-MSMLP CHECK INSIDE TRAINER - REAL MODEL]")
        return None
    return _cb


def make_epoch_callback():
    def _cb(trainer):
        head = find_last_head(getattr(trainer, "model", None))
        if head is None:
            return None
        apply_cfg(head, CFG, reset_debug=False)
        ep = int(getattr(trainer, "epoch", 0) or 0)
        set_epoch(head, ep)
        if ep == int(CFG["species_start_epoch"]):
            print(f"[SACR-MSMLP] Species MLP losses are now ACTIVE at epoch {ep}.")
        if hasattr(head, "_get_effective_refine_alpha") and (ep < 45 or ep % 10 == 0):
            print(f"[SACR-MSMLP] epoch={ep}, effective_refine_alpha={head._get_effective_refine_alpha():.6f}")
        return None
    return _cb


def main():
    data_yaml = r"D:/Users/User/Desktop/newdata/yolo_dataset_chm_depth_01/tree_tp_seg.yaml"
    model_yaml = ROOT / "yolov8x-sacr-fenlei2-msmlp.yaml"

    model = YOLO(str(model_yaml))
    head = find_last_head(model.model)
    apply_cfg(head, CFG, reset_debug=True)
    set_epoch(head, 0)
    print_check(head, "[SACR-MSMLP CHECK BEFORE model.train()]")

    try:
        model.add_callback("on_train_start", make_train_start_callback())
        model.add_callback("on_train_epoch_start", make_epoch_callback())
    except Exception as e:
        print(f"[WARN] add_callback failed: {e}")

    model.train(
        data=data_yaml,
        imgsz=512,
        epochs=80,
        batch=24,
        device=0,
        workers=16,
        single_cls=False,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        mask_ratio=2,
        overlap_mask=False,  # REQUIRED for object-level GT mask pooling: batch['masks'] should be [N,H,W]
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=2.0,
        translate=0.10,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        cache=False,
        close_mosaic=0,
        patience=50,
        project=str(ROOT / "runs/segment"),
        name=CFG["name"],
        pretrained=False,
        val=True,
    )


if __name__ == "__main__":
    main()
