# -*- coding: utf-8 -*-
"""姿态识别工具（阶段3 · 支撑 D4 教姿教态评价）。

输入：试讲视频路径。
输出：逐帧姿态统计 + 聚合指标（手势活跃度、身体倾斜、面向变化）。

技术：MediaPipe PoseLandmarker（新版 tasks API），对视频均匀抽帧检测 33 个姿态关键点。
D4 教姿教态指标：
  - D4.1 站姿自然   -> 身体倾斜度（鼻子相对肩中点的水平偏移）
  - D4.2 手势得体   -> 双手张开度（腕距/肩宽）的均值与活跃度（标准差）
  - D4.3 面向学生   -> 肩宽随时间稳定性 / 身体转向（左右肩不对称度）
  - D4.4 教态稳定   -> 姿态检测率（多少帧成功检测到人体）
"""
from __future__ import annotations
import os
from typing import Any

import cv2
import numpy as np

from app.tools.registry import ToolSpec, ToolRegistry, registry

# 延迟导入 MediaPipe（避免无模型时阻塞）
_detector = None
_MODEL = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_lite.task")


def _get_detector():
    global _detector
    if _detector is None:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        base = BaseOptions(model_asset_path=_MODEL)
        opts = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        _detector = vision.PoseLandmarker.create_from_options(opts)
    return _detector


def _analyze_frame(frame, detector) -> dict | None:
    """单帧姿态分析。返回手势张开度、身体倾斜、肩宽等。"""
    import mediapipe as mp
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect(img)
    if not res.pose_landmarks:
        return None
    lm = res.pose_landmarks[0]
    H, W = frame.shape[:2]

    def pt(i):
        return np.array([lm[i].x * W, lm[i].y * H])

    nose = pt(0)
    lsh, rsh = pt(11), pt(12)
    lw, rw = pt(15), pt(16)          # 左右腕
    lel, rel = pt(13), pt(14)        # 左右肘
    shoulder_w = float(np.hypot(*(lsh - rsh)))
    if shoulder_w < 1e-4:
        return None
    wrist_dist = float(np.hypot(*(lw - rw)))
    spread = wrist_dist / shoulder_w                 # 手势张开度
    mid_shoulder = (lsh + rsh) / 2
    lean = float((nose[0] - mid_shoulder[0]) / shoulder_w)  # 身体水平倾斜
    # 面向：左右肩到鼻子距离差 -> 正=面向右侧, 负=面向左侧（转身信号）
    dl = float(np.hypot(*(nose - lsh)))
    dr = float(np.hypot(*(nose - rsh)))
    face = (dl - dr) / max(shoulder_w, 1e-4)
    # 手臂上举度（肘高于肩=举手/板书）
    hand_up = 1.0 if (lel[1] < lsh[1] or rel[1] < rsh[1]) else 0.0
    # 原始关键点序列（33个，归一化坐标），供落盘/追溯/可视化
    keypoints = [{"x": round(lm[i].x, 4), "y": round(lm[i].y, 4),
                  "z": round(float(lm[i].z), 4), "v": round(float(lm[i].visibility), 3)}
                 for i in range(len(lm))]
    return {"spread": round(spread, 3), "lean": round(lean, 3),
            "face": round(face, 3), "hand_up": hand_up,
            "keypoints": keypoints}


def _pose_analyze(video_path: str, n_frames: int = 20) -> dict:
    """对视频抽帧做姿态分析，聚合为 D4 教姿指标。"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频不存在: {video_path}")
    detector = _get_detector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, n_total // n_frames)
    frames_meta = []
    fi = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if fi % step == 0 and len(frames_meta) < n_frames:
            meta = _analyze_frame(frame, detector)
            if meta:
                meta["t_sec"] = round(fi / fps, 2)
                frames_meta.append(meta)
        fi += 1
    cap.release()

    if not frames_meta:
        return {"detected": 0, "total_samples": 0, "message": "未检测到人体姿态",
                "metrics": {}}
    spreads = [f["spread"] for f in frames_meta]
    leans = [f["lean"] for f in frames_meta]
    faces = [f["face"] for f in frames_meta]
    hand_ups = [f["hand_up"] for f in frames_meta]
    metrics = {
        "gesture_mean": round(float(np.mean(spreads)), 3),      # 平均手势张开度
        "gesture_std": round(float(np.std(spreads)), 3),        # 手势活跃度(变化)
        "lean_mean": round(float(np.mean(leans)), 3),           # 身体倾斜均值
        "lean_abs_max": round(float(np.max(np.abs(leans))), 3), # 最大倾斜
        "face_std": round(float(np.std(faces)), 3),             # 面向/转身变化
        "hand_up_ratio": round(float(np.mean(hand_ups)), 3),    # 举手/板书比例
        "detected": len(frames_meta),
        "total_samples": len(frames_meta),
    }
    return {"detected": len(frames_meta), "total_samples": len(frames_meta),
            "metrics": metrics, "samples": frames_meta[:n_frames]}


POSE_SPEC = ToolSpec(
    name="pose_analyze",
    description="对试讲视频做姿态识别（MediaPipe PoseLandmarker），输出手势张开度、身体倾斜、面向变化等教姿教态指标，"
                "供 D4 教姿教态评价使用。",
    parameters={"type": "object", "properties": {
        "video_path": {"type": "string", "description": "试讲视频绝对路径"},
        "n_frames": {"type": "integer", "description": "抽帧数，默认20"},
        },
        "required": ["video_path"]},
    category="vision",
)


def register_pose_tool(reg: ToolRegistry = registry) -> None:
    reg.register(POSE_SPEC, _pose_analyze)


register_pose_tool()
