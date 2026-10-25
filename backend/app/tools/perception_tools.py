# -*- coding: utf-8 -*-
"""多模态感知工具（L3 工具层 · 阶段3）。

把"从视频提取多模态信号"封装为可插拔工具，注册进 ToolRegistry：
- asr_transcribe：从视频/音频提取语音文本 + 词级时间戳 + 语音特征（D2 讲授语言）
- ocr_board：从视频帧提取板书文字 + 版面信号（D3 板书）
- timeline_slice：基于 ASR 分段做教学环节时间轴切片（D1 教学设计）

设计：先验证后封装——每个工具都基于真实视频实测通过。ffmpeg 负责分离音频/抽帧。
"""
from __future__ import annotations
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from app.config import settings
from app.tools.registry import ToolSpec, ToolRegistry, registry

FFMPEG = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
ASR_MODEL_SIZE = "small"
_asr_model = None  # 懒加载单例


# ============================================================
# ASR：从视频/音频提取语音文本 + 时间戳 + 语音特征
# ============================================================
def _load_asr_model():
    global _asr_model
    if _asr_model is None:
        from faster_whisper import WhisperModel
        _asr_model = WhisperModel(ASR_MODEL_SIZE, device="cpu", compute_type="int8")
    return _asr_model


def _extract_audio(video_path: str) -> str:
    """从视频分离 16k 单声道 wav，返回 wav 路径。"""
    tmp = tempfile.mkdtemp(prefix="asr_audio_")
    wav = str(Path(tmp) / "audio.wav")
    subprocess.run([FFMPEG, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1", wav],
                   capture_output=True, check=True)
    return wav


def _asr_transcribe(video_or_audio: str) -> dict:
    """从视频/音频提取语音文本 + 词级时间戳 + 语音特征。"""
    p = Path(video_or_audio)
    if p.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
        audio = _extract_audio(str(p))
    else:
        audio = str(p)
    model = _load_asr_model()
    segments, info = model.transcribe(audio, language="zh", beam_size=5,
                                      word_timestamps=True, vad_filter=True)
    seg_list, words = [], []
    texts = []
    for seg in segments:
        texts.append(seg.text.strip())
        seg_list.append({"start": round(seg.start, 2), "end": round(seg.end, 2),
                         "text": seg.text.strip()})
        for w in (seg.words or []):
            words.append({"word": w.word, "start": round(w.start, 2), "end": round(w.end, 2)})
    full = "".join(texts)

    # 语音特征（支撑 D2 流畅度/节奏）
    dur = info.duration or 1.0
    word_count = len(words)
    feats = {
        "duration_sec": round(dur, 1),
        "word_count": word_count,
        "wpm": round(word_count / (dur / 60), 1),          # 语速(字/分钟)
        "segments": len(seg_list),
        # 口头禅候选（简单高频词检测）
        "filler_words": _detect_fillers(full),
        # 停顿：相邻词间隙 >1.5s 记为停顿
        "pauses": _detect_pauses(words),
    }
    return {"text": full, "segments": seg_list, "words": words, "feats": feats}


_FILLERS = ("然后", "那么", "呃", "嗯", "啊", "就是说", "那个", "这个")


def _detect_fillers(text: str) -> dict:
    counts = {}
    for f in _FILLERS:
        c = text.count(f)
        if c:
            counts[f] = c
    return counts


def _detect_pauses(words: list[dict], gap_th: float = 1.5) -> int:
    if len(words) < 2:
        return 0
    n = 0
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap > gap_th:
            n += 1
    return n


ASR_SPEC = ToolSpec(
    name="asr_transcribe",
    description="从试讲视频或音频提取讲授语音：完整文本、分段、词级时间戳、语速/口头禅/停顿特征。"
                "供 D2 讲授语言与 D1 教学设计评价使用。",
    parameters={"type": "object", "properties": {
        "video_or_audio": {"type": "string", "description": "视频(.mp4等)或音频(.wav)路径"}},
        "required": ["video_or_audio"]},
    output_schema={"type": "object", "properties": {"text": "string", "segments": "array",
                                                    "words": "array", "feats": "object"}},
    category="audio",
)


# ============================================================
# OCR：从视频帧提取板书（D3）
# ============================================================
def _extract_frames(video_path: str, fps: float = 0.5) -> list[str]:
    """低帧率抽帧，返回帧图片路径列表。"""
    tmp = tempfile.mkdtemp(prefix="ocr_frames_")
    out_pat = str(Path(tmp) / "frame_%03d.png")
    subprocess.run([FFMPEG, "-y", "-i", video_path, "-vf", f"fps={fps}",
                    "-qscale:v", "2", out_pat],
                   capture_output=True, check=True)
    return sorted(str(p) for p in Path(tmp).glob("frame_*.png"))


def _ocr_board(video_or_image: str, max_frames: int = 30) -> dict:
    """从视频帧/图片提取板书文字 + 版面信号。"""
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    p = Path(video_or_image)
    if p.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
        frames = _extract_frames(str(p))[:max_frames]
    else:
        frames = [str(p)]

    all_text, all_boxes = [], []
    for fr in frames:
        try:
            result, _ = ocr(fr)
            if not result:
                continue
            for line in result:
                box, text, conf = line[0], line[1], line[2]
                all_text.append(text)
                all_boxes.append({"text": text, "conf": round(float(conf), 3),
                                  "box": [round(float(x), 1) for pt in box for x in pt]})
        except Exception:
            continue
    # 去重（跨帧重复识别同一板书内容）
    seen, uniq = set(), []
    for t in all_text:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return {"text": "\n".join(uniq), "lines": uniq,
            "box_count": len(all_boxes), "frames": len(frames)}


OCR_SPEC = ToolSpec(
    name="ocr_board",
    description="从试讲视频抽帧或板书图片，OCR 识别板书文字与版面检测框。供 D3 板书评价使用。",
    parameters={"type": "object", "properties": {
        "video_or_image": {"type": "string", "description": "视频路径或板书图片路径"},
        "max_frames": {"type": "integer", "description": "最大抽帧数(可选)"}},
        "required": ["video_or_image"]},
    category="board",
)


# ============================================================
# 注册
# ============================================================
def register_perception_tools(reg: ToolRegistry = registry) -> None:
    reg.register(ASR_SPEC, _asr_transcribe)
    reg.register(OCR_SPEC, _ocr_board)


register_perception_tools()
