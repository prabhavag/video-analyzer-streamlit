import io
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import ffmpeg
import numpy as np
import streamlit as st
import torch
from PIL import Image
import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import CLIPModel, CLIPProcessor, pipeline


def format_ffmpeg_error(error: Exception) -> str:
    stderr = getattr(error, "stderr", None)
    if isinstance(stderr, (bytes, bytearray)):
        return stderr.decode("utf-8", errors="ignore")
    if stderr:
        return str(stderr)
    return str(error)


def probe_video(video_path: Path) -> Dict[str, Any]:
    """Return ffprobe metadata for the supplied video file."""
    try:
        return ffmpeg.probe(str(video_path))
    except Exception as exc:  # pragma: no cover - ffmpeg error handling
        raise RuntimeError(format_ffmpeg_error(exc)) from exc


def extract_stream(streams: list[Dict[str, Any]], stream_type: str) -> Optional[Dict[str, Any]]:
    """Pick the first stream of a given type ('video' or 'audio')."""
    return next((stream for stream in streams if stream.get("codec_type") == stream_type), None)


def format_duration(seconds: Optional[str]) -> Optional[str]:
    if seconds is None:
        return None
    try:
        total_seconds = float(seconds)
    except (TypeError, ValueError):
        return None
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"
    return f"{int(minutes):02d}:{secs:06.3f}"


def summarize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    format_info = metadata.get("format", {})
    streams = metadata.get("streams", [])
    video_stream = extract_stream(streams, "video")
    audio_stream = extract_stream(streams, "audio")

    summary: Dict[str, Any] = {
        "File Format": format_info.get("format_long_name"),
        "File Size (MB)": round(float(format_info.get("size", 0)) / (1024 * 1024), 2)
        if format_info.get("size")
        else None,
        "Duration": format_duration(format_info.get("duration")),
        "Bitrate (kbps)": round(float(format_info.get("bit_rate", 0)) / 1000, 2)
        if format_info.get("bit_rate")
        else None,
    }

    if video_stream:
        summary["Video Codec"] = video_stream.get("codec_name")
        summary["Video Resolution"] = f"{video_stream.get('width')}x{video_stream.get('height')}"
        summary["Video Frame Rate"] = video_stream.get("avg_frame_rate")

    if audio_stream:
        summary["Audio Codec"] = audio_stream.get("codec_name")
        summary["Audio Channels"] = audio_stream.get("channels")
        summary["Audio Sample Rate (Hz)"] = audio_stream.get("sample_rate")

    return {key: value for key, value in summary.items() if value is not None}


def get_video_duration(metadata: Dict[str, Any]) -> Optional[float]:
    duration_str = metadata.get("format", {}).get("duration")
    if not duration_str:
        return None
    try:
        return float(duration_str)
    except (TypeError, ValueError):
        return None


@st.cache_resource(show_spinner=False)
def load_caption_model(model_name: str = "Salesforce/blip-image-captioning-base"):
    return pipeline("image-to-text", model=model_name)


CAPTION_MODEL = load_caption_model()


@st.cache_resource(show_spinner=False)
def load_clip_model(model_name: str = "openai/clip-vit-base-patch32"):
    try:
        processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
    except ImportError:
        processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    return processor, model


CLIP_PROCESSOR, CLIP_MODEL = load_clip_model()


def extract_sample_frames(
    video_path: Path,
    metadata: Dict[str, Any],
    duration_limit: float = 30.0,
    max_frames: int = 6,
) -> list[Image.Image]:
    detected_duration = get_video_duration(metadata)
    duration_seconds = min(duration_limit, detected_duration) if detected_duration else duration_limit

    if duration_seconds <= 0:
        duration_seconds = duration_limit

    frame_count = max(1, min(max_frames, int(math.ceil(duration_seconds))))
    step = duration_seconds / (frame_count + 1)

    frames: list[Image.Image] = []
    for idx in range(frame_count):
        timestamp = max(0.0, step * (idx + 1))
        try:
            stdout, _ = (
                ffmpeg.input(str(video_path), ss=timestamp)
                .output("pipe:", vframes=1, format="image2", vcodec="mjpeg")
                .run(capture_stdout=True, capture_stderr=True)
            )
            if not stdout:
                continue
            image = Image.open(io.BytesIO(stdout)).convert("RGB")
            frames.append(image)
        except Exception:
            continue

    return frames


def generate_video_caption(frames: list[Image.Image]) -> str:
    if not frames:
        raise RuntimeError("Could not sample frames from the first 30 seconds of the video.")
    captions: list[str] = []
    for frame in frames:
        result = CAPTION_MODEL(frame)
        if not result:
            continue
        text = result[0].get("generated_text")
        if text:
            text = text.strip()
            if text and text not in captions:
                captions.append(text)

    if not captions:
        raise RuntimeError("The captioning model did not produce any output.")

    return " ".join(captions)


def compute_caption_alignment(frames: list[Image.Image], caption: str) -> float:
    if not frames:
        raise RuntimeError("No frames available for alignment.")
    if not caption:
        raise RuntimeError("Generated caption is empty.")

    text_inputs = CLIP_PROCESSOR(
        text=[caption],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    image_inputs = CLIP_PROCESSOR(
        images=frames,
        return_tensors="pt",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIP_MODEL.to(device)
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    image_inputs = {key: value.to(device) for key, value in image_inputs.items()}

    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        image_features = clip_model.get_image_features(**image_inputs)

    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    image_features = torch.nn.functional.normalize(image_features, dim=-1)

    image_embedding = image_features.mean(dim=0, keepdim=True)
    image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)

    similarity = (image_embedding @ text_features.T).squeeze().item()
    return float(similarity)


def detect_scenes(
    video_path: Path,
    duration_seconds: Optional[float],
    threshold: float = 27.0,
) -> list[tuple[float, float]]:
    video_manager = VideoManager([str(video_path)])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    try:
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
    except Exception as exc:  # pragma: no cover - scene detection error handling
        raise RuntimeError(str(exc)) from exc
    finally:
        video_manager.release()

    if scene_list:
        return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

    if duration_seconds and duration_seconds > 0:
        return [(0.0, duration_seconds)]

    raise RuntimeError("No scenes detected in the video.")


def compute_scene_motion(
    video_path: Path,
    scenes: list[tuple[float, float]],
    sample_fps: float = 5.0,
) -> list[Dict[str, float | int]]:
    if not scenes:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Unable to open video for motion analysis.")

    results: list[Dict[str, float | int]] = []
    try:
        for index, (start, end) in enumerate(scenes, start=1):
            if end <= start:
                continue

            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
            ret, prev_frame = cap.read()
            if not ret:
                continue

            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            magnitudes: list[float] = []
            last_sample_time = start

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if current_time >= end:
                    break

                if current_time - last_sample_time < 1.0 / sample_fps:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                magnitudes.append(float(np.mean(mag)))
                prev_gray = gray
                last_sample_time = current_time

            if magnitudes:
                raw_motion = float(np.mean(magnitudes))
                normalized_motion = float(np.clip(raw_motion / 10.0, 0.0, 1.0))
            else:
                raw_motion = 0.0
                normalized_motion = 0.0

            results.append(
                {
                    "Scene": index,
                    "Start (s)": start,
                    "End (s)": end,
                    "Duration (s)": end - start,
                    "Motion (raw)": raw_motion,
                    "Motion Score": normalized_motion,
                }
            )
    finally:
        cap.release()

    return results


def main() -> None:
    st.set_page_config(page_title="Video Analyzer", page_icon="🎬", layout="wide")
    st.title("🎬 Video Analyzer")
    st.write("Upload a video file to inspect its metadata and preview a thumbnail.")

    st.session_state.setdefault("caption_original", None)
    st.session_state.setdefault("caption_editor", "")
    st.session_state.setdefault("alignment_result", {"score": None, "error": None, "caption": None})

    uploaded_file = st.file_uploader(
        "Select a video file",
        type=["mp4", "mov", "avi", "mkv", "flv", "webm"],
        accept_multiple_files=False,
        help="Supported formats include MP4, MOV, AVI, MKV, FLV, WEBM.",
    )

    if not uploaded_file:
        st.info("Awaiting video upload...")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_video_path = Path(tmp.name)

    try:
        metadata = probe_video(temp_video_path)
        summary = summarize_metadata(metadata)
    except RuntimeError as error:
        st.error(f"Unable to process the video: {error}")
        return

    st.subheader("Preview")
    preview_col, _ = st.columns([1, 2])
    with preview_col:
        st.video(str(temp_video_path))

    st.subheader("Metadata Summary")
    st.json(summary)

    with st.spinner("Sampling frames..."):
        frames = extract_sample_frames(temp_video_path, metadata)

    caption: Optional[str] = None
    caption_error: Optional[str] = None
    if frames:
        with st.spinner("Generating caption..."):
            try:
                caption = generate_video_caption(frames)
            except RuntimeError as error:
                caption_error = str(error)
    else:
        caption_error = "Could not sample frames from the first 30 seconds of the video."

    st.subheader("Video Caption")
    active_caption: Optional[str] = None
    recompute_clicked = False
    if caption:
        if st.session_state.get("caption_original") != caption:
            st.session_state["caption_original"] = caption
            st.session_state["caption_editor"] = caption
            st.session_state["alignment_result"] = {"score": None, "error": None, "caption": None}

        user_caption = st.text_area(
            "Caption (edit to refine)",
            key="caption_editor",
            help="The model-generated caption is pre-filled. Update the text to re-score alignment.",
        )
        active_caption = user_caption.strip()
        recompute_clicked = st.button(
            "Recompute Caption Alignment",
            disabled=not active_caption,
        )
        st.caption("Model-generated caption above; use the button to refresh the CLIP score after edits.")
        if not active_caption:
            st.info("Enter a caption to evaluate alignment.")
    elif caption_error:
        st.warning(caption_error)
        st.session_state["caption_original"] = None
        st.session_state["caption_editor"] = ""
        st.session_state["alignment_result"] = {"score": None, "error": None, "caption": None}

    alignment_error: Optional[str] = None
    if frames:
        if active_caption:
            alignment_result = st.session_state.get(
                "alignment_result", {"score": None, "error": None, "caption": None}
            )
            should_compute = False
            if (
                alignment_result.get("caption") is None
                and alignment_result.get("score") is None
                and alignment_result.get("error") is None
            ):
                should_compute = True
            if recompute_clicked:
                should_compute = True

            if should_compute:
                with st.spinner("Evaluating caption alignment..."):
                    try:
                        score = compute_caption_alignment(frames, active_caption)
                        alignment_result = {"score": score, "error": None, "caption": active_caption}
                    except RuntimeError as error:
                        alignment_result = {"score": None, "error": str(error), "caption": active_caption}
                    st.session_state["alignment_result"] = alignment_result
        else:
            if caption_error:
                alignment_error = "Caption generation failed; cannot evaluate alignment."
            else:
                alignment_error = "Provide a caption to evaluate alignment."
            st.session_state["alignment_result"] = {"score": None, "error": alignment_error, "caption": None}
    else:
        alignment_error = "No frames available; skipping alignment."
        st.session_state["alignment_result"] = {"score": None, "error": alignment_error, "caption": None}

    alignment_result = st.session_state.get(
        "alignment_result", {"score": None, "error": None, "caption": None}
    )
    display_score = alignment_result.get("score")
    display_error = alignment_result.get("error")

    if active_caption:
        stored_caption = alignment_result.get("caption")
        if stored_caption and stored_caption != active_caption:
            display_score = None
            alignment_error = "Caption edited; click 'Recompute Caption Alignment' to update the score."
        elif not stored_caption and display_score is None and display_error is None:
            alignment_error = alignment_error or "Click 'Recompute Caption Alignment' to evaluate the caption."
    else:
        display_score = None

    st.subheader("Caption Alignment")
    if display_score is not None:
        st.metric("CLIP Score", f"{display_score:.3f}")
    else:
        message = alignment_error or display_error
        if message:
            if message.startswith("Caption edited") or message.startswith("Provide a caption"):
                st.info(message)
            else:
                st.warning(message)

    duration_seconds = get_video_duration(metadata)
    scene_error: Optional[str] = None
    scenes: list[tuple[float, float]] = []
    with st.spinner("Detecting scenes..."):
        try:
            scenes = detect_scenes(temp_video_path, duration_seconds)
        except RuntimeError as error:
            scene_error = str(error)

    motion_results: list[Dict[str, float | int]] = []
    motion_error: Optional[str] = None
    if scenes:
        with st.spinner("Analyzing motion by scene..."):
            try:
                motion_results = compute_scene_motion(temp_video_path, scenes)
            except RuntimeError as error:
                motion_error = str(error)
    else:
        motion_error = scene_error or "No scenes identified; skipping motion analysis."

    st.subheader("Scene Motion")
    if motion_results:
        average_motion = float(
            np.mean([result["Motion Score"] for result in motion_results])
        )
        st.metric("Average Motion (0-1)", f"{average_motion:.3f}")
        st.dataframe(motion_results, use_container_width=True)
    else:
        if motion_error:
            st.warning(motion_error)
        if scene_error:
            st.info(f"Scene detection issue: {scene_error}")


if __name__ == "__main__":
    main()


