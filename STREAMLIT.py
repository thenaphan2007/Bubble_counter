"""
Streamlit UI for Bubble Counter

Run:
  streamlit run bubble_counter/streamlit_app.py

This app lets you upload a video and configure detection/tracking parameters,
then runs the bubble counting logic headlessly and shows the total count.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

# Import the counter logic robustly to avoid package/module name collisions
try:
    # If this directory is treated as a package
    from .bubble_counter import run as count_bubbles  # type: ignore
except Exception:
    # Fallback: import by file path when running as a plain script
    import importlib.util
    from pathlib import Path as _Path
    _mod_path = _Path(__file__).parent / "bubble_counter.py"
    spec = importlib.util.spec_from_file_location("bubble_counter_module", str(_mod_path))
    if spec and spec.loader:
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
        count_bubbles = _mod.run  # type: ignore
    else:
        raise ImportError("Unable to import bubble_counter.run")


ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def save_uploaded_file(uploaded_file) -> str:
    """
    Save an uploaded file to a temporary file and return its path.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name


def main():
    st.set_page_config(page_title="Bubble Counter", layout="centered")
    st.title("Bubble Counter")
    st.caption("Count bubbles moving bottom-to-top in a video (OpenCV + Streamlit)")

    with st.expander("How it works", expanded=False):
        st.markdown(
            "- Background subtraction isolates moving objects (bubbles).\n"
            "- Contours are filtered by area and circularity to detect bubbles.\n"
            "- A simple centroid tracker associates detections across frames.\n"
            "- A bubble is counted once when its track crosses a horizontal line moving upward.\n"
            "- Only tracks originating near the bottom are considered."
        )

    uploaded = st.file_uploader(
        "Upload a video", type=[ext.strip(".") for ext in ALLOWED_EXTENSIONS], accept_multiple_files=False
    )

    st.sidebar.header("Detection")
    min_area = st.sidebar.number_input("Min area", min_value=1, value=80, step=1, help="Minimum contour area (pixels)")
    max_area = st.sidebar.number_input("Max area", min_value=1, value=8000, step=1, help="Maximum contour area (pixels)")
    min_circularity = st.sidebar.slider(
        "Min circularity", min_value=0.0, max_value=1.0, value=0.5, step=0.01, help="0..1; higher = more circular"
    )

    st.sidebar.header("Tracking / Counting")
    line_ratio = st.sidebar.slider(
        "Counting line (Y ratio)", min_value=0.0, max_value=1.0, value=0.2, step=0.01, help="0 = top, 1 = bottom"
    )
    start_ratio = st.sidebar.slider(
        "Start ratio (Y)", min_value=0.0, max_value=1.0, value=0.7, step=0.01,
        help="Only count tracks that started below this ratio (toward the bottom)",
    )
    max_distance = st.sidebar.number_input(
        "Max association distance (px)", min_value=1, value=60, step=1,
        help="Maximum distance (pixels) to associate detections with a track",
    )
    max_missed = st.sidebar.number_input(
        "Max missed frames", min_value=0, value=10, step=1, help="Drop a track after this many missed frames"
    )

    run_clicked = st.button("Process Video", type="primary", disabled=(uploaded is None))

    if run_clicked:
        if uploaded is None:
            st.error("Please upload a video first.")
            st.stop()

        try:
            tmp_path = save_uploaded_file(uploaded)
            with st.spinner("Processing video..."):
                total = count_bubbles(
                    video_path=tmp_path,
                    min_area=min_area,
                    max_area=max_area,
                    min_circularity=min_circularity,
                    line_ratio=line_ratio,
                    start_ratio=start_ratio,
                    max_distance=float(max_distance),
                    max_missed=int(max_missed),
                    display=False,
                )
        except Exception as e:
            st.error(f"Processing error: {e}")
            return
        finally:
            # Clean up temp file
            try:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        st.success("Processing complete.")
        st.metric(label="Total bubbles (upward crossings)", value=total)

        with st.expander("Parameters used", expanded=False):
            st.json(
                {
                    "min_area": min_area,
                    "max_area": max_area,
                    "min_circularity": min_circularity,
                    "line_ratio": line_ratio,
                    "start_ratio": start_ratio,
                    "max_distance": max_distance,
                    "max_missed": max_missed,
                }
            )

        st.info(
            "Tips:\n"
            "- If under-counting, try lowering min_area or min_circularity.\n"
            "- If over-counting noise, raise min_area or min_circularity.\n"
            "- Adjust the counting line with line_ratio (e.g., 0.25)."
        )

    st.caption("Note: Processing runs headlessly; no display window is opened.")


if __name__ == "__main__":
    # When running via 'python bubble_counter/streamlit_app.py', do nothing.
    # Launch with:
    #   streamlit run bubble_counter/streamlit_app.py
    main()
