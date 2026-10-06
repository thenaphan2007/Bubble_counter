"""
Bubble Counter: Count bubbles moving from bottom to top in a video.

Approach:
- Background subtraction to isolate moving objects (bubbles).
- Contour-based detection with area and circularity filtering.
- Lightweight centroid tracker to persist IDs across frames.
- Count a bubble once when its track crosses a horizontal line going upward.

Requirements:
- Python 3.8+
- OpenCV (pip install opencv-python)
- NumPy (pip install numpy)

Usage:
  python bubble_counter.py --video /path/to/video.mp4 --display 1

Key options:
  --line-ratio 0.2     # Counting line Y-position as a fraction of frame height (near top)
  --start-ratio 0.7    # Only count tracks that started below this Y-ratio (i.e., near bottom)
  --min-area 80        # Min contour area to consider as bubble
  --max-area 8000      # Max contour area to consider as bubble
  --min-circularity 0.5  # 4πA/P^2 threshold
  --max-distance 60    # Max pixel distance for track association
  --max-missed 10      # Frames allowed to miss detection before track deletion
  --display 1          # Show visualization window

Notes:
- Tune thresholds for your specific video (lighting, bubble size, frame rate).
- The script counts upward crossings (bottom-to-top). It ignores downward movement.
"""

import argparse
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


@dataclass
class Track:
    id: int
    centroid: Tuple[int, int]
    last_centroid: Tuple[int, int]
    misses: int = 0
    counted: bool = False
    history: deque = None
    started_y: Optional[int] = None

    def __post_init__(self):
        if self.history is None:
            self.history = deque(maxlen=32)
        self.history.append(self.centroid)
        if self.started_y is None:
            self.started_y = self.centroid[1]


class CentroidTracker:
    def __init__(self, max_distance: float = 60.0, max_missed: int = 10):
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.max_distance = max_distance
        self.max_missed = max_missed

    def _distance(self, p: Tuple[int, int], q: Tuple[int, int]) -> float:
        dx = float(p[0] - q[0])
        dy = float(p[1] - q[1])
        return (dx * dx + dy * dy) ** 0.5

    def update(self, detections: List[Tuple[int, int]]) -> Dict[int, Track]:
        """
        Associate detections to existing tracks using greedy nearest matching.

        Args:
            detections: list of centroids (x, y)

        Returns:
            dict of track_id -> Track (updated)
        """
        if len(self.tracks) == 0:
            # Initialize tracks for all detections
            for c in detections:
                self._start_track(c)
            return self.tracks

        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[i].centroid for i in track_ids]

        used_tracks = set()
        used_dets = set()

        # Precompute distance matrix
        if len(track_centroids) and len(detections):
            dist_matrix = np.zeros((len(track_centroids), len(detections)), dtype=np.float32)
            for i, tc in enumerate(track_centroids):
                for j, dc in enumerate(detections):
                    dist_matrix[i, j] = self._distance(tc, dc)

            # Greedy global minimum assignment
            while True:
                min_val = np.min(dist_matrix) if dist_matrix.size else np.inf
                if not np.isfinite(min_val):
                    break
                i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)  # type: ignore
                if i in used_tracks or j in used_dets:
                    dist_matrix[i, j] = np.inf
                    continue
                if min_val > self.max_distance:
                    # No viable assignment remains
                    break

                tid = track_ids[i]
                self._update_track(tid, detections[j])
                used_tracks.add(i)
                used_dets.add(j)
                dist_matrix[i, :] = np.inf
                dist_matrix[:, j] = np.inf
        # Unassigned tracks -> increment miss
        for idx, tid in enumerate(track_ids):
            if idx not in used_tracks:
                self.tracks[tid].misses += 1

        # Remove stale tracks
        to_delete = [tid for tid, t in self.tracks.items() if t.misses > self.max_missed]
        for tid in to_delete:
            del self.tracks[tid]

        # Unassigned detections -> start new tracks
        for j, dc in enumerate(detections):
            if j not in used_dets:
                self._start_track(dc)

        return self.tracks

    def _start_track(self, c: Tuple[int, int]):
        t = Track(id=self.next_id, centroid=c, last_centroid=c, misses=0, counted=False)
        self.tracks[self.next_id] = t
        self.next_id += 1

    def _update_track(self, tid: int, c: Tuple[int, int]):
        t = self.tracks[tid]
        t.last_centroid = t.centroid
        t.centroid = c
        t.misses = 0
        t.history.append(c)


def compute_circularity(area: float, perimeter: float) -> float:
    if perimeter <= 0.0:
        return 0.0
    return float((4.0 * np.pi * area) / (perimeter * perimeter))


def detect_bubble_centroids(
    frame_bgr: np.ndarray,
    bg: cv2.BackgroundSubtractor,
    min_area: int,
    max_area: int,
    min_circularity: float,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Returns:
        centroids: list of (x, y)
        debug_mask: binary mask used for detection (for display)
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Background subtraction
    fmask = bg.apply(gray)  # default learning rate
    # Remove shadows if present (MOG2 shadows ~127). Threshold to binary.
    _, bin_mask = cv2.threshold(fmask, 200, 255, cv2.THRESH_BINARY)

    # Denoise and solidify blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    bin_mask = cv2.dilate(bin_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids: List[Tuple[int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        circ = compute_circularity(area, perimeter)
        if circ < min_circularity:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroids.append((cx, cy))

    return centroids, bin_mask


def draw_visualization(
    frame: np.ndarray,
    tracks: Dict[int, Track],
    centroids: List[Tuple[int, int]],
    line_y: int,
    start_y_min: Optional[int],
    total_count: int,
):
    # Draw detected centroids
    for (x, y) in centroids:
        cv2.circle(frame, (x, y), 5, (0, 255, 255), 2)

    # Draw counting line
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
    if start_y_min is not None:
        cv2.line(frame, (0, start_y_min), (frame.shape[1], start_y_min), (255, 0, 0), 1)

    # Draw tracks
    for tid, t in tracks.items():
        x, y = t.centroid
        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"ID {tid}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        # trajectory
        pts = list(t.history)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (0, 200, 0), 2)

    # Draw total count
    cv2.putText(
        frame,
        f"Count: {total_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def run(video_path: str,
        min_area: int = 80,
        max_area: int = 8000,
        min_circularity: float = 0.5,
        line_ratio: float = 0.2,
        start_ratio: float = 0.7,
        max_distance: float = 60.0,
        max_missed: int = 10,
        display: bool = True) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Warmup to get frame size
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to read first frame from video.")
    height, width = frame.shape[:2]

    line_y = int(height * line_ratio)
    start_y_min = int(height * start_ratio) if start_ratio is not None else None

    # Background subtractor
    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    tracker = CentroidTracker(max_distance=max_distance, max_missed=max_missed)
    total_count = 0

    # Rewind to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = int(1000.0 / fps) if display else 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        centroids, _mask = detect_bubble_centroids(
            frame, bg, min_area=min_area, max_area=max_area, min_circularity=min_circularity
        )

        tracks = tracker.update(centroids)

        # Count upward crossing
        for tid, t in tracks.items():
            if t.counted:
                continue
            # Only consider tracks that originated below start_y_min (near bottom)
            if start_y_min is not None and (t.started_y is None or t.started_y < start_y_min):
                continue

            # Need previous and current positions
            prev_y = t.last_centroid[1] if t.last_centroid else t.centroid[1]
            curr_y = t.centroid[1]
            dy = curr_y - prev_y

            # crossing upward: previously below line, now above or on the line, and moving upward (dy < 0)
            if prev_y > line_y and curr_y <= line_y and dy < 0:
                total_count += 1
                t.counted = True

        if display:
            vis = frame.copy()
            draw_visualization(
                vis, tracks, centroids, line_y=line_y, start_y_min=start_y_min, total_count=total_count
            )
            cv2.imshow("Bubble Counter", vis)
            key = cv2.waitKey(delay_ms) & 0xFF
            # Quit on 'q' or ESC
            if key == ord('q') or key == 27:
                break

    cap.release()
    if display:
        cv2.destroyAllWindows()

    print(f"Total bubbles counted (upward crossings): {total_count}")
    return total_count


def parse_args():
    p = argparse.ArgumentParser(description="Count bottom-to-top bubbles in a video.")
    p.add_argument("--video", required=True, help="Path to the input video file")
    p.add_argument("--min-area", type=int, default=80, help="Min contour area")
    p.add_argument("--max-area", type=int, default=8000, help="Max contour area")
    p.add_argument("--min-circularity", type=float, default=0.5, help="Min circularity 4πA/P^2")
    p.add_argument("--line-ratio", type=float, default=0.2, help="Counting line Y ratio (0 top ... 1 bottom)")
    p.add_argument("--start-ratio", type=float, default=0.7, help="Only count tracks that started below this ratio")
    p.add_argument("--max-distance", type=float, default=60.0, help="Max pixel distance for track association")
    p.add_argument("--max-missed", type=int, default=10, help="Max consecutive missed frames before dropping a track")
    p.add_argument("--display", type=int, default=1, help="Show visualization window (1/0)")
    return p.parse_args()


def main():
    args = parse_args()
    run(
        video_path=args.video,
        min_area=args.min_area,
        max_area=args.max_area,
        min_circularity=args.min_circularity,
        line_ratio=args.line_ratio,
        start_ratio=args.start_ratio,
        max_distance=args.max_distance,
        max_missed=args.max_missed,
        display=bool(args.display),
    )


if __name__ == "__main__":
    main()
