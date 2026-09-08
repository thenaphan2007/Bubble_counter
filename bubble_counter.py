import cv2
import numpy as np

def count_bubbles(video_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    tracks = {}             
    counted = set()
    track_id = 0
    bubble_count = 0

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    bottom_zone = int(frame_h * 0.80)
    reflection_zone = int(frame_h * 0.40)     # top reflections
    bottom_reflection_zone = int(frame_h * 0.90)   # 🔥 NEW (bottom reflections)

    # tuning
    max_dist = 40
    min_track_len = 8
    min_up = 20
    max_side_motion = 25
    max_speed = 35
    max_radius_std = 8

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        bright_mask = cv2.inRange(hsv, (0, 0, 0), (180, 70, 200))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        _, mask2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        mask = cv2.bitwise_and(bright_mask, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 50 or area > 4000:
                continue

            (x, y), r = cv2.minEnclosingCircle(c)
            x, y, r = int(x), int(y), int(r)

            if r < 4 or r > 40:
                continue

            # 🔥 IGNORE TOP REFLECTIONS
            if y < reflection_zone:
                continue

            # 🔥 IGNORE BOTTOM REFLECTIONS
            if y > bottom_reflection_zone:
                continue

            detections.append((x, y, r))

        used = set()
        for tid, data in tracks.items():
            if len(data["points"]) == 0:
                continue

            last_x, last_y = data["points"][-1]
            best_i = None
            best_dist = 9999

            for i, (x, y, r) in enumerate(detections):
                if i in used:
                    continue

                d = np.hypot(last_x - x, last_y - y)
                if d < best_dist:
                    best_dist = d
                    best_i = i

            if best_i is not None and best_dist < max_dist:
                used.add(best_i)
                x, y, r = detections[best_i]
                data["points"].append((x, y))
                data["radii"].append(r)

        for i, (x, y, r) in enumerate(detections):
            if i not in used and y >= bottom_zone:
                tracks[track_id] = {
                    "points": [(x, y)],
                    "radii": [r]
                }
                track_id += 1

        for tid, data in tracks.items():
            if tid in counted:
                continue

            pts = data["points"]

            if len(pts) < min_track_len:
                continue

            y0 = pts[0][1]
            y_last = pts[-1][1]
            if (y0 - y_last) < min_up:
                continue

            xs = [p[0] for p in pts]
            if max(xs) - min(xs) > max_side_motion:
                continue

            dy = abs(pts[-1][1] - pts[-2][1])
            if dy > max_speed:
                continue

            if np.std(data["radii"]) > max_radius_std:
                continue

            bubble_count += 1
            counted.add(tid)

    cap.release()
    return bubble_count

