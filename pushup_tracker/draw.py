import cv2
from mediapipe.tasks.python.vision import PoseLandmark, PoseLandmarksConnections


def draw_pose(frame, landmarks, min_visibility: float = 0.5) -> None:
    """Draw pose skeleton on a BGR frame using OpenCV."""
    h, w = frame.shape[:2]

    def pt(index: int) -> tuple[int, int] | None:
        lm = landmarks[index]
        if lm.visibility < min_visibility:
            return None
        return int(lm.x * w), int(lm.y * h)

    for conn in PoseLandmarksConnections.POSE_LANDMARKS:
        a = pt(conn.start)
        b = pt(conn.end)
        if a and b:
            cv2.line(frame, a, b, (0, 255, 255), 2, cv2.LINE_AA)

    for landmark in PoseLandmark:
        p = pt(landmark.value)
        if p:
            cv2.circle(frame, p, 4, (0, 128, 255), -1, cv2.LINE_AA)
