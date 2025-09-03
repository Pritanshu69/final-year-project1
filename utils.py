import numpy as np

#SECTION - all keypoints are stored as arrays and numpy is storing it as arrays

def _landmarks_to_array(hand_landmarks):
    pts = [([lm.x, lm.y, lm.z]) for lm in hand_landmarks.landmark]
    return np.array(pts,dtype=np.float32)


#SECTION - takes the wrist as ref point 

def _normalize_landmarks(pts):
    origin = pts[0].copy()
    pts_rel = pts - origin
    max_dist = np.linalg.norm(pts_rel, axis=1).max()
    scale = max_dist if max_dist > 1e-6 else 1.0
    pts_norm = pts_rel / scale
    return pts_norm.reshape(-1).astype(np.float32)


#SECTION - detects hand first and it should return 0 if no hand is detected

def extract_hand_keypoints(results):
    if not results or not getattr(results, "multi_hand_landmarks", None):
        return np.zeros((63,), dtype=np.float32)
    hand_landmarks = results.multi_hand_landmarks[0]
    pts = _landmarks_to_array(hand_landmarks)
    return _normalize_landmarks(pts)

#!SECTION - Truncates if too long.pads with zeros if too short.

def pad_or_truncate(sequence, target_len):
    seq = list(sequence)
    if len(seq) >= target_len:
        return np.array(seq[:target_len], dtype=np.float32)
    D = len(seq[0]) if len(seq) > 0 else 63
    padding = np.zeros((target_len - len(seq), D), dtype=np.float32)
    if len(seq) == 0:
        return padding
    return np.vstack([np.array(seq, dtype=np.float32), padding])

#!SECTION - Used during real-time inference.

def sliding_window_push(window, vec, maxlen):
    window.append(vec)
    if len(window) > maxlen:
        del window[0]
    return window
