"""Observation-only effect evidence for fixed-camera PushObj videos.

This module never accepts object state, shape geometry, or task success. Object-
region evidence means RGB changes OUTSIDE a proprio-defined pusher exclusion;
it is a diagnostic, not a label for the entire visual embedding.
"""
import cv2
import numpy as np

PAIRS = ((0, 5), (5, 10), (10, 15), (0, 15))
UNKNOWN, STATIC, EFFECT = -1, 0, 1


def motion_features(xy, velocity, actions):
    delta = np.linalg.norm(xy - xy[0], axis=-1)
    path = float(np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum())
    return {
        "net": float(delta[-1]), "excursion": float(delta.max()),
        "path": path, "net_path_ratio": float(delta[-1] / max(path, 1e-8)),
        "max_speed": float(np.linalg.norm(velocity, axis=-1).max()),
        "action_rms": float(np.sqrt(np.mean(np.sum(actions ** 2, axis=-1)))),
    }


def body_label(f, cfg):
    if (f["net"] >= cfg["body_effect_px"] and
            f["net_path_ratio"] >= cfg["min_net_path_ratio"]):
        return EFFECT
    if (f["excursion"] <= cfg["body_static_px"] and
            f["max_speed"] <= cfg["body_static_speed"] and
            f["action_rms"] >= cfg["min_action_rms_px"]):
        return STATIC
    return UNKNOWN


def exclusion_mask(xy, size, cfg, world_size=512):
    mask = np.full(size, 255, dtype=np.uint8)
    scale = size[0] / world_size
    radius = int(np.ceil(cfg["pusher_mask_radius_world"] * scale))
    for p in xy:
        cv2.circle(mask, tuple(np.rint(p * scale).astype(int)), radius, 0, -1)
    # Cropped border can contain compression artifacts or stationary wall edges.
    mask[:3] = mask[-3:] = 0
    mask[:, :3] = mask[:, -3:] = 0
    return mask


def largest_component(binary):
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0


def tracked_motion(gray0, gray1, mask, cfg):
    p0 = cv2.goodFeaturesToTrack(gray0, maxCorners=100, qualityLevel=.01,
                                 minDistance=3, mask=mask, blockSize=3)
    empty = {"tracked": 0, "moving": 0, "coherent": 0, "max_flow": 0.0}
    if p0 is None:
        return empty
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    p1, s1, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, p0, None, **lk)
    if p1 is None:
        return empty
    pb, sb, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, p1, None, **lk)
    if pb is None:
        return empty
    a, b = p0[:, 0], p1[:, 0]
    ok = (s1[:, 0] > 0) & (sb[:, 0] > 0)
    ok &= np.linalg.norm(pb[:, 0] - a, axis=-1) <= cfg["flow_fb_error_px"]
    finite = np.isfinite(b).all(axis=1)
    ok &= finite
    bindex = np.rint(np.nan_to_num(b)).astype(int)
    inside = ((bindex[:, 0] >= 0) & (bindex[:, 0] < mask.shape[1]) &
              (bindex[:, 1] >= 0) & (bindex[:, 1] < mask.shape[0]))
    ok &= inside
    idx = np.flatnonzero(ok)
    ok[idx] &= mask[bindex[idx, 1], bindex[idx, 0]] > 0
    a, b = a[ok], b[ok]
    displacement = np.linalg.norm(b - a, axis=-1)
    moving = displacement >= cfg["flow_effect_px"]
    coherent = 0
    if moving.sum() >= cfg["min_moving_corners"]:
        # Rotation is allowed: coherence is a rigid/similarity motion, not a
        # common direction (which would discard rotation about the centre).
        _, inliers = cv2.estimateAffinePartial2D(a[moving], b[moving],
            method=cv2.RANSAC, ransacReprojThreshold=1.0, maxIters=100,
            confidence=.99, refineIters=5)
        coherent = int(inliers.sum()) if inliers is not None else 0
    return {"tracked": len(a), "moving": int(moving.sum()), "coherent": coherent,
            "max_flow": float(displacement.max()) if len(displacement) else 0.0}


def measure_window(rgb, proprio, executed_actions, cfg, world_size=512):
    """16 dense frames, (16,4) raw proprio, and at least 15 pixel actions.

    Frames are PRE-action observations. Pair [s,e] uses actions[s:e]. The
    corresponding training slicer contains frames 0,5,10,15 and 20 actions.
    """
    if rgb.dtype != np.uint8 or rgb.shape[0] != 16 or proprio.shape != (16, 4):
        raise ValueError("Expected uint8 RGB[16,H,W,3], raw proprio[16,4]")
    if executed_actions.shape[0] < 15:
        raise ValueError("Insufficient actions for [0,15]")
    if not np.isfinite(proprio).all() or not np.isfinite(executed_actions).all():
        raise ValueError("Non-finite observation/action")
    gray = np.stack([cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in rgb])
    result = []
    for s, e in PAIRS:
        body = motion_features(proprio[s:e+1, :2], proprio[s:e+1, 2:],
                               executed_actions[s:e])
        outside = exclusion_mask(proprio[s:e+1, :2], rgb.shape[1:3], cfg, world_size)
        diff = np.max(np.abs(rgb[s:e+1].astype(np.int16) - rgb[s].astype(np.int16)), -1)
        changed = diff > cfg["rgb_diff_floor"]
        outside_changed = changed & (outside[None] > 0)
        outside_counts = outside_changed.sum(axis=(1, 2))
        whole_counts = changed.sum(axis=(1, 2))
        flow = tracked_motion(gray[s], gray[e], outside, cfg)
        f = {
            "pair": [s, e], "body": body,
            "outside_end_changed": int(outside_counts[-1]),
            "outside_max_changed": int(outside_counts.max()),
            "outside_end_component": largest_component(outside_changed[-1]),
            "whole_end_component": largest_component(changed[-1]),
            "whole_max_changed": int(whole_counts.max()), "flow": flow,
        }
        prop = body_label(body, cfg)
        object_effect = (f["outside_end_component"] >= cfg["changed_component_pixels"]
                         and flow["coherent"] >= cfg["min_moving_corners"])
        object_static = (flow["tracked"] >= cfg["min_tracked_corners"]
                         and flow["max_flow"] <= cfg["flow_static_px"]
                         and f["outside_max_changed"] <= cfg["max_static_changed_pixels"]
                         and body["action_rms"] >= cfg["min_action_rms_px"])
        obj = EFFECT if object_effect else STATIC if object_static else UNKNOWN
        # A moving pusher makes the full image non-static, regardless of the
        # object-only diagnostic. Never contract the full latent from obj=0.
        visual_effect = object_effect or (prop == EFFECT and
                         f["whole_end_component"] >= cfg["changed_component_pixels"])
        visual_static = (prop == STATIC and object_static and
                         f["whole_max_changed"] <= cfg["max_static_changed_pixels"])
        visual = EFFECT if visual_effect else STATIC if visual_static else UNKNOWN
        f["labels"] = {"proprio": prop, "visual": visual, "object_diagnostic": obj}
        result.append(f)
    # Compression-noise diagnostic on the outermost two rows, excluding the
    # pusher. Used on TRAIN only to validate the fixed RGB threshold.
    bg = exclusion_mask(proprio[:, :2], rgb.shape[1:3], cfg, world_size)
    bg[:] = 0
    bg[:2] = bg[-2:] = 255
    bg[:, :2] = bg[:, -2:] = 255
    scale = rgb.shape[1] / world_size
    for p in proprio[:, :2]:
        cv2.circle(bg, tuple(np.rint(p * scale).astype(int)),
                   int(np.ceil(cfg["pusher_mask_radius_world"] * scale)), 0, -1)
    differences = np.max(np.abs(np.diff(rgb.astype(np.int16), axis=0)), axis=-1)
    hist = np.bincount(differences[:, bg > 0].ravel(), minlength=256)
    return {"pairs": result, "background_diff_hist": hist.tolist()}
