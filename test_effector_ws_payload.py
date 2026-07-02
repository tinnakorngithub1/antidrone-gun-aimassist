#!/usr/bin/env python3
"""Local smoke tests for effector_ws_exporter.build_effector_payload (no hardware)."""

import time

from effector_ws_exporter import (
    MODE_AUTO,
    MODE_LOCK,
    MODE_MANUAL,
    build_effector_payload,
    format_target_id,
)


def _geo():
    return {"site_lat": 14.29, "site_lng": 105.13, "heading_deg": 90.0}


def _cam8():
    return {"site_lat": 14.291, "site_lng": 105.131, "heading_deg": 100.0}


def _cue(cx=2560.0, obj_id=5, dist=120.0):
    return {
        "cx": cx,
        "cy": 720.0,
        "frame_w": 5120,
        "frame_h": 1440,
        "target_id": str(obj_id),
        "distance_m": dist,
        "recv_timestamp": time.time(),
        "cue_ttl_ms": 500,
        "source_camera": "cam8",
    }


def _base(**kw):
    d = dict(
        pos_x=2.5,
        pos_y=3.6,
        home_arm_pan=2.511,
        last_target_pan=None,
        last_target_tilt=None,
        ready_to_fire=False,
        arm_mode=MODE_AUTO,
        latest_cue=None,
        target_det=None,
        distance_m=None,
        lock_csrt_initialized=False,
        lock_csrt_lost=False,
        lock_csrt_smooth_px=None,
        lock_csrt_smooth_py=None,
        cx_frame=320,
        cy_frame=240,
        w=640,
        h=480,
        px_per_deg_x=10.0,
        px_per_deg_y=10.0,
        cue_ttl_ms=500,
        effector_geo=_geo(),
        cam8_geo=_cam8(),
        fov_deg=30.0,
        cam8_fov_h=180.0,
    )
    d.update(kw)
    return build_effector_payload(**d)


def test_searching_null_targets():
    p = _base(arm_mode=MODE_MANUAL)
    assert p["lock_state"] == "searching"
    assert p["lock_progress"] == 0.0
    assert p["target_id"] is None
    assert p["target_lat"] is None
    assert p["target_lng"] is None


def test_ready_locked():
    p = _base(ready_to_fire=True, latest_cue=_cue(), distance_m=100.0)
    assert p["lock_state"] == "locked"
    assert p["lock_progress"] == 1.0


def test_target_id_format():
    assert format_target_id(5) == "uav_005"
    assert format_target_id("42") == "uav_042"


def test_auto_cue_acquiring():
    cue = _cue(cx=2560.0)
    p = _base(arm_mode=MODE_AUTO, latest_cue=cue, distance_m=100.0, pos_x=2.511)
    assert p["lock_progress"] > 0
    assert p["lock_state"] in ("acquiring", "locked")
    assert p["target_id"] == "uav_005"
    assert p["target_lat"] is not None


def test_auto_fallback_last_target_pan():
    p = _base(arm_mode=MODE_AUTO, last_target_pan=2.511, pos_x=2.511)
    assert p["lock_progress"] == 1.0
    assert p["lock_state"] == "acquiring"


def test_lock_reticle_progress():
    p = _base(
        arm_mode=MODE_LOCK,
        lock_csrt_initialized=True,
        lock_csrt_smooth_px=321.0,
        lock_csrt_smooth_py=240.0,
        distance_m=80.0,
    )
    assert p["lock_progress"] > 0
    assert p["target_lat"] is not None


def test_phase2_cue_latlng():
    cue = _cue()
    cue["target_lat"] = 14.292
    cue["target_lng"] = 105.133
    p = _base(arm_mode=MODE_AUTO, latest_cue=cue, distance_m=100.0, pos_x=2.511)
    assert p["target_lat"] == 14.292
    assert p["target_lng"] == 105.133


def main():
    test_searching_null_targets()
    test_ready_locked()
    test_target_id_format()
    test_auto_cue_acquiring()
    test_auto_fallback_last_target_pan()
    test_lock_reticle_progress()
    test_phase2_cue_latlng()
    print("All effector payload tests passed.")


if __name__ == "__main__":
    main()
