"""Experimental single-camera/multiple-LED reconstruction (Guestrin 2006 II-B).

Known LED identities and positions, corneal radius R, pupil offset K and index n
are REQUIRED. No population values or LED coordinates are silently substituted.
This returns an OPTICAL axis, not calibrated visual gaze or robot coordinates.
Uses NumPy/OpenCV only; the nonlinear unknown is a bounded corneal depth.
"""
from pathlib import Path
import argparse
import json
import math

import cv2
import numpy as np


def unit(vector):
    """Normalize a direction, rejecting zero/nonfinite values instead of NaNs."""
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector, axis=-1, keepdims=True)
    if not np.all(np.isfinite(vector)) or np.any(length <= 1e-12):
        raise ValueError("invalid direction")
    return vector/length


def sphere_hit(direction, center, radius):
    """Nearest positive intersection of camera ray t*direction with a sphere.

    Camera is at (0,0,0); direction must be unit length. The near surface is
    physically visible. A missed/tangent sphere does not provide a usable normal.
    Supports a single ray or an N-by-3 batch.
    """
    b = np.asarray(direction) @ center
    disc = b*b-(np.dot(center, center)-radius*radius)
    if np.any(disc <= 0):
        raise ValueError("ray misses or grazes corneal sphere")
    distance = b-np.sqrt(disc)
    if np.any(distance <= 0):
        raise ValueError("surface is behind camera or camera is inside sphere")
    return np.asarray(direction)*np.expand_dims(distance, -1)


def reconstruct(glint_rays, led_positions, pupil_ray, *, radius_mm,
                pupil_offset_mm, refractive_index, depth_bounds_mm,
                max_reflection_error_deg=2.0):
    """Return centers in mm and a unit optical direction in camera XYZ.

    glint_rays[i] and led_positions[i] MUST describe the same physical LED.
    Raw optical camera frame: +X right, +Y down, +Z forward into camera scene.
    R = corneal radius; K = distance from corneal center to physical pupil.
    depth_bounds_mm bounds camera-to-corneal-center DISTANCE along a ray.
    The residual cutoff is an engineering gate, not an accuracy guarantee.
    """
    fail = lambda reason: dict(valid=False, reason=reason, optical_axis=None,
                              visual_gaze_available=False, robot_allowed=False)
    rays = np.asarray(glint_rays, dtype=float)
    leds = np.asarray(led_positions, dtype=float)
    if rays.ndim != 2 or rays.shape[1:] != (3,) or len(rays) < 2 or leds.shape != rays.shape:
        return fail("need at least two matched 3D LED/ray pairs")
    try:
        rays, pupil = unit(rays), unit(pupil_ray)
        if pupil.shape != (3,) or np.any(rays[:, 2] <= 0) or pupil[2] <= 0:
            return fail("rays must point forward from camera")
        lo, hi = map(float, depth_bounds_mm)
        values = [radius_mm, pupil_offset_mm, refractive_index, lo, hi, max_reflection_error_deg]
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(leds)):
            return fail("nonfinite geometry/settings")
        if not (0 < pupil_offset_mm < radius_mm < lo < hi and refractive_index > 1
                and max_reflection_error_deg > 0):
            return fail("require 0 < K < R < distance lower < upper, n > 1")

        # The camera, LED, glint ray and corneal center share a plane.
        # Its normal is LED x ray. SVD finds the direction closest to all planes;
        # two independent plane normals are necessary to constrain that direction.
        normals = unit(np.cross(leds, rays))
        _, singular, vt = np.linalg.svd(normals, full_matrices=True)
        if singular[1]/singular[0] < 1e-3:
            return fail("degenerate LED/glint planes")
        axis = vt[-1]
        if axis[2] < 0:
            axis = -axis

        def residual(distance):
            center = axis*distance
            try:
                surface = sphere_hit(rays, center, radius_mm)
                normal = (surface-center)/radius_mm
                to_led = unit(leds-surface)
                to_camera = -rays
                if np.any(np.sum(normal*to_led, axis=1) <= 0):
                    return math.inf
                # Reflection normal bisects the directions toward light/camera.
                error = normal-unit(to_led+to_camera)
                return float(np.mean(np.sum(error*error, axis=1)))
            except ValueError:
                return math.inf

        # A coarse bounded scan gives a bracket without a fragile eye-center
        # guess. Golden-section refinement needs only scalar function evaluations.
        grid = np.linspace(lo, hi, 257)
        losses = np.array([residual(distance) for distance in grid])
        best = int(np.argmin(losses))
        if not np.isfinite(losses[best]):
            return fail("no feasible corneal sphere within distance bounds")
        if best in (0, len(grid)-1):
            return fail("corneal solution reaches distance bound; check geometry")
        a, b = grid[best-1], grid[best+1]
        ratio = (np.sqrt(5)-1)/2
        c, d = b-ratio*(b-a), a+ratio*(b-a)
        fc, fd = residual(c), residual(d)
        for _ in range(64):
            if b-a < 1e-8:
                break
            if fc < fd:
                b, d, fd = d, c, fc
                c = b-ratio*(b-a)
                fc = residual(c)
            else:
                a, c, fc = c, d, fd
                d = a+ratio*(b-a)
                fd = residual(d)
        distance = (a+b)/2
        center = axis*distance
        surface = sphere_hit(rays, center, radius_mm)
        normal = (surface-center)/radius_mm
        # Reflect incoming LED light; compare it with the actual camera ray.
        incoming = unit(surface-leds)
        predicted = incoming-2*np.sum(incoming*normal, axis=1)[:, None]*normal
        errors = np.degrees(np.arccos(np.clip(np.sum(predicted*(-rays), axis=1), -1, 1)))
        if float(errors.max()) > max_reflection_error_deg:
            return fail("reflection residual too large; check matching/calibration")

        q = sphere_hit(pupil, center, radius_mm)
        outward_normal = (q-center)/radius_mm
        cos_incident = -float(np.dot(outward_normal, pupil))
        eta = 1/refractive_index  # Back-trace light: air -> effective eye medium.
        transmitted = unit(eta*pupil+(eta*cos_incident-np.sqrt(
            1-eta*eta*(1-cos_incident*cos_incident)))*outward_normal)
        # Snell's law gives a direction, NOT pupil depth. Intersect this refracted
        # ray with the K-radius sphere about C and select its first positive hit.
        delta = q-center
        bq = float(np.dot(delta, transmitted))
        discriminant = bq*bq-(np.dot(delta, delta)-pupil_offset_mm**2)
        if discriminant <= 0:
            return fail("refracted pupil ray misses anatomical K sphere")
        steps = [-bq-np.sqrt(discriminant), -bq+np.sqrt(discriminant)]
        positive = [step for step in steps if step > 0]
        if not positive:
            return fail("pupil lies behind refracted ray")
        point = q+min(positive)*transmitted
        return dict(valid=True, reason="assumed-eye-model optical axis reconstructed",
                    corneal_center_mm=center.tolist(), pupil_center_mm=point.tolist(),
                    optical_axis=unit(point-center).tolist(),
                    reflection_error_deg=errors.tolist(),
                    corneal_distance_mm=float(distance),
                    visual_gaze_available=False, robot_allowed=False)
    except (ValueError, TypeError, np.linalg.LinAlgError) as error:
        return fail(str(error))


def template():
    """Missing measurements stay null so placeholders cannot produce real gaze."""
    return dict(coordinate_system="raw_calibration_camera_xyz_mm",
                calibration_path="left_camera_calibration.npz", camera_id="left-eye-camera",
                calibration_rotation="none", corneal_radius_mm=None,
                pupil_offset_mm=None, refractive_index=None,
                corneal_distance_bounds_mm=[None, None], max_reflection_error_deg=2.0,
                leds=[dict(id=f"LED{i}", position_mm=None) for i in range(1, 7)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-template", type=Path)
    parser.add_argument("--detections", type=Path, help="detections.jsonl from the detector")
    parser.add_argument("--make-matches", type=Path, help="Write null per-frame LED assignments for editing")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--matches", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-unverified-calibration", action="store_true",
                        help="Explicit exploratory use of legacy calibration; does not establish accuracy")
    args = parser.parse_args(argv)
    if args.write_template:
        with args.write_template.open("x") as stream:
            json.dump(template(), stream, indent=2)
        return
    if args.detections is None:
        parser.error("provide --detections or --write-template")
    from glint_detection import file_sha256
    if args.make_matches:
        matches = {}
        with args.detections.open() as stream:
            for line in stream:
                record = json.loads(line)
                if record["glints"]["candidates"]:
                    matches[str(record["frame_index"])] = {str(c["candidate_id"]): None for c in record["glints"]["candidates"]}
        with args.make_matches.open("x") as stream:
            json.dump(dict(detections_sha256=file_sha256(args.detections), frames=matches), stream, indent=2)
        return
    if any(value is None for value in (args.config, args.matches, args.output_dir)):
        parser.error("reconstruction requires --config, --matches and --output-dir")
    config = json.loads(args.config.read_text())
    if config["coordinate_system"] != "raw_calibration_camera_xyz_mm":
        parser.error("LEDs must be in the raw calibration camera coordinate system, in mm")
    required = [config.get(k) for k in ("corneal_radius_mm", "pupil_offset_mm", "refractive_index")]
    bounds = config.get("corneal_distance_bounds_mm", [None, None])
    if any(value is None for value in required+bounds):
        parser.error("fill the eye parameters and distance bounds; placeholders cannot be used")
    leds = {}
    for led in config["leds"]:
        if led["position_mm"] is None:
            parser.error("fill measured LED positions; placeholders cannot be used")
        value = np.asarray(led["position_mm"], dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)) or led["id"] in leds:
            parser.error("LED IDs must be unique and positions finite XYZ triplets")
        leds[led["id"]] = value
    assignments = json.loads(args.matches.read_text())
    if assignments["detections_sha256"] != file_sha256(args.detections):
        parser.error("matches belong to a different detection run")
    metadata = json.loads((args.detections.parent/"metadata.json").read_text())
    from calibration import CameraCalibration
    calibration_path = (args.config.parent/config["calibration_path"]).resolve()
    calibration = CameraCalibration.load(calibration_path,
        frame_rotation=metadata["rotation"], calibration_rotation=config["calibration_rotation"],
        expected_camera_id=config["camera_id"], require_identity=not args.allow_unverified_calibration)
    calibration.validate_video_dimensions(tuple(metadata["source_size"]),
                                          tuple(metadata["processed_size"]), metadata["eye"])

    def rays_from_pixels(points):
        # Detection coordinates include rotation and crop translation already.
        # Undo rotation BEFORE undistortion; normalized output becomes raw-camera
        # rays, the same frame in which the LED positions were measured.
        raw = calibration.processed_to_raw_points(np.asarray(points, dtype=float))
        normalized = cv2.undistortPoints(raw.reshape(-1, 1, 2),
                        calibration.raw_camera_matrix, calibration.distortion_coefficients).reshape(-1, 2)
        return unit(np.column_stack((normalized, np.ones(len(normalized)))))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = dict(config=config, calibration_sha256=file_sha256(calibration_path),
                      calibration_identity_status=calibration.identity_status,
                      detections_sha256=file_sha256(args.detections), matches_sha256=file_sha256(args.matches),
                      geometry_source_sha256=file_sha256(__file__),
                      coordinate_system=config["coordinate_system"], visual_gaze_available=False)
    (args.output_dir/"metadata.json").write_text(json.dumps(provenance, indent=2)+"\n")
    frames, valid = 0, 0
    with args.detections.open() as source, (args.output_dir/"optical_axes.jsonl").open("x") as output:
        for line in source:
            record = json.loads(line)
            outcome = dict(valid=False, reason="blink gate or missing LED assignments", optical_axis=None,
                           visual_gaze_available=False, robot_allowed=False)
            if record["quality"]["allow_model_update"] and record["quality"]["state"] == "usable":
                pairs = assignments["frames"].get(str(record["frame_index"]), {})
                candidates = {str(c["candidate_id"]): c for c in record["glints"]["candidates"]}
                matched = [(cid, lid) for cid, lid in pairs.items() if lid is not None]
                if any(cid not in candidates or lid not in leds for cid, lid in matched):
                    raise ValueError("match references an unknown candidate or LED")
                if len(set(lid for _, lid in matched)) != len(matched):
                    raise ValueError("a physical LED cannot be assigned twice in one frame")
                if len(matched) >= 2:
                    all_rays = rays_from_pixels([candidates[cid]["center_px"] for cid, _ in matched]
                                               + [record["pupil_ellipse"][0]])
                    outcome = reconstruct(all_rays[:-1], [leds[lid] for _, lid in matched], all_rays[-1],
                        radius_mm=required[0], pupil_offset_mm=required[1], refractive_index=required[2],
                        depth_bounds_mm=bounds, max_reflection_error_deg=config["max_reflection_error_deg"])
            frames += 1
            valid += int(outcome["valid"])
            output.write(json.dumps(dict(frame_index=record["frame_index"], eye=record["eye"],
                                         timestamp_s=record["timestamp_s"], **outcome), allow_nan=False)+"\n")
    (args.output_dir/"summary.json").write_text(json.dumps(dict(frames=frames, reconstructed=valid,
        status="complete", scope="optical-axis prototype; no visual-axis calibration or robot output"), indent=2)+"\n")


if __name__ == "__main__":
    main()
