# Pupil-crop tracking: development handoff

September 30, 2026. The current fixed-crop blink-filter performance is accepted by the project owner for continued development. Independent-recording evaluation is deferred because no additional paired recordings are available. This is a development decision, not evidence of generalization or calibrated gaze accuracy.

## What is implemented

The optional tracker finds a pupil in the existing broad eye region, then tries a smaller region around its last good measurement on the next frame. A region of interest (ROI) is simply a rectangle of image pixels. Left and right eyes have separate histories. No model weights were changed.

One new runtime module, `pupil_roi.py`, contains the tracking logic. Existing files keep their roles:

| File | Responsibility |
|---|---|
| `pupil_roi.py` | Predict a crop, keep a safety margin, reset after loss/blinks/time gaps, retry broad detection once when needed. |
| `pupil_detection.py` | Run segmentation on the selected crop, restore full-frame coordinates, and examine broad eye context for blink/boundary evidence. |
| `eye_pipeline.py` | Choose fixed/tracked mode, apply the blink gate before geometry/model updates, export crop and timing diagnostics. |
| `blink_validation.py` and `blink_review.html` | Run both baseline and blink-filter modes with the same crop setting and show saved results. |
| `blink_scoring.py` | Compare matching inputs while allowing measured processing times to differ. |
| `test_pupil_detection.py` | Regression and synthetic tracking tests. |

The tracker does not infer a blink merely because a narrow search failed. It retries the broad eye region in that same frame. It refreshes broadly at least every 0.5 seconds and stays broad during recovery from an unusable observation. Missing, stale or rejected measurements are never reused as current pupil measurements.

## Why the coordinate and image-context details matter

A pupil found at pixel (20, 30) inside a crop beginning at (100, 200) belongs at (120, 230) in the full image. The detector performs this translation once, before lens correction. Tracking always uses raw rotated-frame pixels; corrected 3D coordinates cannot be used to crop an image.

A small crop can hide the eyelid and make a partial pupil look complete. Therefore segmentation uses the small crop, but blink/boundary evidence retains the original broad eye region. A contour clipped by the tracking crop triggers a broad retry. The allowed pupil area still uses the broad region as its reference, so a smaller crop does not silently change area thresholds.

The fit is checked for containment with a margin. This is a useful fallback trigger, not proof that an occluded real pupil is fully visible.

## Order of processing

1. Decode and rotate the two eye images.
2. Locate each pupil using the selected crop strategy; retry broadly if needed.
3. Apply image-quality, blink and recovery checks independently to each eye.
4. Only eligible observations undergo lens correction and geometric checks.
5. Only observations that pass those checks update pye3d. Rejections produce explicit skipped output.
6. Evaluate output reliability and the separate robot-command gate after both eyes. Offline recordings never authorize a robot command.

Pupil localization must run before these image-based blink checks can inspect the pupil. The important ordering is that rejection happens before lens correction, 3D learning and downstream robot use.

## How to use it

1. Close the old review server with Control-C in its terminal, then double-click `Blink Review.command` again. Refresh the browser or use the newly printed address.
2. In **Try settings**, choose **Pupil search**: fixed is the accepted development mode; tracked is the optional experiment.
3. Keep minimum confidence 0.7, recovery confidence 0.7, recovery duration 0.05 seconds and frame limit 0 for the comparison described here.
4. Run baseline + blink filter. In **Read results**, choose that new saved run and score the same reviewer labels. Changing settings alone never rewrites an old run.
5. Compare fixed versus tracked reports using the same label revision. Do not treat the baseline column as “fixed”: it means blink-specific rejection/recovery is bypassed, with the selected crop strategy applied to both columns.

Normal `eye_pipeline.py` runs keep fixed crops. Its minimum-confidence default now matches the accepted 0.7 profile. To explicitly try tracking from a terminal in the project folder:

```sh
.venv/bin/python eye_pipeline.py --roi-mode tracked
```

## Measurements and limits

Each exported eye record includes `roi_tracking`: requested and actual crop, broad search region, fallback reason, number of neural calls and crop area divided by broad area. `frame_pair_timing_ms` measures decode/rotation, detection, quality/geometry and model diagnostics. The same pair duration appears in both eye records; count it once. These stage durations exclude subsequent export/display; full-run summaries include wider overhead.

TinyUNet still receives a 320 by 192 tensor in both modes. A smaller source crop does not itself reduce neural computation, and broad retries can increase runtime. Speed improvements must be measured, not inferred from crop size. Tracking also changes the pupil's apparent scale to the network; its filtering performance may differ from fixed crops.

The implementation and current recordings can test code behavior and development tradeoffs. Unseen-recording performance, true gaze accuracy and live robotic movement remain separate, deferred work. Future speed work should begin with the measured bottleneck and preserve the accepted fixed-crop mode as a regression reference.

## Completed full-video validation

Both strategies completed baseline and filtered passes of all 1,267 paired frames: 2,534 eye records per pass, 10,136 records across the four completed passes. Both baseline/filter pairs passed exact input compatibility checks. Reviewer ryan's frozen revision 412 contains 285 reviewed frames: 97 unusable, 176 usable, and 12 uncertain excluded from mistake rates.

| Filtered result | Fixed crop | Tracked crop |
|---|---:|---:|
| Unusable frames admitted | 20/97 (20.6%) | 30/97 (30.9%) |
| Usable frames rejected | 11/176 (6.25%) | 11/176 (6.25%) |
| Smaller crop actually used | 0/2,534 | 1,872/2,534 |
| Mean final crop area relative to broad ROI | 100% | 52.8% |
| Same-frame broad retries | 0 | 153 |
| Mean detection time per frame pair | 109.8 ms | 113.0 ms |
| Whole processing loop | 157.35 s | 161.34 s |
| Offline robot commands authorized | 0 | 0 |

**Decision: fixed remains default.** The tracker is implemented and tested but did not improve this development comparison: ten more labeled unusable observations were admitted with no improvement in good-frame retention. Smaller crops change segmentation, so tracking is not interchangeable with the accepted detector. The observed timing is one CPU run per strategy, not a repeated benchmark; no speed improvement is claimed. Average final crop area excludes the extra area processed during a narrow attempt followed by broad retry.

Run IDs visible in Read results:
- Fixed: `20260930T142200_2a3882`
- Tracked: `20260930T142720_cc10ee`

Each saved run includes the report, label snapshot, source snapshot and `roi_comparison_analysis.json`. The tracked filtered pass was interrupted once and restarted from frame zero with a fresh model. Its incomplete outputs are retained separately with `.interrupted` suffixes and excluded from scoring.

The fixed run reproduced all 5,068 accept/skip decisions from the September 28 baseline/filter comparison. New scores reflect the additional human labels. There are 197 passing regression tests, including crop containment/coordinates, broad retries, per-eye independence, recovery and time-gap handling, broad blink context, physical area thresholds, invalid settings, input compatibility and early blink/robot gate ordering. A browser-script smoke check also verifies the selected tracking mode reaches the run request and mistake navigation remains functional.

The blink-filter development handoff and optional tracker implementation are complete for this phase. A successful speed optimization is still future work: retain this rejected-for-default comparison as evidence, profile the detector, and test any subsequent change against the same accepted reference. Fresh data is deferred rather than invented or relabeled as independent validation.
