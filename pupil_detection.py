"""Segment a pupil inside an eye ROI and fit an OpenCV ellipse to its mask.

The detector operates on the rotated video frame before lens undistortion.
It returns an ellipse in that frame's pixel coordinates; calibration.py handles
the later conversion to undistorted coordinates for the 3D eye model. An
experimental, stateless eyelid analysis adds diagnostics without changing pupil
measurements. ``assess_pupil_quality`` proposes single-frame accept/skip decisions;
``TemporalQualityTracker`` adds per-eye recovery history. The pipeline uses
these decisions to gate each eye's pye3d updates.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.torch_version import TorchVersion

# Keep these names available here so existing scripts and notebooks still work.
# Their implementations now live together in pupil_quality.py.
from pupil_quality import (
    BoundaryEvidence, EyelidObservation, PupilObservation, FrameQualityDecision,
    assess_pupil_quality, TemporalQualityTracker,
)


class _ConvBlock(nn.Module):
    """Two convolutions that change channel count while preserving image size."""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class TinyUNet(nn.Module):
    """Single-channel U-Net producing pupil logits with shape (N, 1, H, W).

    H and W must be divisible by eight so the three downsampling stages and
    skip connections line up. Layer names and base_channels must match the
    checkpoint's state dictionary when changing or retraining this model.
    """

    def __init__(self, base_channels=16):
        super().__init__()
        channels = int(base_channels)
        self.encoder1 = _ConvBlock(1, channels)
        self.encoder2 = _ConvBlock(channels, channels * 2)
        self.encoder3 = _ConvBlock(channels * 2, channels * 4)
        self.bottleneck = _ConvBlock(channels * 4, channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(channels * 8, channels * 4, 2, stride=2)
        self.decoder3 = _ConvBlock(channels * 8, channels * 4)
        self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, 2, stride=2)
        self.decoder2 = _ConvBlock(channels * 4, channels * 2)
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.decoder1 = _ConvBlock(channels * 2, channels)
        self.output = nn.Conv2d(channels, 1, 1)

    def forward(self, inputs):
        """Decode grayscale images using encoder features at matching scales."""
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        bottleneck = self.bottleneck(self.pool(encoder3))
        # Concatenate along channels (dimension 1), preserving batch/H/W axes.
        decoder3 = self.decoder3(torch.cat((self.up3(bottleneck), encoder3), 1))
        decoder2 = self.decoder2(torch.cat((self.up2(decoder3), encoder2), 1))
        decoder1 = self.decoder1(torch.cat((self.up1(decoder2), encoder1), 1))
        return self.output(decoder1)


def detect_eyelid_closure(roi, local_ellipse=None, pupil_confidence=0.0,
                         roi_origin=(0, 0)):
    """Find conservative upper-lid/lash evidence in an upright grayscale/BGR ROI.

    Inputs:
        roi: The cropped eye image, not the entire camera frame. Expected shape
            is (height, width) for grayscale or (height, width, 3) for BGR, with
            uint8 brightness values from 0 to 255. The eye should be upright
            in a landscape crop. The input image is never changed.
        local_ellipse: Optional pupil ellipse in this CROP's pixels:
            ((center_x, center_y), (diameter_a, diameter_b), angle_degrees).
            The axes are full diameters, not radii. None means no fitted pupil.
        pupil_confidence: Segmentation/shape score in [0, 1], used to decide
            whether the supplied ellipse is reliable enough to guide searching.
        roi_origin: (x, y) location of the crop's top-left corner in the full
            rotated frame. This offset is used only for output coordinates.

    Returns:
        EyelidObservation with the candidate state, edge samples, support and
        reason. If roi_origin=(0, 100), a sample at crop (120, 80) is returned
        as full-frame (120, 180). Unsupported views return "unknown".

    This first prototype uses a bright-to-dark transition shared by many image
    columns. The pupil's own edge is excluded from the support calculation when
    a reliable ellipse is available. Local edge positions allow modest lid
    curvature; returned samples show the evidence without bridging the pupil.

    With a visible pupil, a supported boundary inside its upper portion suggests
    occlusion. Without a reliable pupil, a broad low boundary is only a closure
    candidate for the supplied fixed eye crops. Crop shifts, camera roll, lashes
    and downward gaze can confuse this heuristic. Missing pupils alone remain
    unknown. The 0.60 pupil-reliability cutoff here is a diagnostic setting,
    independent of the 3D model's acceptance threshold. No temporal state,
    trained eyelid model or anatomical gap estimate is implied. The raw eyelid
    evidence is displayed diagnostically; the downstream quality decision is gated.
    """
    if roi.dtype != np.uint8:
        raise ValueError("eyelid ROI must contain uint8 pixels")
    if roi.ndim == 2:
        gray = roi
    elif roi.ndim == 3 and roi.shape[2] == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("eyelid ROI must be grayscale or BGR")
    height, width = gray.shape  # Arrays use (rows, columns), i.e. (height, width).
    if min(height, width) < 32:
        return EyelidObservation(reason="eye crop too small")
    if width < height or width > 4 * height:
        return EyelidObservation(reason="expected a landscape eye crop")

    # Shrink the longest dimension to at most 320 pixels, keeping the crop's
    # aspect ratio. INTER_AREA averages source pixels when shrinking. This is
    # a separate analysis copy: TinyUNet's resizing/preprocessing is unchanged.
    scale = min(1.0, 320.0 / max(height, width))
    gray = cv2.resize(gray, (round(width * scale), round(height * scale)),
                      interpolation=cv2.INTER_AREA)
    # A 5x5 Gaussian neighborhood smooths fine texture. Floating-point values
    # are essential for signed subtraction: uint8 cannot represent negatives.
    gray = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    h, w = gray.shape
    # Rounding resized dimensions can make the true x/y scale slightly unequal.
    scale_x, scale_y = w / width, h / height
    # Percentiles ignore the darkest/brightest extremes. A narrow middle range
    # means little usable contrast; isolated extreme pixels should not rescue it.
    if float(np.percentile(gray, 95) - np.percentile(gray, 5)) < 30:
        return EyelidObservation(reason="insufficient image contrast")

    gap = max(2, round(h * 0.02))  # Distance above/below a candidate edge, in pixels.
    tolerance = max(2, round(h * 0.025))  # Nearby rows allowed for modest curvature.
    # These slices compare rows 2*gap apart. contrast[i, x] represents an edge
    # centered at analysis row i+gap. For example, brightness 200 above and 50
    # below gives +150: a bright-to-dark transition like a lid/lash margin.
    contrast = gray[:-2 * gap] - gray[2 * gap:]
    # On this floating-point contrast map, dilation takes a local MAXIMUM.
    # The tall, one-column kernel searches only vertically. This tolerates an
    # edge curving into nearby rows; it does not erode or modify the eye image.
    nearby = cv2.dilate(contrast, np.ones((2 * tolerance + 1, 1), np.uint8))
    # Search the central 80% of crop width to reduce side-border distractions.
    columns = np.arange(round(0.1 * w), round(0.9 * w))
    reliable_pupil = local_ellipse is not None and pupil_confidence >= 0.60
    pupil_top = pupil_cy = radius_y = None
    if reliable_pupil:
        (cx, cy), (axis_a, axis_b), angle = local_ellipse
        theta = np.deg2rad(angle)  # NumPy trig functions take radians, not degrees.
        # These are horizontal/vertical half-extents of a ROTATED ellipse.
        # hypot(u, v) means sqrt(u*u + v*v). Divide the full axis lengths by two
        # to obtain radii; the angle mixes each axis into x and y extents.
        radius_x = np.hypot(axis_a * np.cos(theta), axis_b * np.sin(theta)) / 2
        radius_y = np.hypot(axis_a * np.sin(theta), axis_b * np.cos(theta)) / 2
        if (cx - radius_x <= 0 or cx + radius_x >= width
                or cy - radius_y <= 0 or cy + radius_y >= height):
            return EyelidObservation(reason="pupil touches crop boundary")
        # Require evidence on BOTH sides, outside the pupil's horizontal extent.
        # The 1.1 multiplier adds a 10% margin around that extent. Boolean array
        # indexing keeps only columns satisfying each left/right condition.
        left_columns = columns[columns < (cx - 1.1 * radius_x) * scale_x]
        right_columns = columns[columns > (cx + 1.1 * radius_x) * scale_x]
        if min(len(left_columns), len(right_columns)) < 0.08 * w:
            return EyelidObservation(reason="insufficient view beside pupil")
        # Concatenate joins the two lists without including columns over the pupil.
        columns = np.concatenate((left_columns, right_columns))
        # From here, comparisons use the smaller analysis image's pixel units.
        pupil_top = (cy - radius_y) * scale_y
        pupil_cy = cy * scale_y
        radius_y *= scale_y

    # ':' selects every row; 'columns' selects just the search columns. Each
    # True means at least a 20-level brightness drop was found near that row.
    supported = nearby[:, columns] >= 20.0
    # mean(axis=1) works across columns: 65 True values out of 100 gives 0.65.
    support = supported.mean(axis=1)
    # Rank rows using typical edge strength times its coverage. This ranking
    # score is neither pupil confidence nor a calibrated blink probability.
    score = np.median(nearby[:, columns], axis=1) * support
    # Restore the gap offset from the paired-row subtraction above.
    rows = np.arange(len(score)) + gap
    # '&' combines element-by-element Boolean conditions. These fractions and
    # intensity cutoffs are prototype settings, not universal anatomy constants.
    eligible = (support >= 0.65) & (rows >= 0.10 * h) & (rows <= 0.88 * h)
    # Searching nearby rows can otherwise cherry-pick unrelated noise peaks.
    # Require a signed transition shared at the same row as well as local support.
    eligible &= np.median(contrast[:, columns], axis=1) >= 20.0
    if reliable_pupil:
        # Each flank must contribute, preventing a pupil or lash cluster on one
        # side from masquerading as an eyelid spanning the eye.
        eligible &= (nearby[:, left_columns] >= 20).mean(axis=1) >= 0.55
        eligible &= (nearby[:, right_columns] >= 20).mean(axis=1) >= 0.55
        # '&=' keeps earlier constraints and adds this search window: around
        # the pupil's top, down to its center. The lower pupil edge is excluded.
        eligible &= (rows >= pupil_top - 0.25 * radius_y) & (rows <= pupil_cy)
    if not np.any(eligible):
        return EyelidObservation(reason="no broad supported edge near pupil" if reliable_pupil
                                 else "no broad supported eyelid edge")
    # where gives invalid rows a score of negative infinity so they cannot win.
    # argmax returns the winning ARRAY INDEX, not its score or full-frame y.
    index = int(np.argmax(np.where(eligible, score, -np.inf)))

    # Recover actual local edge rows, rather than drawing the dilated search row.
    # Clip the search window to available rows. Python slice 'stop' is exclusive.
    start, stop = max(0, index - tolerance), min(len(contrast), index + tolerance + 1)
    # axis=0 searches down rows separately for each column. Add 'start' because
    # argmax indexes the sliced window, then 'gap' to restore analysis-image y.
    edge_rows = np.argmax(contrast[start:stop, columns], axis=0) + start + gap
    edge_strength = np.max(contrast[start:stop, columns], axis=0)
    valid = edge_strength >= 20
    # A typical supported edge height is robust to a few outlying lash samples.
    # It remains an approximate row, not a fitted eyelid curve over the pupil.
    measured_y = float(np.median(edge_rows[valid]))
    ox, oy = roi_origin
    points = []
    # Draw only up to eight representative samples rather than every column.
    # 'band' contains indices INTO columns, not the image x coordinates directly.
    for band in np.array_split(np.arange(len(columns)), 8):
        accepted = band[valid[band]]  # Keep samples with enough actual contrast.
        if len(accepted):
            # Choose an actual measured column; an average could land inside
            # the excluded pupil, where no edge evidence was collected.
            chosen = accepted[len(accepted) // 2]
            # Undo shrinking first, then add the crop origin for overlay drawing.
            points.append((float(columns[chosen]) / scale_x + ox,
                           float(edge_rows[chosen]) / scale_y + oy))
    if reliable_pupil:
        # Image y increases downward. This small margin asks for an edge inside
        # the pupil's upper region, not merely one approximately level with it.
        overlaps = measured_y > pupil_top + 0.08 * radius_y
        state = "occlusion_possible" if overlaps else "no_closure_evidence"
        reason = ("lid-like edge overlaps upper pupil" if overlaps
                  else "edge above measured pupil")
    elif measured_y >= 0.55 * h:
        # Without a usable pupil, a low broad edge is only a closure candidate.
        # Its position depends on framing, so this is deliberately not 'closed'.
        state, reason = "closed_possible", "low lid-like boundary; pupil unreliable"
    else:
        state, reason = "unknown", "edge found but pupil visibility unresolved"
    return EyelidObservation(state, tuple(points), float(support[index]), reason)


def pupil_boundary_evidence(roi, contour, ellipse):
    """Assess local pupil boundaries without modifying the image or fitted oval.

    Near-border contours are checked against RAW image corridors to the crop
    edge: a dark connected corridor plus insufficient iris separation supports
    clipping even if segmentation retreats a few pixels from the edge.
    Paired ellipse samples outside the image are excluded, never zero-padded.
    Brightness asymmetry alone is a warning; rejection additionally requires
    weakened upper-boundary contrast. Thresholds remain development heuristics.
    """
    height, width = roi.shape[:2]
    points = np.asarray(contour).reshape(-1, 2)
    margin = float(min(points[:, 0].min(), width-1-points[:, 0].max(),
                       points[:, 1].min(), height-1-points[:, 1].max()))
    if margin <= 0:
        return BoundaryEvidence("rejected", "pupil contour clipped by ROI boundary", margin)
    (cx, cy), (a, b), angle = ellipse
    theta = np.linspace(0, 2*np.pi, 96, endpoint=False)
    rotation = np.deg2rad(angle)
    dx = a/2*np.cos(theta)*np.cos(rotation)-b/2*np.sin(theta)*np.sin(rotation)
    dy = a/2*np.cos(theta)*np.sin(rotation)+b/2*np.sin(theta)*np.cos(rotation)
    source = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = cv2.GaussianBlur(source, (5, 5), 0).astype(np.float32)
    inside_mask = np.zeros((height, width), np.uint8)
    cv2.ellipse(inside_mask, ((cx, cy), (.8*a, .8*b), angle), 1, -1)
    interior = gray[inside_mask > 0]
    if len(interior) == 0:
        return BoundaryEvidence("unknown", "no pupil interior samples", margin)
    dark = float(np.median(interior))
    # A radius-relative neighborhood bounds the border search. Proximity alone
    # never rejects: require a connected dark corridor across a broad arc.
    reach = max(2., .12*min(a, b))
    connection = 0.
    for axis, edge in ((0, 0), (0, width-1), (1, 0), (1, height-1)):
        extent = width if axis == 0 else height
        near = np.abs(points[:, axis]-edge) <= reach
        candidates = points[near]
        if len(candidates) < 5:
            continue
        # Subsample to keep work bounded independently of camera resolution.
        candidates = candidates[::max(1, len(candidates)//64)]
        fractions = []
        for x, y in candidates:
            other = int(y if axis == 0 else x)
            end = int(x if axis == 0 else y)
            lo, hi = sorted((end, int(edge)))
            strip = gray[other, lo:hi+1] if axis == 0 else gray[lo:hi+1, other]
            fractions.append(bool(len(strip)) and bool(np.all(strip <= dark+15)))
        connection = max(connection, float(np.mean(fractions)))
    samples, valid = [], np.ones(96, dtype=bool)
    for scale in (.85, 1.15):
        x, y = cx+scale*dx, cy+scale*dy
        valid &= (x >= 0) & (x <= width-1) & (y >= 0) & (y <= height-1)
        samples.append(cv2.remap(gray, x.astype(np.float32)[None, :],
                                y.astype(np.float32)[None, :], cv2.INTER_LINEAR)[0])
    fraction = float(np.mean(valid))
    evidence = dict(contour_margin_px=margin, valid_fraction=fraction,
                    border_connection_fraction=connection)
    if connection >= .6 and float(np.percentile(gray, 90))-dark >= 30:
        return BoundaryEvidence("rejected", "dark pupil evidence reaches ROI boundary", **evidence)
    upper = (dy < -.35*np.max(np.abs(dy))) & valid
    lower = (dy > .35*np.max(np.abs(dy))) & valid
    # A few remaining pixels are not enough to characterize an entire arc.
    if min(np.count_nonzero(upper), np.count_nonzero(lower)) < 12:
        return BoundaryEvidence("unknown", "insufficient visible boundary samples", **evidence)
    contrast = samples[1]-samples[0]
    low = float(np.median(contrast[lower])); up = float(np.median(contrast[upper]))
    difference = float(np.median(samples[0][upper])-np.median(samples[0][lower]))
    weak = float(np.mean(contrast[upper] < 10))
    evidence.update(lower_contrast=low, upper_contrast=up,
                    upper_interior_difference=difference, weak_upper_fraction=weak)
    if low >= 15 and weak >= .55:
        return BoundaryEvidence("rejected", "possible upper pupil boundary coverage", **evidence)
    if low >= 20 and difference > 20:
        if up < .6*low:
            return BoundaryEvidence("rejected", "brightness asymmetry with weak upper boundary", **evidence)
        return BoundaryEvidence("warning", "brightness asymmetry without supporting boundary loss", **evidence)
    if fraction < .9 or low < 15:
        return BoundaryEvidence("unknown", "incomplete or low-contrast boundary evidence", **evidence)
    return BoundaryEvidence("clear", "boundary checks passed", **evidence)


def existing_boundary_quality(roi, contour, ellipse):
    """Return an optional rejection reason from crop and boundary evidence.

    All geometry is LOCAL to roi. A selected contour on the crop border is
    incomplete, even if its fitted ellipse looks plausible. For coverage, sample
    paired points just inside/outside the ellipse: an unobscured pupil should
    transition from dark pupil to brighter iris. A broadly unsupported upper arc
    with a supported lower arc suggests lid/lash coverage. This is a development
    heuristic, not an anatomical blink classifier; reflections and unusual iris
    appearance can affect it. Neither the pixels nor fitted ellipse are changed.
    """
    height, width = roi.shape[:2]
    points = np.asarray(contour).reshape(-1, 2)
    if np.any((points[:, 0] <= 0) | (points[:, 0] >= width - 1)
              | (points[:, 1] <= 0) | (points[:, 1] >= height - 1)):
        return "pupil contour clipped by ROI boundary"
    (cx, cy), (a, b), angle = ellipse
    # Uniform angular samples on the rotated ellipse; axes are diameters.
    theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    rotation = np.deg2rad(angle)
    dx = a / 2 * np.cos(theta) * np.cos(rotation) - b / 2 * np.sin(theta) * np.sin(rotation)
    dy = a / 2 * np.cos(theta) * np.sin(rotation) + b / 2 * np.sin(theta) * np.cos(rotation)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    samples = []
    for scale in (0.85, 1.15):
        x, y = cx + scale * dx, cy + scale * dy
        if np.any((x < 0) | (x > width - 1) | (y < 0) | (y > height - 1)):
            # Missing surrounding iris is insufficient evidence of closure.
            return None
        samples.append(cv2.remap(gray, x.astype(np.float32)[None, :],
                                y.astype(np.float32)[None, :], cv2.INTER_LINEAR)[0])
    contrast = samples[1] - samples[0]
    radius_y = np.max(np.abs(dy))
    upper, lower = dy < -0.35 * radius_y, dy > 0.35 * radius_y
    # Lashes/partial closure may leave a strong OUTER edge while contaminating
    # the fitted pupil's upper interior. Compare broad arcs, using medians to
    # avoid treating a few bright LED glints as broad coverage. The 20-level
    # difference is a development heuristic (8-bit images), not a probability.
    # Require an identifiable lower pupil/iris transition as supporting evidence.
    if (np.median(contrast[lower]) >= 20
            and np.median(samples[0][upper]) - np.median(samples[0][lower]) > 20):
        return "possible upper pupil contamination"
    # Require support below, so uniformly dark/low-contrast images cannot be
    # interpreted as a specific upper-lid obstruction by this rule.
    if (np.median(contrast[lower]) >= 15
            and np.mean(contrast[upper] < 10) >= 0.55):
        return "possible upper pupil boundary coverage"
    return None


def pupil_boundary_quality(roi, contour, ellipse, policy="existing"):
    """Select a learning gate; experimental evidence is not enabled by default.

    The corroborated/raw-border candidate regressed on development labels. Keep
    it available for research, while retaining the established input policy.
    Robot eligibility independently requires clear image evidence.
    """
    if policy == "existing":
        return existing_boundary_quality(roi, contour, ellipse)
    if policy != "experimental":
        raise ValueError("boundary policy must be existing or experimental")
    evidence = pupil_boundary_evidence(roi, contour, ellipse)
    return evidence.reason if evidence.status == "rejected" else None


def prepare_tinyunet_input(roi, input_size, device):
    """Convert an 8-bit grayscale/BGR crop to a float32 (1, 1, H, W) tensor.

    input_size follows OpenCV's (width, height) convention. Resizing stretches
    the crop to that size without padding; preprocessing must match training.
    """
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    elif roi.ndim == 2:
        gray = roi
    else:
        raise ValueError("pupil ROI must be a grayscale or BGR image")
    resized = cv2.resize(gray, input_size, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    # Add batch and grayscale-channel axes; normalize expected 0..255 input.
    return tensor[None, None].to(device)


def generate_probability_map(model, tensor, roi_size):
    """Return a CPU (ROI height, ROI width) probability map for one image.

    roi_size is (width, height). Resize before thresholding so contour geometry
    is measured in original ROI pixels rather than the model's smaller grid.
    The caller is responsible for setting the model to evaluation mode.
    """
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor))
    probability = probabilities[0, 0].cpu().numpy()
    return cv2.resize(probability, roi_size, interpolation=cv2.INTER_LINEAR)


def create_pupil_mask(probability, threshold):
    """Threshold per-pixel pupil probabilities into an OpenCV 0/255 mask."""
    return (probability >= float(threshold)).astype(np.uint8) * 255


def select_pupil_contour(mask, min_area_ratio=0.002, max_area_ratio=0.35, *, reference_area=None):
    """Choose the largest external contour within the allowed ROI area range.

    Area bounds are fractions of the entire ROI, with a minimum of 20 square
    pixels. Changing the ROI can therefore change which pupils are accepted.
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    # A tracked mask can be small without changing the allowed physical pupil
    # area. Its caller supplies the unchanged broad eye ROI as the reference.
    roi_area = int(mask.shape[0] * mask.shape[1]) if reference_area is None else int(reference_area)
    if roi_area <= 0:
        raise ValueError("contour reference area must be positive")
    minimum_area = max(20.0, roi_area * float(min_area_ratio))
    maximum_area = roi_area * float(max_area_ratio)
    candidates = [
        contour
        for contour in contours
        if len(contour) >= 5  # cv2.fitEllipse requires at least five points.
        and minimum_area <= cv2.contourArea(contour) <= maximum_area
    ]
    return max(candidates, key=cv2.contourArea) if candidates else None


def fit_pupil_ellipse(contour, roi_origin):
    """Return ROI-local and full-frame ellipses, each with diameter axes.

    Only the center needs the ROI's (x, y) offset; sizes and angle are unchanged.
    The local ellipse is used for mask overlap, the full ellipse downstream.
    """
    (center_x, center_y), axes, angle = cv2.fitEllipse(contour)
    x, y = roi_origin
    local_ellipse = ((center_x, center_y), axes, angle)
    full_ellipse = ((center_x + x, center_y + y), axes, angle)
    return local_ellipse, full_ellipse


def calculate_pupil_confidence(probability, mask, contour, local_ellipse):
    """Score mean pupil probability times contour/ellipse intersection-over-union.

    Both shapes are rasterized inside the ROI, so overlap is clipped at its
    edges. This heuristic penalizes non-elliptical masks; it is distinct from
    both the pixel threshold and the downstream 3D model's confidence.
    """
    contour_mask = np.zeros_like(mask)
    ellipse_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    cv2.ellipse(ellipse_mask, local_ellipse, 255, -1)
    contour_pixels = contour_mask > 0
    ellipse_pixels = ellipse_mask > 0
    union = np.count_nonzero(contour_pixels | ellipse_pixels)
    intersection = np.count_nonzero(contour_pixels & ellipse_pixels)
    agreement = intersection / union if union else 0.0
    mean_probability = (
        float(np.mean(probability[contour_pixels]))
        if np.any(contour_pixels)
        else 0.0
    )
    return float(np.clip(mean_probability * agreement, 0.0, 1.0))


class PupilDetector:
    """Reuse one evaluation-mode segmentation model for successive eye ROIs.

    Detection is per-frame with no tracking state, allowing the pipeline to
    share this instance across eyes. Input sizes use (width, height).
    """

    def __init__(
        self,
        model,
        device,
        mask_threshold,
        input_size=(320, 192),
        min_area_ratio=0.002,
        max_area_ratio=0.35,
        boundary_policy="existing",
    ):
        if boundary_policy not in ("existing", "experimental"):
            raise ValueError("boundary policy must be existing or experimental")
        self.boundary_policy = boundary_policy
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.input_size = tuple(int(value) for value in input_size)
        self.mask_threshold = float(mask_threshold)
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        if len(self.input_size) != 2 or any(value <= 0 for value in self.input_size):
            raise ValueError("TinyUNet input dimensions must be positive")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("mask threshold must be between zero and one")

    def warm_up(self):
        """Run one dummy inference to initialize the model before video playback."""
        width, height = self.input_size
        example = torch.zeros((1, 1, height, width), device=self.device)
        with torch.inference_mode():
            self.model(example)

    def detect(self, frame, roi_rect, *, evidence_roi=None):
        """Detect within (x, y, width, height) of the supplied full frame.

        The ROI must fit completely; it is not clipped automatically. A valid
        contour returns blink=False even if the resulting confidence is low;
        the eye model applies its own acceptance threshold afterward. Eyelid
        evidence is attached for inspection and does not modify that behavior.
        """
        frame_height, frame_width = frame.shape[:2]
        x, y, width, height = (int(value) for value in roi_rect)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("ROI must contain nonnegative x/y and positive size")
        if x + width > frame_width or y + height > frame_height:
            raise ValueError(
                f"ROI {roi_rect} does not fit frame {(frame_width, frame_height)}"
            )

        # In tracked mode segmentation uses the small ROI, while eyelid and
        # boundary evidence retains the original broad eye view. All returned
        # coordinates still refer to the rotated full frame, before calibration.
        ex, ey, ew, eh = (x, y, width, height) if evidence_roi is None else tuple(evidence_roi)
        if (any(type(v) is not int for v in (ex, ey, ew, eh)) or min(ex, ey) < 0
                or min(ew, eh) <= 0 or ex+ew > frame_width or ey+eh > frame_height
                or x < ex or y < ey or x+width > ex+ew or y+height > ey+eh):
            raise ValueError("evidence ROI must fit frame and contain detection ROI")
        context = frame[ey:ey+eh, ex:ex+ew]
        roi = frame[y:y + height, x:x + width]
        tensor = prepare_tinyunet_input(roi, self.input_size, self.device)
        probability = generate_probability_map(
            self.model,
            tensor,
            (width, height),
        )
        mask = create_pupil_mask(probability, self.mask_threshold)
        contour = select_pupil_contour(
            mask,
            self.min_area_ratio,
            self.max_area_ratio,
            reference_area=ew*eh if evidence_roi is not None else None,
        )
        if contour is None:
            # No independent eyelid classifier: all rejected/missing masks
            # share the blink result, including poor crop placement/segmentation.
            eyelid = detect_eyelid_closure(context, roi_origin=(ex, ey))
            return PupilObservation(None, True, 0.0, eyelid)

        local_ellipse, full_ellipse = fit_pupil_ellipse(contour, (x, y))
        confidence = calculate_pupil_confidence(
            probability,
            mask,
            contour,
            local_ellipse,
        )
        context_ellipse = ((full_ellipse[0][0]-ex, full_ellipse[0][1]-ey),
                           full_ellipse[1], full_ellipse[2])
        context_contour = contour + np.array([x-ex, y-ey], dtype=contour.dtype)
        eyelid = detect_eyelid_closure(context, context_ellipse, confidence, (ex, ey))
        evidence = pupil_boundary_evidence(context, context_contour, context_ellipse)
        rejection = (existing_boundary_quality(context, context_contour, context_ellipse)
                     if self.boundary_policy == "existing"
                     else evidence.reason if evidence.status == "rejected" else None)
        if evidence_roi is not None:
            points = contour.reshape(-1, 2)
            if np.any((points[:, 0] <= 0) | (points[:, 0] >= width-1)
                      | (points[:, 1] <= 0) | (points[:, 1] >= height-1)):
                rejection = "pupil contour clipped by tracking crop"
        return PupilObservation(full_ellipse, False, confidence, eyelid, rejection, evidence)


def load_pupil_detector(checkpoint_path, device, mask_threshold, boundary_policy="existing"):
    """Load weights plus architecture/preprocessing metadata, then warm up.

    The checkpoint must contain model_state_dict and a config mapping with
    base_channels, input_width, and input_height. Input is always grayscale;
    any input_channels metadata is not used. mask_threshold comes from the
    caller rather than the checkpoint and controls pixel-mask binarization.
    """
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"TinyUNet checkpoint not found: {path}")
    if device == "auto":
        # Automatic selection uses CUDA when available, otherwise CPU (no MPS).
        selected_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    # Permit the stored TorchVersion metadata while keeping weights-only loading.
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(
            path,
            map_location=selected_device,
            weights_only=True,
        )
    try:
        config = checkpoint["config"]
        state_dict = checkpoint["model_state_dict"]
        base_channels = int(config["base_channels"])
        input_size = (int(config["input_width"]), int(config["input_height"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"TinyUNet checkpoint has invalid metadata: {error}") from error

    model = TinyUNet(base_channels=base_channels)
    # Strict loading catches architecture/weight mismatches before playback.
    model.load_state_dict(state_dict)
    detector = PupilDetector(
        model,
        selected_device,
        input_size=input_size,
        mask_threshold=mask_threshold,
        boundary_policy=boundary_policy,
    )
    detector.warm_up()
    return detector
