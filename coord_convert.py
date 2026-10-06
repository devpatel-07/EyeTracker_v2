# Convert mm to inches
def mm2in(mm):
    return mm / 25.4

# Convert gaze point to reach point for arm
# Gaze point (mm): origin on the ground below the person's eyes. x forward, y to the person's left, z up
# Arm point (in): origin at the arm reference point, d ahead of the person and h + o above the ground.
# x points back toward the person, y to the person's left, z up
def convert_coords(gx, gy, gz, d, h, o):

    rx = d - mm2in(gx)
    ry = mm2in(gy)
    rz = mm2in(gz) - (h + o)

    return (rx,ry,rz)
