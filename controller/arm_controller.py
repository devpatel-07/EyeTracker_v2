#!/usr/bin/env python3
"""
Xbox One S -> Mac -> ESP32 robotic arm controller.

Requires:
    python3 -m pip install pygame pyserial

IMPORTANT:
    The ESP32 firmware must include the `a <base> <shoulder> <elbow1> <elbow2>`
    command shown in the accompanying instructions. It updates all four joints
    in one coordinated move; repeatedly sending individual `j` commands would
    interrupt the other joints because jogJoint() clears the motion queue.
"""

import argparse
import sys
import time

import pygame
import serial
from serial.tools import list_ports


# ---------- Configuration ----------
BAUD_RATE = 115200
UPDATE_HZ = 10.0
DEADZONE = 0.16

# Maximum joint speed in degrees/second at full stick deflection.
JOINT_SPEEDS = [18.0, 14.0, 14.0, 18.0]  # base, shoulder, elbow 1, elbow 2
TWIST_SPEED = 45.0

# Signs can be changed if a joint moves in the wrong direction.
# Left stick X, -left stick Y, -right stick Y, right stick X.
AXIS_SIGNS = [1.0, 1.0, 1.0, 1.0]

# Absolute angle limits derived from the uploaded ESP32 firmware's PWM limits,
# CENTER=285, PWM_PER_DEGREE=200/90, and joint direction constants.
JOINT_LIMITS = [
    (-90.0, 90.0),       # base
    (-80.1, 75.1),       # shoulder
    (-90.0, 92.2),       # elbow 1 (firmware direction is -1)
    (-90.0, 87.7),       # elbow 2 (firmware direction is -1)
]
TWIST_LIMITS = (-86.8, 86.8)

# Button mapping confirmed by the user's Pygame test (Xbox One S Controller).
BUTTON_A = 0
BUTTON_B = 1
BUTTON_X = 2
BUTTON_Y = 3
BUTTON_LB = 9
BUTTON_RB = 10

# Axis mapping confirmed by the user's Pygame output.
AXIS_LEFT_X = 0
AXIS_LEFT_Y = 1
AXIS_RIGHT_X = 2
AXIS_RIGHT_Y = 3
AXIS_LEFT_TRIGGER = 4
AXIS_RIGHT_TRIGGER = 5


def apply_deadzone(value, deadzone=DEADZONE):
    """Remove stick drift and rescale the remaining range to 0..1."""
    if abs(value) <= deadzone:
        return 0.0
    magnitude = (abs(value) - deadzone) / (1.0 - deadzone)
    return (1.0 if value > 0 else -1.0) * magnitude


def clamp(value, limits):
    return max(limits[0], min(limits[1], value))


def trigger_amount(axis_value):
    """Xbox trigger rests at -1 and reaches +1 when fully pressed."""
    return max(0.0, min(1.0, (axis_value + 1.0) / 2.0))


def choose_serial_port(requested):
    ports = list(list_ports.comports())
    if requested:
        return requested

    likely = [
        p.device for p in ports
        if any(token in (p.device + " " + (p.description or "")).lower()
               for token in ("usbserial", "usbmodem", "cp210", "ch340",
                             "silicon labs", "esp32", "usb uart"))
    ]
    if len(likely) == 1:
        return likely[0]
    if len(ports) == 1:
        return ports[0].device

    print("\nCould not uniquely identify the ESP32 serial port.")
    print("Available ports:")
    for p in ports:
        print(f"  {p.device} — {p.description}")
    print("\nRun again with: python3 arm_controller.py --port /dev/cu.YOUR_PORT")
    raise SystemExit(1)


def send_line(ser, command):
    ser.write((command + "\n").encode("ascii"))
    ser.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="ESP32 serial port, e.g. /dev/cu.usbserial-XXXX")
    args = parser.parse_args()

    port = choose_serial_port(args.port)
    print(f"Connecting to ESP32 on {port} at {BAUD_RATE} baud...")
    print("Opening the serial port may reset the ESP32.")
    print("Waiting 17 seconds for the ESP32 firmware/Wi-Fi startup...")
    ser = serial.Serial(port, BAUD_RATE, timeout=0.1, write_timeout=0.2)
    time.sleep(17)

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() < 1:
        send_line(ser, "s")
        ser.close()
        raise SystemExit("No Xbox controller detected. Check Bluetooth pairing.")

    pad = pygame.joystick.Joystick(0)
    pad.init()
    print(f"Controller: {pad.get_name()}")
    print("\nControls:")
    print("  Left stick X/Y  -> base / shoulder")
    print("  Right stick X/Y -> elbow 2 / elbow 1")
    print("  LT / RT         -> claw close / open")
    print("  LB / RB         -> wrist twist left / right")
    print("  A               -> return to center (then 4-second re-arm delay)")
    print("  B               -> STOP and disarm")
    print("  Ctrl+C           -> stop and exit")
    print("\nStarting with all four joint targets at center (0 degrees).")
    print("Keep the arm's workspace clear. First test with reduced servo power/speed.")

    joint_angles = [0.0, 0.0, 0.0, 0.0]
    twist_angle = 0.0
    previous_buttons = [0] * pad.get_numbuttons()
    armed = True
    rearm_at = 0.0
    last_update = time.monotonic()
    update_period = 1.0 / UPDATE_HZ
    last_claw_percent = None

    try:
        while True:
            now = time.monotonic()
            dt = min(now - last_update, 0.1)
            last_update = now

            pygame.event.pump()
            if pygame.joystick.get_count() < 1:
                print("Controller disconnected; stopping arm.")
                send_line(ser, "s")
                break

            axes = [pad.get_axis(i) for i in range(pad.get_numaxes())]
            buttons = [pad.get_button(i) for i in range(pad.get_numbuttons())]

            def button(index):
                return index < len(buttons) and bool(buttons[index])

            def just_pressed(index):
                return (index < len(buttons)
                        and buttons[index]
                        and not previous_buttons[index])

            # B stops queued motion and disarms joystick control.
            if just_pressed(BUTTON_B):
                send_line(ser, "s")
                armed = False
                print("STOP sent. Press A to return to center and re-arm.")

            # A returns to center. Wait for the move to finish before re-arming.
            if just_pressed(BUTTON_A):
                send_line(ser, "c")
                joint_angles = [0.0, 0.0, 0.0, 0.0]
                twist_angle = 0.0
                armed = False
                rearm_at = now + 4.0
                print("Center command sent; joystick re-arms in 4 seconds.")

            if not armed and rearm_at and now >= rearm_at:
                armed = True
                rearm_at = 0.0
                print("Controller re-armed.")

            if armed:
                # Apply the confirmed axis mapping. Y axes are inverted so
                # pushing a stick forward/up produces a positive command.
                stick_values = [
                    apply_deadzone(axes[AXIS_LEFT_X]) * AXIS_SIGNS[0],
                    apply_deadzone(-axes[AXIS_LEFT_Y]) * AXIS_SIGNS[1],
                    apply_deadzone(-axes[AXIS_RIGHT_Y]) * AXIS_SIGNS[2],
                    apply_deadzone(axes[AXIS_RIGHT_X]) * AXIS_SIGNS[3],
                ]

                for i, value in enumerate(stick_values):
                    joint_angles[i] = clamp(
                        joint_angles[i] + value * JOINT_SPEEDS[i] * dt,
                        JOINT_LIMITS[i],
                    )

                # Send one coordinated four-joint target. The firmware's
                # new `a` command is required for simultaneous joint control.
                if any(abs(v) > 0 for v in stick_values):
                    send_line(
                        ser,
                        "a " + " ".join(f"{angle:.2f}" for angle in joint_angles)
                    )

                # Bumpers rotate the wrist twist. If both are held, they cancel.
                twist_direction = int(button(BUTTON_RB)) - int(button(BUTTON_LB))
                if twist_direction:
                    twist_angle = clamp(
                        twist_angle + twist_direction * TWIST_SPEED * dt,
                        TWIST_LIMITS,
                    )
                    send_line(ser, f"r {twist_angle:.2f}")

                # Triggers are analog: LT closes, RT opens. If both are pulled,
                # the stronger trigger wins through subtraction.
                lt = trigger_amount(axes[AXIS_LEFT_TRIGGER])
                rt = trigger_amount(axes[AXIS_RIGHT_TRIGGER])
                claw_percent = max(0.0, min(100.0, (lt - rt) * 100.0))
                if (last_claw_percent is None
                        or abs(claw_percent - last_claw_percent) >= 2.0):
                    send_line(ser, f"g {claw_percent:.0f}")
                    last_claw_percent = claw_percent

            previous_buttons = buttons

            # Keep the loop near 10 Hz so the ESP32 has time to interpolate
            # each small target update rather than constantly restarting moves.
            elapsed = time.monotonic() - now
            time.sleep(max(0.0, update_period - elapsed))

    except KeyboardInterrupt:
        print("\nStopping arm...")
        send_line(ser, "s")
    except (serial.SerialException, OSError) as exc:
        print(f"\nSerial error: {exc}")
        try:
            send_line(ser, "s")
        except Exception:
            pass
    finally:
        try:
            ser.close()
        except Exception:
            pass
        pygame.quit()
        print("Controller program exited.")


if __name__ == "__main__":
    main()
