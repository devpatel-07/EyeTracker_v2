// Types used by arm_live.ino. They're in their own header because the Arduino IDE
// inserts function prototypes near the top of the .ino, before any types defined there.
#pragma once
#include <Arduino.h>

struct IKResult {
  bool ok;
  const char* why;
  float pitch, base, shoulder, elbow, wrist;
  uint16_t pwm[4];
};

enum StepType : uint8_t { STEP_WAIT, STEP_MOVE };
struct MotionStep { StepType type; uint32_t waitMs; uint16_t target[4]; };
