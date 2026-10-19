#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define BASEMIN 60
#define BASEMAX 512

#define SHOULDERMIN 85
#define SHOULDERMAX 485

#define ELBOWMIN 60
#define ELBOWMAX 530

// Set these to the physical limits of the wrist servo
#define WRISTMIN 90
#define WRISTMAX 490

#define CENTER 285
#define QUARTER 200

const float PWM_PER_DEGREE = 200.0f / 90.0f;

// Arm geometry
const float BASE_TO_SHOULDER = 0.5f;
const float SHOULDER_TO_ELBOW = 4.0f;
const float ELBOW_TO_WRIST = 3.5f;
const float WRIST_TO_TIP = 7.0f;

const uint8_t BASE_CHANNEL = 0;
const uint8_t SHOULDER_CHANNEL = 1;
const uint8_t ELBOW_CHANNEL = 2;
const uint8_t WRIST_CHANNEL = 3;

// Change these to -1 if a servo moves in the opposite direction
const int BASE_DIRECTION = -1;
const int SHOULDER_DIRECTION = 1;
const int ELBOW_DIRECTION = 1;
const int WRIST_DIRECTION = -1;

// Movement speed: each step changes the furthest-moving servo by PWM_STEP,
// then waits STEP_DELAY_MS. Lower PWM_STEP or raise STEP_DELAY_MS to go slower.
// 1 PWM every 10 ms is about 45 degrees per second.
const int PWM_STEP = 2;
const int STEP_DELAY_MS = 20;

const uint8_t SERVO_CHANNELS[4] = {0, 1, 2, 3};  // base, shoulder, elbow, wrist
uint16_t currentPWM[4] = {CENTER, CENTER, CENTER, CENTER};

// Preferred angle of the tip from horizontal (-90 = pointing straight down).
// If the target can't be reached at this angle, nearby angles are tried.
const float DEFAULT_PITCH = -90.0f;
const float PITCH_SEARCH_STEP = 5.0f;


// moveToPoint prototype
bool moveToPoint(float x, float y, float z, float preferredPitch = DEFAULT_PITCH);

void setup() {
  Serial.begin(115200);

  Wire.begin();
  pwm.begin();
  pwm.setPWMFreq(50);

  // Start at center
  goToCenter();
  moveToPoint(0.0,0.0,11.0);
}


void loop() {

  if (Serial.available() > 0) {

    float x = Serial.parseFloat();
    float y = Serial.parseFloat();
    float z = Serial.parseFloat();

    moveToPoint(x, y, z);
  }
}

// Go to center position and wait for 1000ms before and after
void goToCenter() {
  delay(1000);
  moveServosSmooth(CENTER, CENTER, CENTER, CENTER);
  delay(1000);
}

// Converts a robot joint angle into PWM.
// The output is clamped to the servo limits; returns false if clamping was needed.
bool angleToPWM(float angle, int direction, int minPWM, int maxPWM, uint16_t &outputPWM) {

  float pwmValue = CENTER + direction * angle * PWM_PER_DEGREE;
  bool inRange = true;

  // Make sure this angle is physically allowed
  if (pwmValue < minPWM) {
      pwmValue = minPWM;
      inRange = false;
  } else if  (pwmValue > maxPWM) {
      pwmValue = maxPWM;
      inRange = false;
  }

  outputPWM = roundf(pwmValue);
  return inRange;
}

// Wrap an angle in degrees to [-180, 180]
float wrapDegrees(float angle) {
  while (angle > 180.0f) angle -= 360.0f;
  while (angle < -180.0f) angle += 360.0f;
  return angle;
}


// Solve shoulder, elbow and wrist angles in the arm's vertical plane.
//   r         - horizontal distance from the base axis to the tip
//   shoulderZ - tip height relative to the shoulder
//   pitch     - tip angle from horizontal in degrees (-90 = straight down)
// Returns false if the wrist can't reach the required point.
bool solvePlanarIK(float r, float shoulderZ, float pitch,
                   float &shoulderAngle, float &elbowAngle, float &wristAngle) {

  float pitchRadians = pitch * PI / 180.0f;

  // WRIST POSITION
  // Step back from the tip along the tool direction
  float wristR = r - WRIST_TO_TIP * cosf(pitchRadians);
  float wristZ = shoulderZ - WRIST_TO_TIP * sinf(pitchRadians);

  // CHECK WRIST DISTANCE
  float distance = sqrtf(wristR * wristR + wristZ * wristZ);
  float maxReach = SHOULDER_TO_ELBOW + ELBOW_TO_WRIST;
  float minReach = fabsf(SHOULDER_TO_ELBOW - ELBOW_TO_WRIST);

  if (distance < minReach) {
    return false;
  }


  // ELBOW IK
  float cosElbow =
      (wristR * wristR +
       wristZ * wristZ -
       SHOULDER_TO_ELBOW * SHOULDER_TO_ELBOW -
       ELBOW_TO_WRIST * ELBOW_TO_WRIST)
      /
      (2.0f *
       SHOULDER_TO_ELBOW *
       ELBOW_TO_WRIST);

  // Protect against tiny floating point errors
  cosElbow = constrain(cosElbow, -1.0f, 1.0f);
  float elbowRadians = acosf(cosElbow);


  // SHOULDER IK
  float targetAngle = atan2f(wristZ, wristR);

  float correctionAngle =
      atan2f(
        ELBOW_TO_WRIST * sinf(elbowRadians),
        SHOULDER_TO_ELBOW +
        ELBOW_TO_WRIST * cosf(elbowRadians)
      );

  float shoulderFromHorizontal = targetAngle - correctionAngle;


  // WRIST IK
  // Whatever rotation is left over to point the tip at the requested pitch
  float wristRadians = pitchRadians - shoulderFromHorizontal - elbowRadians;


  // Shoulder 0° is straight UP, not horizontal.
  shoulderAngle = 90.0f - shoulderFromHorizontal * 180.0f / PI;
  elbowAngle = elbowRadians * 180.0f / PI;
  // Wrist 0° is in line with the forearm, positive in the same sense as the elbow.
  wristAngle = wrapDegrees(wristRadians * 180.0f / PI);

  return true;
}

// Move all servos from their current PWM to the targets together,
// in small steps so they all arrive at the same time
void moveServosSmooth(uint16_t basePWM, uint16_t shoulderPWM, uint16_t elbowPWM, uint16_t wristPWM) {

  uint16_t target[4] = {basePWM, shoulderPWM, elbowPWM, wristPWM};
  int start[4];
  int maxChange = 0;

  for (int i = 0; i < 4; i++) {
    start[i] = currentPWM[i];
    maxChange = max(maxChange, abs((int)target[i] - start[i]));
  }

  int steps = max(1, maxChange / PWM_STEP);

  for (int step = 1; step <= steps; step++) {
    for (int i = 0; i < 4; i++) {
      // long avoids overflowing a 16-bit int on AVR boards
      long change = (long)((int)target[i] - start[i]) * step / steps;
      pwm.setPWM(SERVO_CHANNELS[i], 0, start[i] + change);
    }
    delay(STEP_DELAY_MS);
  }

  for (int i = 0; i < 4; i++) {
    currentPWM[i] = target[i];
  }
}


// Move end of arm to x, y, z
bool moveToPoint(float x, float y, float z, float preferredPitch) {

  // BASE
  // Horizontal distance from Z axis
  float r = sqrtf(x * x + y * y);

  // Base rotation
  float baseAngle = atan2f(y, x) * 180.0f / PI;

  uint16_t basePWM;
  if (!angleToPWM(baseAngle, BASE_DIRECTION, BASEMIN, BASEMAX, basePWM)) {
    Serial.println("Target is outside the base rotation range.");
    return false;
  }


  // REMOVE FIXED 0.5 IN OFFSET
  // Shoulder is 0.5 inches above base
  float shoulderZ = z - BASE_TO_SHOULDER;


  // SEARCH FOR A WORKING TIP PITCH
  // Try the preferred pitch first, then step outward on both sides of it
  // until a solution is found that is reachable and inside every servo's limits.
  float shoulderAngle, elbowAngle, wristAngle, pitch;
  uint16_t shoulderPWM, elbowPWM, wristPWM;
  bool found = false;

  for (float offset = 0.0f; offset <= 180.0f && !found; offset += PITCH_SEARCH_STEP) {
    for (int side = 0; side < 2 && !found; side++) {

      // Offset 0 and 180 only need one side
      if (side == 1 && (offset == 0.0f || offset == 180.0f)) {
        continue;
      }

      pitch = wrapDegrees(preferredPitch + (side == 0 ? offset : -offset));

      if (!solvePlanarIK(r, shoulderZ, pitch, shoulderAngle, elbowAngle, wristAngle)) {
        continue;
      }

      found =
          angleToPWM(shoulderAngle, SHOULDER_DIRECTION, SHOULDERMIN, SHOULDERMAX, shoulderPWM) &&
          angleToPWM(elbowAngle, ELBOW_DIRECTION, ELBOWMIN, ELBOWMAX, elbowPWM) &&
          angleToPWM(wristAngle, WRIST_DIRECTION, WRISTMIN, WRISTMAX, wristPWM);
    }
  }

  if (!found) {
    Serial.println("Target is unreachable.");
    return false;
  }


  // PRINT IK RESULTS
  Serial.print("Tip pitch: ");
  Serial.println(pitch);

  Serial.print("Base angle: ");
  Serial.println(baseAngle);

  Serial.print("Shoulder angle: ");
  Serial.println(shoulderAngle);

  Serial.print("Elbow angle: ");
  Serial.println(elbowAngle);

  Serial.print("Wrist angle: ");
  Serial.println(wristAngle);


  // MOVE SERVOS
  goToCenter();
  moveServosSmooth(basePWM, shoulderPWM, elbowPWM, wristPWM);

  return true;
}
