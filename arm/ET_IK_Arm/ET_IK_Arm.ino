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

#define CENTER 285
#define QUARTER 200

const float PWM_PER_DEGREE = 200.0f / 90.0f;

// Arm geometry
const float BASE_TO_SHOULDER = 0.5f;
const float SHOULDER_TO_ELBOW = 4.5f;
const float ELBOW_TO_TIP = 1.0f;

const uint8_t BASE_CHANNEL = 0;
const uint8_t SHOULDER_CHANNEL = 1;
const uint8_t ELBOW_CHANNEL = 2;

// Change these to -1 if a servo moves in the opposite direction
const int BASE_DIRECTION = -1;
const int SHOULDER_DIRECTION = 1;
const int ELBOW_DIRECTION = 1;


void setup() {
  Serial.begin(115200);

  Wire.begin();
  pwm.begin();
  pwm.setPWMFreq(50);

  // Start at center
  goToCenter();
  //moveToPoint(0.0,5.5,0.5);
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
  pwm.setPWM(BASE_CHANNEL, 0, CENTER);
  pwm.setPWM(SHOULDER_CHANNEL, 0, CENTER);
  pwm.setPWM(ELBOW_CHANNEL, 0, CENTER);
  delay(1000);
}

// Converts a robot joint angle into PWM
bool angleToPWM(float angle, int direction, int minPWM, int maxPWM, uint16_t &outputPWM) {

  float pwmValue = CENTER + direction * angle * PWM_PER_DEGREE;

  // Make sure this angle is physically allowed
  if (pwmValue < minPWM) {
      pwmValue = minPWM;
  } else if  (pwmValue > maxPWM) {
      pwmValue = maxPWM;
  }

  outputPWM = roundf(pwmValue);
  return true;
}


// Move end of arm to x, y, z
bool moveToPoint(float x, float y, float z) {

  // BASE
  // Horizontal distance from Z axis
  float r = sqrtf(x * x + y * y);

  // Base rotation
  float baseAngle = atan2f(y, x) * 180.0f / PI;


  // REMOVE FIXED 0.5 IN OFFSET
  // Shoulder is 0.5 inches above base
  float shoulderZ = z - BASE_TO_SHOULDER;


  // CHECK TARGET DISTANCE
  float distance = sqrtf(r * r + shoulderZ * shoulderZ);
  //float maxReach = SHOULDER_TO_ELBOW + ELBOW_TO_TIP;
  float minReach = fabsf(SHOULDER_TO_ELBOW - ELBOW_TO_TIP);

  if (distance < minReach) {
    Serial.println("Target is unreachable.");
    return false;
  }


  // ELBOW IK
  float cosElbow =
      (r * r +
       shoulderZ * shoulderZ -
       SHOULDER_TO_ELBOW * SHOULDER_TO_ELBOW -
       ELBOW_TO_TIP * ELBOW_TO_TIP)
      /
      (2.0f *
       SHOULDER_TO_ELBOW *
       ELBOW_TO_TIP);

  // Protect against tiny floating point errors
  cosElbow = constrain(cosElbow, -1.0f, 1.0f);
  float elbowRadians = acosf(cosElbow);
  float elbowAngle = elbowRadians * 180.0f / PI;


  // SHOULDER IK
  float targetAngle = atan2f(shoulderZ, r);

  float correctionAngle =
      atan2f(
        ELBOW_TO_TIP * sinf(elbowRadians),
        SHOULDER_TO_ELBOW +
        ELBOW_TO_TIP * cosf(elbowRadians)
      );

  float shoulderFromHorizontal = targetAngle - correctionAngle;

  // Shoulder 0° is straight UP, not horizontal.
  float shoulderAngle = 90.0f - shoulderFromHorizontal * 180.0f / PI;


  // PRINT IK RESULTS
  Serial.print("Base angle: ");
  Serial.println(baseAngle);

  Serial.print("Shoulder angle: ");
  Serial.println(shoulderAngle);

  Serial.print("Elbow angle: ");
  Serial.println(elbowAngle);


  // ANGLES -> PWM
  uint16_t basePWM;
  uint16_t shoulderPWM;
  uint16_t elbowPWM;

  angleToPWM(baseAngle, BASE_DIRECTION, BASEMIN, BASEMAX, basePWM);
  angleToPWM(shoulderAngle, SHOULDER_DIRECTION, SHOULDERMIN, SHOULDERMAX, shoulderPWM);
  angleToPWM(elbowAngle, ELBOW_DIRECTION, ELBOWMIN, ELBOWMAX, elbowPWM);

  // MOVE SERVOS
  goToCenter();
  pwm.setPWM(BASE_CHANNEL, 0, basePWM);
  pwm.setPWM(SHOULDER_CHANNEL, 0, shoulderPWM);
  pwm.setPWM(ELBOW_CHANNEL, 0, elbowPWM);

  return true;
}