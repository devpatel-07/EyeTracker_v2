// ESP32 robot arm with a live 3D web view.
//
// Open http://robotarm.local (or the IP printed on Serial) in a browser on the same network.
// The page draws the reachable workspace, animates the arm from the live servo PWMs,
// and lets you click a point to send the arm there.
//
// Libraries (Library Manager):
//   - Adafruit PWM Servo Driver Library
//   - WebSockets by Markus Sattler (links2004)
// Board: any ESP32 (Tools > Board > ESP32 Dev Module works for most).
// Wiring: PCA9685 SDA -> GPIO 21, SCL -> GPIO 22 (change I2C_SDA / I2C_SCL below).
//
// Servos (PCA9685 channels):
//   0 base (turns)   1 shoulder   2 elbow 1   3 elbow 2   4 wrist (twists claw)   5 claw
// The IK positions the tip with channels 0-3. In the constants below, "WRIST_..." names
// belong to the last up/down joint, which is your elbow 2 (same as your original code).
// The wrist twist and claw are set separately and are never moved by an IK move,
// so the claw keeps holding whatever it has.
//
// Serial and web commands:
//   x y z      move the tip (inches)          c   arm to center
//   r <deg>    wrist twist angle              s   stop
//   g <0-100>  claw: 0 = open, 100 = closed
//   j <n> <deg> move one joint only: n = 0 base, 1 shoulder, 2 elbow 1, 3 elbow 2
//
// Which way "positive" should turn (check each with the j command; if a joint goes
// the other way, flip its *_DIRECTION below):
//   base      + turns counterclockwise, seen from above
//   shoulder  0 = straight up, + leans the arm forward (away from the base's back)
//   elbow 1   + bends the next link upward (toward the sky when the arm reaches forward)
//   elbow 2   + tilts the claw upward the same way

#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <WebSocketsServer.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h>
#include "webpage.h"
#include "arm_types.h"   // types live here so the Arduino IDE's auto-generated prototypes can see them

// ===================== USER SETTINGS =====================
// TEST MODE (leave true for your first runs):
//   - moves at about 11 degrees per second instead of about 45
//   - wakes the servos one at a time, 1.5 s apart, so you can watch each one
//   - skips the startup move; the arm just goes to center
// Set false once every joint moves the right way and the limits are safe.
const bool TEST_MODE = false;

const char* WIFI_SSID = "YOUR_WIFI_NAME";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";

// If the Wi-Fi above can't be joined within 15 s, the ESP32 makes its own network.
// Join it and open http://192.168.4.1
const char* AP_SSID = "RobotArm";
const char* AP_PASS = "robotarm1";      // at least 8 characters

const char* HOSTNAME = "robotarm";      // -> http://robotarm.local

const int I2C_SDA = 21;
const int I2C_SCL = 22;

// Try the elbow bent the other way too (more reachable points). Set false to keep
// the original single-bend behaviour.
const bool ALLOW_ELBOW_FLIP = true;

// Reject poses where the elbow or wrist joint would go below the table.
const bool BLOCK_BELOW_TABLE = true;
const float TABLE_Z = 0.0f;

// Original behaviour: return to center, pause, then move to the target.
const bool RETURN_TO_CENTER_BETWEEN_MOVES = true;
const uint32_t CENTER_PAUSE_MS = 1000;

// Point to move to at power-up (set STARTUP_MOVE false to just sit at center).
const bool STARTUP_MOVE = true;
const float STARTUP_X = 0.0f, STARTUP_Y = 0.0f, STARTUP_Z = 11.0f;
// =========================================================

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
WebServer server(80);
WebSocketsServer ws(81);

#define BASEMIN 85
#define BASEMAX 485
#define SHOULDERMIN 107
#define SHOULDERMAX 452
#define ELBOWMIN 80
#define ELBOWMAX 485
#define WRISTMIN 90
#define WRISTMAX 485
#define CENTER 285

// Wrist twist (channel 4) and claw (channel 5). Find these by testing your servos;
// don't let the claw stall closed.
#define TWISTMIN 92
#define TWISTMAX 478
#define CLAW_OPEN_PWM 200
#define CLAW_CLOSED_PWM 322
const int TWIST_DIRECTION = 1;
const uint8_t TWIST_CHANNEL = 4;
const uint8_t CLAW_CHANNEL = 5;

const float PWM_PER_DEGREE = 200.0f / 90.0f;

// Arm geometry (inches), measured axis to axis:
//   BASE_TO_SHOULDER  table/base reference up to the shoulder axis
//   SHOULDER_TO_ELBOW shoulder axis to elbow 1 axis
//   ELBOW_TO_WRIST    elbow 1 axis to elbow 2 axis
//   WRIST_TO_TIP      elbow 2 axis to the point between the claw fingertips
const float BASE_TO_SHOULDER = 0.5f;
const float SHOULDER_TO_ELBOW = 4.0f;
const float ELBOW_TO_WRIST = 3.5f;
const float WRIST_TO_TIP = 7.0f;

const int BASE_DIRECTION = 1;
const int SHOULDER_DIRECTION = 1;
const int ELBOW_DIRECTION = -1;
const int WRIST_DIRECTION = -1;

// Movement speed: the furthest-moving servo changes PWM_STEP every STEP_DELAY_MS.
// 1 PWM step is about 0.45 degrees.
const int PWM_STEP = TEST_MODE ? 1 : 2;
const int STEP_DELAY_MS = TEST_MODE ? 40 : 20;

const uint8_t SERVO_CHANNELS[4] = {0, 1, 2, 3};  // base, shoulder, elbow, wrist
uint16_t currentPWM[4] = {CENTER, CENTER, CENTER, CENTER};

// Wrist twist and claw ramp toward their targets on their own, independent of arm moves.
uint16_t auxPWM[2] = {CENTER, CLAW_OPEN_PWM};      // twist, claw
uint16_t auxTarget[2] = {CENTER, CLAW_OPEN_PWM};
const uint8_t AUX_CHANNELS[2] = {TWIST_CHANNEL, CLAW_CHANNEL};
uint32_t auxLastTick = 0;

const float DEFAULT_PITCH = -90.0f;
const float PITCH_SEARCH_STEP = 5.0f;

// ---------- live state ----------
bool hasTarget = false;
float targetXYZ[3] = {0, 0, 0};
uint32_t lastBroadcast = 0;

// ---------- small helpers ----------
void broadcastLog(const char* msg) {
  Serial.println(msg);
  char buf[200];
  snprintf(buf, sizeof(buf), "{\"t\":\"l\",\"m\":\"%s\"}", msg);
  ws.broadcastTXT(buf);
}

bool angleToPWM(float angle, int direction, int minPWM, int maxPWM, uint16_t &outputPWM) {
  float pwmValue = CENTER + direction * angle * PWM_PER_DEGREE;
  bool inRange = true;
  if (pwmValue < minPWM) { pwmValue = minPWM; inRange = false; }
  else if (pwmValue > maxPWM) { pwmValue = maxPWM; inRange = false; }
  outputPWM = roundf(pwmValue);
  return inRange;
}

float wrapDegrees(float angle) {
  while (angle > 180.0f) angle -= 360.0f;
  while (angle < -180.0f) angle += 360.0f;
  return angle;
}

// Planar IK. elbowSign = +1 (original bend) or -1 (bent the other way).
// Also returns the heights of the elbow and wrist joints above the base.
bool solvePlanarIK(float r, float shoulderZ, float pitch, int elbowSign,
                   float &shoulderAngle, float &elbowAngle, float &wristAngle,
                   float &elbowJointZ, float &wristJointZ) {
  float pitchRadians = pitch * PI / 180.0f;

  float wristR = r - WRIST_TO_TIP * cosf(pitchRadians);
  float wristZ = shoulderZ - WRIST_TO_TIP * sinf(pitchRadians);

  float distance = sqrtf(wristR * wristR + wristZ * wristZ);
  float maxReach = SHOULDER_TO_ELBOW + ELBOW_TO_WRIST;
  float minReach = fabsf(SHOULDER_TO_ELBOW - ELBOW_TO_WRIST);

  // FIX: the original only checked minReach, so out-of-reach targets were accepted
  if (distance < minReach || distance > maxReach) return false;

  float cosElbow = (wristR * wristR + wristZ * wristZ -
                    SHOULDER_TO_ELBOW * SHOULDER_TO_ELBOW -
                    ELBOW_TO_WRIST * ELBOW_TO_WRIST) /
                   (2.0f * SHOULDER_TO_ELBOW * ELBOW_TO_WRIST);
  cosElbow = constrain(cosElbow, -1.0f, 1.0f);
  float elbowRadians = elbowSign * acosf(cosElbow);

  float targetAngle = atan2f(wristZ, wristR);
  float correctionAngle = atan2f(ELBOW_TO_WRIST * sinf(elbowRadians),
                                 SHOULDER_TO_ELBOW + ELBOW_TO_WRIST * cosf(elbowRadians));
  float shoulderFromHorizontal = targetAngle - correctionAngle;

  float wristRadians = pitchRadians - shoulderFromHorizontal - elbowRadians;

  shoulderAngle = 90.0f - shoulderFromHorizontal * 180.0f / PI;
  elbowAngle = elbowRadians * 180.0f / PI;
  wristAngle = wrapDegrees(wristRadians * 180.0f / PI);

  elbowJointZ = BASE_TO_SHOULDER + SHOULDER_TO_ELBOW * sinf(shoulderFromHorizontal);
  wristJointZ = BASE_TO_SHOULDER + wristZ;
  return true;
}


IKResult solveIK(float x, float y, float z, float preferredPitch) {
  IKResult res = {};
  float r = sqrtf(x * x + y * y);
  res.base = atan2f(y, x) * 180.0f / PI;

  if (!angleToPWM(res.base, BASE_DIRECTION, BASEMIN, BASEMAX, res.pwm[0])) {
    res.why = "outside the base rotation range";
    return res;
  }

  float shoulderZ = z - BASE_TO_SHOULDER;
  bool sawTableHit = false;

  for (float offset = 0.0f; offset <= 180.0f; offset += PITCH_SEARCH_STEP) {
    for (int side = 0; side < 2; side++) {
      if (side == 1 && (offset == 0.0f || offset == 180.0f)) continue;
      float pitch = wrapDegrees(preferredPitch + (side == 0 ? offset : -offset));

      for (int b = 0; b < (ALLOW_ELBOW_FLIP ? 2 : 1); b++) {
        int sign = (b == 0) ? 1 : -1;
        float sh, el, wr, elbowZ, wristZ;
        if (!solvePlanarIK(r, shoulderZ, pitch, sign, sh, el, wr, elbowZ, wristZ)) continue;

        uint16_t sp, ep, wp;
        bool inLimits =
            angleToPWM(sh, SHOULDER_DIRECTION, SHOULDERMIN, SHOULDERMAX, sp) &&
            angleToPWM(el, ELBOW_DIRECTION, ELBOWMIN, ELBOWMAX, ep) &&
            angleToPWM(wr, WRIST_DIRECTION, WRISTMIN, WRISTMAX, wp);
        if (!inLimits) continue;

        if (BLOCK_BELOW_TABLE && (elbowZ < TABLE_Z || wristZ < TABLE_Z)) {
          sawTableHit = true;
          continue;
        }

        res.ok = true;
        res.pitch = pitch;
        res.shoulder = sh; res.elbow = el; res.wrist = wr;
        res.pwm[1] = sp; res.pwm[2] = ep; res.pwm[3] = wp;
        return res;
      }
    }
  }

  res.why = sawTableHit ? "only reachable by hitting the table" : "unreachable within the servo limits";
  return res;
}

// ---------- non-blocking motion queue ----------

MotionStep queueSteps[4];
uint8_t qLen = 0, qIdx = 0;
bool stepStarted = false, busy = false;
uint32_t stepT0 = 0, lastTick = 0;
int moveStart[4], moveSteps = 1, moveStepNo = 0;

void writeServos() {
  for (int i = 0; i < 4; i++) pwm.setPWM(SERVO_CHANNELS[i], 0, currentPWM[i]);
}

void clearQueue() { qLen = 0; qIdx = 0; stepStarted = false; }

void addWait(uint32_t ms) {
  if (qLen >= 4) return;
  queueSteps[qLen].type = STEP_WAIT;
  queueSteps[qLen].waitMs = ms;
  qLen++;
}

void addMove(uint16_t b, uint16_t s, uint16_t e, uint16_t w) {
  if (qLen >= 4) return;
  queueSteps[qLen].type = STEP_MOVE;
  uint16_t t[4] = {b, s, e, w};
  memcpy(queueSteps[qLen].target, t, sizeof(t));
  qLen++;
}

void broadcastState(bool force) {
  uint32_t now = millis();
  if (!force && now - lastBroadcast < 40) return;
  lastBroadcast = now;
  char buf[160];
  if (hasTarget) {
    snprintf(buf, sizeof(buf), "{\"t\":\"s\",\"p\":[%u,%u,%u,%u,%u,%u],\"b\":%d,\"g\":[%.2f,%.2f,%.2f]}",
             currentPWM[0], currentPWM[1], currentPWM[2], currentPWM[3], auxPWM[0], auxPWM[1], busy ? 1 : 0,
             targetXYZ[0], targetXYZ[1], targetXYZ[2]);
  } else {
    snprintf(buf, sizeof(buf), "{\"t\":\"s\",\"p\":[%u,%u,%u,%u,%u,%u],\"b\":%d,\"g\":null}",
             currentPWM[0], currentPWM[1], currentPWM[2], currentPWM[3], auxPWM[0], auxPWM[1], busy ? 1 : 0);
  }
  ws.broadcastTXT(buf);
}

bool auxMoving() { return auxPWM[0] != auxTarget[0] || auxPWM[1] != auxTarget[1]; }

void serviceAux() {
  if (!auxMoving()) return;
  uint32_t now = millis();
  if (now - auxLastTick < (uint32_t)STEP_DELAY_MS) return;
  auxLastTick = now;
  for (int i = 0; i < 2; i++) {
    int diff = (int)auxTarget[i] - (int)auxPWM[i];
    if (diff == 0) continue;
    auxPWM[i] += constrain(diff, -PWM_STEP, PWM_STEP);
    pwm.setPWM(AUX_CHANNELS[i], 0, auxPWM[i]);
  }
  broadcastState(!auxMoving());
}

void setTwist(float angle) {
  uint16_t p;
  bool ok = angleToPWM(angle, TWIST_DIRECTION, TWISTMIN, TWISTMAX, p);
  auxTarget[0] = p;
  char msg[80];
  snprintf(msg, sizeof(msg), "Wrist twist %.0f deg%s", angle, ok ? "" : " (clamped to the servo limit)");
  broadcastLog(msg);
}

void setClaw(float percent) {
  percent = constrain(percent, 0.0f, 100.0f);
  auxTarget[1] = roundf(CLAW_OPEN_PWM + (CLAW_CLOSED_PWM - CLAW_OPEN_PWM) * percent / 100.0f);
  char msg[60];
  snprintf(msg, sizeof(msg), "Claw %.0f%% closed", percent);
  broadcastLog(msg);
}

void serviceMotion() {
  if (qIdx >= qLen) {
    if (busy && !auxMoving()) { busy = false; broadcastState(true); }
    else if (auxMoving()) busy = true;
    return;
  }
  busy = true;
  MotionStep &m = queueSteps[qIdx];
  uint32_t now = millis();

  if (!stepStarted) {
    stepStarted = true;
    stepT0 = now;
    if (m.type == STEP_MOVE) {
      int maxChange = 0;
      for (int i = 0; i < 4; i++) {
        moveStart[i] = currentPWM[i];
        maxChange = max(maxChange, abs((int)m.target[i] - moveStart[i]));
      }
      moveSteps = max(1, maxChange / PWM_STEP);
      moveStepNo = 0;
      lastTick = now - STEP_DELAY_MS;
    }
  }

  if (m.type == STEP_WAIT) {
    if (now - stepT0 >= m.waitMs) { qIdx++; stepStarted = false; }
    return;
  }

  if (now - lastTick < (uint32_t)STEP_DELAY_MS) return;
  lastTick = now;
  moveStepNo++;
  for (int i = 0; i < 4; i++) {
    long change = (long)((int)m.target[i] - moveStart[i]) * moveStepNo / moveSteps;
    currentPWM[i] = moveStart[i] + change;
  }
  writeServos();
  if (moveStepNo >= moveSteps) { qIdx++; stepStarted = false; }
  broadcastState(false);
}

// ---------- commands ----------
void sendIKResult(const IKResult &r, float x, float y, float z) {
  char buf[260];
  if (r.ok) {
    snprintf(buf, sizeof(buf),
             "{\"t\":\"k\",\"ok\":1,\"g\":[%.2f,%.2f,%.2f],\"pitch\":%.1f,\"a\":[%.2f,%.2f,%.2f,%.2f],\"p\":[%u,%u,%u,%u]}",
             x, y, z, r.pitch, r.base, r.shoulder, r.elbow, r.wrist,
             r.pwm[0], r.pwm[1], r.pwm[2], r.pwm[3]);
  } else {
    snprintf(buf, sizeof(buf), "{\"t\":\"k\",\"ok\":0,\"g\":[%.2f,%.2f,%.2f],\"why\":\"%s\"}", x, y, z, r.why);
  }
  ws.broadcastTXT(buf);
}

bool requestMove(float x, float y, float z) {
  IKResult r = solveIK(x, y, z, DEFAULT_PITCH);
  sendIKResult(r, x, y, z);
  char msg[160];
  if (!r.ok) {
    snprintf(msg, sizeof(msg), "(%.2f, %.2f, %.2f) rejected: %s", x, y, z, r.why);
    broadcastLog(msg);
    return false;
  }
  snprintf(msg, sizeof(msg), "(%.2f, %.2f, %.2f) pitch %.0f  base %.1f  shoulder %.1f  elbow %.1f  wrist %.1f",
           x, y, z, r.pitch, r.base, r.shoulder, r.elbow, r.wrist);
  broadcastLog(msg);

  hasTarget = true;
  targetXYZ[0] = x; targetXYZ[1] = y; targetXYZ[2] = z;

  clearQueue();   // a new command replaces whatever was running, starting from where the arm is now
  if (RETURN_TO_CENTER_BETWEEN_MOVES) {
    addWait(CENTER_PAUSE_MS);
    addMove(CENTER, CENTER, CENTER, CENTER);
    addWait(CENTER_PAUSE_MS);
  }
  addMove(r.pwm[0], r.pwm[1], r.pwm[2], r.pwm[3]);
  broadcastState(true);
  return true;
}

void jogJoint(int joint, float angle) {
  static const char* names[4] = {"base", "shoulder", "elbow 1", "elbow 2"};
  static const int dirs[4] = {BASE_DIRECTION, SHOULDER_DIRECTION, ELBOW_DIRECTION, WRIST_DIRECTION};
  static const int mins[4] = {BASEMIN, SHOULDERMIN, ELBOWMIN, WRISTMIN};
  static const int maxs[4] = {BASEMAX, SHOULDERMAX, ELBOWMAX, WRISTMAX};
  if (joint < 0 || joint > 3) { broadcastLog("Joint must be 0-3 (base, shoulder, elbow 1, elbow 2)"); return; }

  uint16_t p;
  bool ok = angleToPWM(angle, dirs[joint], mins[joint], maxs[joint], p);
  uint16_t t[4] = {currentPWM[0], currentPWM[1], currentPWM[2], currentPWM[3]};
  t[joint] = p;

  hasTarget = false;
  clearQueue();
  addMove(t[0], t[1], t[2], t[3]);

  char msg[100];
  snprintf(msg, sizeof(msg), "Jog %s to %.0f deg (pwm %u)%s", names[joint], angle, p,
           ok ? "" : ", clamped to the servo limit");
  broadcastLog(msg);
  broadcastState(true);
}

const char* skipSep(const char* s) {
  while (*s == ' ' || *s == ',' || *s == '\t') s++;
  return s;
}

bool parse3(const char* s, float &x, float &y, float &z) {
  float v[3];
  for (int i = 0; i < 3; i++) {
    s = skipSep(s);
    char* end;
    v[i] = strtof(s, &end);
    if (end == s) return false;
    s = end;
  }
  x = v[0]; y = v[1]; z = v[2];
  return true;
}

// "m x y z" or "x y z" = move, "c" = center, "s" = stop
void handleCommand(const char* cmd) {
  cmd = skipSep(cmd);
  if (cmd[0] == 'c' || cmd[0] == 'C') {
    hasTarget = false;
    clearQueue();
    addMove(CENTER, CENTER, CENTER, CENTER);
    broadcastLog("Moving to center");
    broadcastState(true);
    return;
  }
  if (cmd[0] == 'r' || cmd[0] == 'R') {
    char* end;
    float a = strtof(skipSep(cmd + 1), &end);
    if (end != skipSep(cmd + 1)) setTwist(a);
    else broadcastLog("Send: r <angle>");
    return;
  }
  if (cmd[0] == 'j' || cmd[0] == 'J') {
    float n, a;
    const char* p = skipSep(cmd + 1);
    char* end;
    n = strtof(p, &end);
    if (end == p) { broadcastLog("Send: j <joint 0-3> <angle>"); return; }
    p = skipSep(end);
    a = strtof(p, &end);
    if (end == p) { broadcastLog("Send: j <joint 0-3> <angle>"); return; }
    jogJoint((int)n, a);
    return;
  }
  if (cmd[0] == 'g' || cmd[0] == 'G') {
    char* end;
    float p = strtof(skipSep(cmd + 1), &end);
    if (end != skipSep(cmd + 1)) setClaw(p);
    else broadcastLog("Send: g <0-100>");
    return;
  }
  if (cmd[0] == 's' || cmd[0] == 'S') {
    clearQueue();
    auxTarget[0] = auxPWM[0];
    auxTarget[1] = auxPWM[1];
    broadcastLog("Stopped");
    broadcastState(true);
    return;
  }
  if (cmd[0] == 'm' || cmd[0] == 'M') cmd++;
  float x, y, z;
  if (parse3(cmd, x, y, z)) requestMove(x, y, z);
  else if (*cmd) broadcastLog("Didn't understand that. Send: x y z");
}

void sendConfig(uint8_t client) {
  char buf[600];
  snprintf(buf, sizeof(buf),
           "{\"t\":\"cfg\",\"L\":[%.3f,%.3f,%.3f,%.3f],"
           "\"min\":[%d,%d,%d,%d],\"max\":[%d,%d,%d,%d],\"dir\":[%d,%d,%d,%d],"
           "\"center\":%d,\"ppd\":%.6f,\"pitch0\":%.1f,\"pstep\":%.1f,"
           "\"flip\":%d,\"table\":%d,\"tableZ\":%.2f,"
           "\"twist\":[%d,%d,%d],\"claw\":[%d,%d],\"test\":%d}",
           BASE_TO_SHOULDER, SHOULDER_TO_ELBOW, ELBOW_TO_WRIST, WRIST_TO_TIP,
           BASEMIN, SHOULDERMIN, ELBOWMIN, WRISTMIN,
           BASEMAX, SHOULDERMAX, ELBOWMAX, WRISTMAX,
           BASE_DIRECTION, SHOULDER_DIRECTION, ELBOW_DIRECTION, WRIST_DIRECTION,
           CENTER, PWM_PER_DEGREE, DEFAULT_PITCH, PITCH_SEARCH_STEP,
           ALLOW_ELBOW_FLIP ? 1 : 0, BLOCK_BELOW_TABLE ? 1 : 0, TABLE_Z,
           TWISTMIN, TWISTMAX, TWIST_DIRECTION, CLAW_OPEN_PWM, CLAW_CLOSED_PWM, TEST_MODE ? 1 : 0);
  ws.sendTXT(client, buf);
}

void onWsEvent(uint8_t num, WStype_t type, uint8_t* payload, size_t length) {
  if (type == WStype_CONNECTED) {
    sendConfig(num);
    broadcastState(true);
  } else if (type == WStype_TEXT) {
    char cmd[64];
    size_t n = min(length, sizeof(cmd) - 1);
    memcpy(cmd, payload, n);
    cmd[n] = 0;
    handleCommand(cmd);
  }
}

// ---------- serial input (non-blocking) ----------
char serialBuf[64];
uint8_t serialLen = 0;

void pollSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialLen) { serialBuf[serialLen] = 0; handleCommand(serialBuf); serialLen = 0; }
    } else if (serialLen < sizeof(serialBuf) - 1) {
      serialBuf[serialLen++] = c;
    }
  }
}

// ---------- setup / loop ----------
void startNetwork() {
  WiFi.mode(WIFI_STA);
  WiFi.setHostname(HOSTNAME);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Joining Wi-Fi");
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Open http://");
    Serial.print(HOSTNAME);
    Serial.print(".local  or  http://");
    Serial.println(WiFi.localIP());
  } else {
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID, AP_PASS);
    Serial.print("Couldn't join Wi-Fi. Join network \"");
    Serial.print(AP_SSID);
    Serial.print("\" and open http://");
    Serial.println(WiFi.softAPIP());
  }

  if (MDNS.begin(HOSTNAME)) MDNS.addService("http", "tcp", 80);

  server.on("/", []() { server.send_P(200, "text/html", INDEX_HTML); });
  server.onNotFound([]() { server.send(404, "text/plain", "Not found"); });
  server.begin();

  ws.begin();
  ws.onEvent(onWsEvent);
}

void setup() {
  Serial.begin(115200);

  Wire.begin(I2C_SDA, I2C_SCL);
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(10);

  // Until a servo gets its first pulse it is limp. Its first pulse makes it jump
  // straight to that position (hobby servos can't report where they are), so in
  // test mode wake them one at a time.
  static const char* names[6] = {"base", "shoulder", "elbow 1", "elbow 2", "wrist", "claw"};
  for (int i = 0; i < 6; i++) {
    uint16_t p = (i < 4) ? currentPWM[i] : auxPWM[i - 4];
    uint8_t ch = (i < 4) ? SERVO_CHANNELS[i] : AUX_CHANNELS[i - 4];
    if (TEST_MODE) {
      Serial.print("Waking ");
      Serial.print(names[i]);
      Serial.print(" (channel ");
      Serial.print(ch);
      Serial.println(")");
    }
    pwm.setPWM(ch, 0, p);
    if (TEST_MODE) delay(1500);
  }
  startNetwork();

  if (TEST_MODE) Serial.println("TEST MODE: slow speed, no startup move. Try: j 0 15");
  else if (STARTUP_MOVE) requestMove(STARTUP_X, STARTUP_Y, STARTUP_Z);
}

void loop() {
  server.handleClient();
  ws.loop();
  pollSerial();
  serviceMotion();
  serviceAux();
}
