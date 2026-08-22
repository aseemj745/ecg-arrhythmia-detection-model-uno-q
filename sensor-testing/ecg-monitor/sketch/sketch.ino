/*
 * ============================================================
 * ECG UNO Q - STM32 MCU SIDE
 * ============================================================
 *
 * BioAmp EXG Pill -> A0
 * MPU6050 (motion/displacement) -> I2C (SDA/SCL)
 *
 * Sampling:
 *   ECG:      125 Hz
 *   MPU6050:   50 Hz
 *
 * Bridge:
 *   MCU -> Linux MPU
 *
 * Transfer:
 *   25 ECG samples per Bridge notification
 *   = 200 ms of ECG data
 *
 *   1 "motion_state" notification per motion state change
 *   (edge-triggered, not sent every sample)
 *
 * Motion gating:
 *   While the MPU6050 reports a big displacement (the patient/
 *   sensor moved suddenly), ECG samples stop being batched and
 *   sent -- a moving electrode produces motion artifact, not
 *   real cardiac signal, so we pause collection rather than
 *   stream garbage. Collection resumes automatically once the
 *   sensor is still again.
 *
 * Arduino_RouterBridge:
 *   0.4.3
 *
 * ============================================================
 */

#include <Arduino_RouterBridge.h>
#include <Wire.h>
#include <vector>

// ------------------------------------------------------------
// Configuration
// ------------------------------------------------------------

const int ECG_PIN = A0;

const unsigned long FS_HZ = 125;
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / FS_HZ;

// 25 samples = 200 ms at 125 Hz
const int BATCH_SIZE = 25;


// ------------------------------------------------------------
// MPU6050 (motion / displacement sensor)
// ------------------------------------------------------------
// Talked to directly over Wire at the register level (no extra
// library dependency needed beyond the board's built-in Wire).
// Wire it up as: VCC -> 3V3/5V, GND -> GND, SDA -> board SDA,
// SCL -> board SCL (on Uno Q that's the dedicated SDA/SCL pins;
// A4/A5 also work on many I2C-capable Arduino boards).

const uint8_t MPU6050_ADDR            = 0x68;
const uint8_t MPU6050_REG_PWR_MGMT_1  = 0x6B;
const uint8_t MPU6050_REG_WHO_AM_I    = 0x75;
const uint8_t MPU6050_REG_ACCEL_XOUT_H = 0x3B;

// LSB per g for the default +/-2g accelerometer range.
const float ACCEL_LSB_PER_G = 16384.0f;

// How far the measured acceleration magnitude may drift from the
// slow-tracked "at rest" baseline (~1 g) before it counts as a
// huge displacement / motion event. Raise this if normal handling
// of the device trips it too easily; lower it to catch smaller
// movements.
const float MOTION_THRESHOLD_G = 0.35f;

// Require this many consecutive over/under-threshold IMU samples
// before flipping state, so one noisy reading doesn't cause
// chatter on the Bridge or a flickering UI popup.
const int MOTION_DEBOUNCE_COUNT = 3;

// Sample the IMU at 50 Hz -- plenty to catch body movement while
// staying light next to the 125 Hz ECG loop.
const unsigned long MPU_SAMPLE_INTERVAL_US = 1000000UL / 50;
unsigned long next_mpu_sample_us = 0;

bool mpu6050_ok = false;
float accel_baseline_g = 1.0f;   // slow-moving "at rest" magnitude
bool motion_detected = false;
int motion_over_count = 0;
int motion_under_count = 0;


// ------------------------------------------------------------
// Timing
// ------------------------------------------------------------

unsigned long next_sample_us = 0;


// ------------------------------------------------------------
// ECG batch
// ------------------------------------------------------------

std::vector<float> ecg_batch;

unsigned long total_samples = 0;
unsigned long total_batches = 0;


// ------------------------------------------------------------
// Debug timing
// ------------------------------------------------------------

unsigned long debug_last_ms = 0;
float last_ecg_value = 0.0f;



// ============================================================
// ECG FILTER
// ============================================================
//
// BioAmp EXG Pill ECG filter
//
// Band-Pass Butterworth IIR digital filter
// Sampling rate: 125 Hz
// Frequency: 0.5 Hz - 44.5 Hz
// Filter order: 4, implemented as second-order sections
//
// ============================================================

float ECGFilter(float input)
{
  float output = input;

  {
    static float z1, z2;
    float x = output - 0.70682283*z1 - 0.15621030*z2;
    output = 0.28064917*x + 0.56129834*z1 + 0.28064917*z2;
    z2 = z1;
    z1 = x;
  }

  {
    static float z1, z2;
    float x = output - 0.95028224*z1 - 0.54073140*z2;
    output = 1.00000000*x + 2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }

  {
    static float z1, z2;
    float x = output - -1.95360385*z1 - 0.95423412*z2;
    output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }

  {
    static float z1, z2;
    float x = output - -1.98048558*z1 - 0.98111344*z2;
    output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }

  return output;
}

// ============================================================
// MPU6050 HELPERS
// ============================================================

void mpu6050_write_reg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

// Wakes the sensor and confirms it's actually there via WHO_AM_I.
// Returns false (rather than blocking forever) if it's not found,
// so a missing/unwired IMU doesn't stop ECG acquisition -- motion
// gating is simply disabled in that case.
bool mpu6050_init() {
  Wire.begin();

  mpu6050_write_reg(MPU6050_REG_PWR_MGMT_1, 0x00);  // wake from sleep
  delay(50);

  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(MPU6050_REG_WHO_AM_I);
  if (Wire.endTransmission(false) != 0) return false;

  if (Wire.requestFrom((int)MPU6050_ADDR, 1) < 1) return false;
  uint8_t who = Wire.read();

  // Some MPU6050/MPU6500 clones report 0x70 or 0x98 instead of the
  // "official" 0x68 -- accept anything that answered on the bus.
  return (who == 0x68 || who == 0x70 || who == 0x98);
}

// Reads accelerometer X/Y/Z in g. Returns false on an I2C error.
bool mpu6050_read_accel_g(float &ax, float &ay, float &az) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(MPU6050_REG_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) return false;

  if (Wire.requestFrom((int)MPU6050_ADDR, 6) < 6) return false;

  int16_t raw_x = (Wire.read() << 8) | Wire.read();
  int16_t raw_y = (Wire.read() << 8) | Wire.read();
  int16_t raw_z = (Wire.read() << 8) | Wire.read();

  ax = raw_x / ACCEL_LSB_PER_G;
  ay = raw_y / ACCEL_LSB_PER_G;
  az = raw_z / ACCEL_LSB_PER_G;
  return true;
}

// Samples the IMU, updates the debounced motion_detected flag, and
// notifies the Linux side (and gates ECG collection) on any state
// change. Called once per MPU_SAMPLE_INTERVAL_US from loop().
void update_motion_state() {
  float ax, ay, az;
  if (!mpu6050_read_accel_g(ax, ay, az)) return;

  float magnitude = sqrtf(ax * ax + ay * ay + az * az);
  float deviation = fabsf(magnitude - accel_baseline_g);

  if (deviation > MOTION_THRESHOLD_G) {
    motion_over_count++;
    motion_under_count = 0;
  } else {
    motion_under_count++;
    motion_over_count = 0;

    // Slowly track the resting orientation/gravity level while
    // still, so simply tilting the sensor doesn't itself look
    // like ongoing motion.
    accel_baseline_g += 0.02f * (magnitude - accel_baseline_g);
  }

  bool was_motion = motion_detected;

  if (!motion_detected && motion_over_count >= MOTION_DEBOUNCE_COUNT) {
    motion_detected = true;
  } else if (motion_detected && motion_under_count >= MOTION_DEBOUNCE_COUNT) {
    motion_detected = false;
  }

  if (motion_detected != was_motion) {
    // Edge-triggered notify -- only sent on state change, not
    // every IMU sample.
    Bridge.notify("motion_state", motion_detected ? 1 : 0);

    if (motion_detected) {
      // A big displacement just started: drop whatever partial
      // batch was being built so we never ship a batch mixing
      // clean samples with motion-corrupted ones.
      ecg_batch.clear();
    }
  }
}

// ============================================================
// SETUP
// ============================================================

void setup() {

  pinMode(ECG_PIN, INPUT);

  // UNO Q Serial is routed through the Router/Monitor system.
  Serial.begin(115200);

  // Initialize Bridge
  Bridge.begin();

  // Reserve memory so the vector does not repeatedly allocate.
  ecg_batch.reserve(BATCH_SIZE);

  // Start sampling immediately
  next_sample_us = micros();
  next_mpu_sample_us = micros();

  mpu6050_ok = mpu6050_init();

  Serial.println();
  Serial.println("========================================");
  Serial.println("       UNO Q ECG ACQUISITION");
  Serial.println("========================================");
  Serial.println("MCU       : STM32U585");
  Serial.println("Input     : BioAmp EXG Pill -> A0");
  Serial.println("Rate      : 125 Hz");
  Serial.println("Batch     : 25 samples");
  Serial.println("Interval  : 200 ms");
  Serial.println("Bridge    : RouterBridge 0.4.3");
  Serial.println("Mode      : Bridge.notify()");
  Serial.println("========================================");
  Serial.println("Filter    : Butterworth Band-Pass");
  Serial.println("Passband  : 0.5 - 44.5 Hz");
  Serial.println("========================================");
  if (mpu6050_ok) {
    Serial.println("Motion    : MPU6050 OK -- displacement gating ON");
  } else {
    Serial.println("Motion    : MPU6050 NOT FOUND -- gating disabled");
    Serial.println("            (check wiring: VCC/GND/SDA/SCL)");
  }
  Serial.println("========================================");
  Serial.println("ECG acquisition started...");
}


// ============================================================
// LOOP
// ============================================================

void loop() {

  unsigned long now = micros();


  // ----------------------------------------------------------
  // Motion sensing (MPU6050) -- runs independently at 50 Hz
  // ----------------------------------------------------------

  if (mpu6050_ok && (long)(now - next_mpu_sample_us) >= 0) {
    next_mpu_sample_us += MPU_SAMPLE_INTERVAL_US;
    update_motion_state();
  }


  // ----------------------------------------------------------
  // Fixed-rate ECG acquisition
  // ----------------------------------------------------------

  if ((long)(now - next_sample_us) >= 0) {

    // Read BioAmp EXG Pill
    int raw = analogRead(ECG_PIN);

    // Apply the 125 Hz Butterworth ECG filter. This keeps running
    // every sample (even during motion) so the filter's internal
    // state stays warm and doesn't produce a transient jump when
    // collection resumes.
    float filtered_ecg = ECGFilter((float)raw);

    last_ecg_value = filtered_ecg;

    // Schedule next sample
    next_sample_us += SAMPLE_INTERVAL_US;

    if (!motion_detected) {

      // Only collect/send samples while the sensor is still. During
      // a big displacement the trace is dominated by motion
      // artifact rather than cardiac signal, so we pause collection
      // and resume automatically once things settle.
      ecg_batch.push_back(filtered_ecg);

      total_samples++;


      // ------------------------------------------------------
      // Batch complete
      // ------------------------------------------------------

      if (ecg_batch.size() >= BATCH_SIZE) {

        /*
         * IMPORTANT:
         *
         * notify() does NOT wait for a response.
         *
         * This is much better for continuous ECG streaming
         * than call().
         */

        Bridge.notify(
          "ecg_batch",
          ecg_batch
        );

        total_batches++;

        // Clear batch without releasing reserved memory
        ecg_batch.clear();
      }
    }


    // --------------------------------------------------------
    // Timing protection
    // --------------------------------------------------------

    if ((long)(now - next_sample_us) >
        (long)SAMPLE_INTERVAL_US) {

      next_sample_us = now + SAMPLE_INTERVAL_US;
    }
  }


  // ==========================================================
  // DEBUG OUTPUT
  // ==========================================================

  unsigned long debug_now = millis();

  if (debug_now - debug_last_ms >= 1000) {

    debug_last_ms = debug_now;

    Serial.print("[ECG DEBUG] Samples: ");
    Serial.print(total_samples);

    Serial.print(" | Batches: ");
    Serial.print(total_batches);

    Serial.print(" | Latest filtered ECG: ");
    Serial.print(last_ecg_value);

    Serial.print(" | Batch size: ");
    Serial.print(ecg_batch.size());

    Serial.print(" | Motion: ");
    Serial.println(motion_detected ? "MOVING (paused)" : (mpu6050_ok ? "still" : "n/a"));
  }
}