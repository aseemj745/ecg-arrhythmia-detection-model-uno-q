/*
 * ============================================================
 * ECG UNO Q - STM32 MCU SIDE
 * ============================================================
 *
 * BioAmp EXG Pill -> A0
 *
 * Sampling:
 *   250 Hz
 *
 * Bridge:
 *   MCU -> Linux MPU  (received by python/main.py's on_ecg_batch)
 *
 * Transfer:
 *   25 ECG samples per Bridge notification
 *   = 100 ms of ECG data
 *
 * Arduino_RouterBridge:
 *   0.4.3
 *
 * ============================================================
 */

#include <Arduino_RouterBridge.h>
#include <vector>

// ------------------------------------------------------------
// Configuration
// ------------------------------------------------------------

const int ECG_PIN = A0;

const unsigned long FS_HZ = 250;
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / FS_HZ;

// 25 samples = 100 ms at 250 Hz -- must match FS_HZ/WINDOW_S assumptions
// in python/main.py (FS_HZ = 250.0 there too).
const int BATCH_SIZE = 25;


// ------------------------------------------------------------
// Timing
// ------------------------------------------------------------

unsigned long next_sample_us = 0;


// ------------------------------------------------------------
// ECG batch
// ------------------------------------------------------------

std::vector<int> ecg_batch;

unsigned long total_samples = 0;
unsigned long total_batches = 0;


// ------------------------------------------------------------
// Debug timing
// ------------------------------------------------------------

unsigned long debug_last_ms = 0;
int last_ecg_value = 0;


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

  Serial.println();
  Serial.println("========================================");
  Serial.println("       UNO Q ECG ACQUISITION");
  Serial.println("========================================");
  Serial.println("MCU       : STM32U585");
  Serial.println("Input     : BioAmp EXG Pill -> A0");
  Serial.println("Rate      : 250 Hz");
  Serial.println("Batch     : 25 samples");
  Serial.println("Interval  : 100 ms");
  Serial.println("Bridge    : RouterBridge 0.4.3");
  Serial.println("Mode      : Bridge.notify()");
  Serial.println("Key       : ecg_batch  <-- must match python/main.py");
  Serial.println("========================================");
  Serial.println("ECG acquisition started...");
}


// ============================================================
// LOOP
// ============================================================

void loop() {

  unsigned long now = micros();


  // ----------------------------------------------------------
  // Fixed-rate ECG acquisition
  // ----------------------------------------------------------

  if ((long)(now - next_sample_us) >= 0) {

    // Read BioAmp
    int raw = analogRead(ECG_PIN);

    last_ecg_value = raw;

    // Add sample to batch
    ecg_batch.push_back(raw);

    total_samples++;

    // Schedule next sample
    next_sample_us += SAMPLE_INTERVAL_US;


    // --------------------------------------------------------
    // Batch complete
    // --------------------------------------------------------

    if (ecg_batch.size() >= BATCH_SIZE) {

      /*
       * IMPORTANT:
       *
       * notify() does NOT wait for a response.
       *
       * This is much better for continuous ECG streaming
       * than call().
       *
       * The key "ecg_batch" here MUST match the Bridge.provide("ecg_batch", ...)
       * registration in python/main.py -- that is the entire link between
       * the MCU and the dashboard.
       */

      Bridge.notify(
        "ecg_batch",
        ecg_batch
      );

      total_batches++;

      // Clear batch without releasing reserved memory
      ecg_batch.clear();
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

    Serial.print(" | Latest ADC: ");
    Serial.print(last_ecg_value);

    Serial.print(" | Batch size: ");
    Serial.println(ecg_batch.size());
  }
}
