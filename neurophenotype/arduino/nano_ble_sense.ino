/*
 * Arduino Nano 33 BLE Sense — IMU + GSR streaming sketch
 *
 * Streams over BLE:
 *   - LSM9DS1 accelerometer + gyroscope (wrist-worn motor/tremor)
 *   - Analog A0 GSR voltage (skin conductance)
 *
 * GSR Circuit:
 *   Finger Electrode 1 → 3.3V
 *   Finger Electrode 2 → 100kΩ → GND
 *                      → Analog Pin A0 (Arduino)
 *
 * BLE characteristic UUID: use nRF Connect or matching Python bleak client to read.
 */

#include <Arduino_LSM9DS1.h>
#include <ArduinoBLE.h>

// BLE Service & Characteristics
BLEService sensorService("12345678-1234-1234-1234-123456789abc");
BLECharacteristic imuChar("12345678-1234-1234-1234-123456789abd", BLERead | BLENotify, 24); // 6 floats
BLECharacteristic gsrChar("12345678-1234-1234-1234-123456789abe", BLERead | BLENotify, 4);  // 1 float

const int GSR_PIN = A0;
const int SAMPLE_RATE_MS = 20; // 50 Hz

void setup() {
  Serial.begin(115200);

  if (!IMU.begin()) {
    Serial.println("IMU init failed");
    while (1);
  }

  if (!BLE.begin()) {
    Serial.println("BLE init failed");
    while (1);
  }

  BLE.setLocalName("NeuroPhenotype");
  BLE.setAdvertisedService(sensorService);
  sensorService.addCharacteristic(imuChar);
  sensorService.addCharacteristic(gsrChar);
  BLE.addService(sensorService);
  BLE.advertise();

  Serial.println("BLE advertising as NeuroPhenotype");
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    Serial.print("Connected: ");
    Serial.println(central.address());

    while (central.connected()) {
      float ax, ay, az, gx, gy, gz;

      if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
        IMU.readAcceleration(ax, ay, az);
        IMU.readGyroscope(gx, gy, gz);

        float imuData[6] = {ax, ay, az, gx, gy, gz};
        imuChar.writeValue(imuData, sizeof(imuData));
      }

      int gsrRaw = analogRead(GSR_PIN);
      float gsrVoltage = gsrRaw * (3.3f / 1023.0f);
      gsrChar.writeValue(gsrVoltage);

      delay(SAMPLE_RATE_MS);
    }

    Serial.println("Disconnected");
  }
}
