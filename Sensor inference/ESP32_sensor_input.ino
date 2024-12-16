// Equipment used:
// ESP32 microcontroller
// S-0018 0.96inch OLED display
// ppg sensor = DFRobot_Heartrate
// accelerometer = Adafruit_MMA8451

#include <Wire.h>
#include <Adafruit_MMA8451.h>
#include <Adafruit_Sensor.h>
#include <LiquidCrystal.h>
#include "BluetoothSerial.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

BluetoothSerial SerialBT;       // bluetooth object
Adafruit_MMA8451 mma = Adafruit_MMA8451();

float conv = 9.80665;

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// set interval to ensure sampling rate of 32
unsigned long lastTime = 0;
const unsigned long interval = 1000 / 32;

void setup() {

  Serial.begin(115200);

  if (!SerialBT.begin("ESP32-Classic")) {
    Serial.println("Failed to initialize Bluetooth!");
    while (1); // Halt the program
  }
  Serial.println("Bluetooth started! Now discoverable as 'ESP32-Classic'.");

    if (!mma.begin()) {
    Serial.println("Could not find accelerometer");
    while (1);
  }

  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { // Address 0x3D for 128x64
    Serial.println(F("SSD1306 allocation failed"));
    for(;;);
  }
  delay(2000);
  display.clearDisplay();

  display.setTextSize(2);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
  display.println("BPM:");
  display.setCursor(0, 30);
  display.println("ACT:");
  display.display(); 

  mma.setRange(MMA8451_RANGE_2_G); // Set accelerometer range
}

void loop() {

  unsigned long currentMillis = millis(); // Capture the current time
  String bpm = ""; // Buffer for heart rate
  String act = ""; // Buffer for activity
  bool isBpm = false; // Flag to track which type of message is being received
  bool isAct = false;

  if (millis() - lastTime >= interval) {
    lastTime = millis();

    // Read the analog value from GPIO36 (VP)
    float ppg = analogRead(36); // GPIO36 is the VP pin

    sensors_event_t event;
    mma.getEvent(&event);
    float ax = event.acceleration.x / conv;
    float ay = event.acceleration.y / conv;
    float az = event.acceleration.z / conv;

    SerialBT.println(String(currentMillis) + "," + String(ppg, 2) + "," + String(ax, 2) + "," + String(ay, 2) + "," + String(az, 2));

  }
  // Add data from both outputs to LCD
  while (SerialBT.available() > 0) {
    char received = SerialBT.read();

    // Parse incoming data
    if (received == 'h') { // Start of a heart rate message
        isBpm = true;
        isAct = false; // Reset activity flag
        bpm = ""; // Clear the heart rate buffer
    } else if (received == 'a') { // Start of an activity message
        isAct = true;
        isBpm = false; // Reset heart rate flag
        act = ""; // Clear the activity buffer
    } else if (received == ',') { // Delimiter between heart rate and activity
        isBpm = false; // Finish heart rate message
        isAct = true;  // Start activity message
    } else if (received == '\n') { // End of message
        if (!bpm.isEmpty()) { // If heart rate data is complete
            display.fillRect(50, 0, 128, 30, BLACK); // Clear heart rate display area
            display.setCursor(50, 0);
            display.println(bpm); // Display heart rate
        }
        if (!act.isEmpty()) { // If activity data is complete
            display.fillRect(50, 30, 128, 60, BLACK); // Clear activity display area
            display.setCursor(50, 30);
            display.println(act); // Display activity
        }
        display.display();

        // Clear buffers for next update
        bpm = "";
        act = "";
        isBpm = false;
        isAct = false;
    } else {
        // Append received characters to the appropriate buffer
        if (isBpm) {
            bpm += received;
        } else if (isAct) {
            act += received;
        }
    }
}
}
