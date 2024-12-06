#include <Wire.h>
#include <Adafruit_MMA8451.h>
#include <Adafruit_Sensor.h>
#include <LiquidCrystal.h>


// initialise accelerometer
Adafruit_MMA8451 mma = Adafruit_MMA8451();
LiquidCrystal lcd(2, 3, 4, 5, 6, 7); // RS, E, D4, D5, D6, D7

// convert from m/s**2 to g
float conv = 9.80665;
String bpm = ""; // Buffer for incoming data


// set interval to ensure sampling rate of 32
unsigned long lastTime = 0;
const unsigned long interval = 1000 / 32;

void setup() {
  // Set up the LCD
  lcd.begin(16, 2);
  lcd.setCursor(0, 0);
  lcd.print("BPM:");
  lcd.setCursor(0, 1);
  lcd.print("Waiting...");
  analogWrite(12, 100); // Brightness level (0 to 255)

  Serial.begin(115200);

  if (!mma.begin()) {
    Serial.println("Could not find accelerometer");
    while (1);
  }
  mma.setRange(MMA8451_RANGE_2_G); // Set accelerometer range
}

void loop() {

    unsigned long currentMillis = millis(); // Capture the current time

    if (millis() - lastTime >= interval) {
      lastTime = millis();
 
      // get ppg data directly from pin
      float ppg = analogRead(A0);
      // accelerometer data
      sensors_event_t event;
      mma.getEvent(&event);
      float ax = event.acceleration.x / conv;
      float ay = event.acceleration.y / conv;
      float az = event.acceleration.z / conv;

      // Send the data
      Serial.println(String(currentMillis) + "," + String(ppg, 2) + "," + String(ax, 2) + "," + String(ay, 2) + "," + String(az, 2));

    }

      // Receive data from HM-10
  while (Serial.available() > 0) {
    char receivedChar = Serial.read();
    
    if (receivedChar == '\n') {
      // End of message
      lcd.setCursor(0, 1);
      lcd.print("               "); // Clear second line
      lcd.setCursor(0, 1);
      lcd.print(bpm); // Display the new BPM data
      bpm = ""; // Clear the buffer
    } else {
      bpm += receivedChar; // Append character to the buffer
    }
  }
}

