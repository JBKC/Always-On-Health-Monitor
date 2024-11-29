#include <Wire.h>
#include <Adafruit_MMA8451.h>
#include <Adafruit_Sensor.h>

// initialise accelerometer
Adafruit_MMA8451 mma = Adafruit_MMA8451();

// convert from m/s**2 to g
float conv = 9.80665;

void setup() {
    Serial.begin(115200);

        if (!mma.begin()) {
      Serial.println("Could not find accelerometer");
      while (1);
      }

    mma.setRange(MMA8451_RANGE_2_G); // set accelerometer range
}

void loop() {
 
    // get ppg data directly from pin
    float ppg = analogRead(A0);
    Serial.println(ppg, 2);     // 2 decimal places

    // accelerometer data
    sensors_event_t event;
    mma.getEvent(&event);
    
    float ax = event.acceleration.x / conv;
    float ay = event.acceleration.y / conv;
    float az = event.acceleration.z / conv;

    // Send the data in the requested format: ax, ay, az
    Serial.print(ax,2);
    Serial.print(",");
    Serial.print(ay,2);
    Serial.print(",");
    Serial.println(az,2);


}
