'''
Passes PPG sensor & accelerometer input through trained model to produce real-time heart-rate inference
Sensors:
- ppg = DFRobot_Heartrate: fs=
- accelerometer = Adafruit_MMA8451: fs=
'''

import torch
import serial
from Heartrate_training_eval.temporal_attention_model import TemporalAttentionModel
import time


def clean_stream():

    # pass through ma removal etc

    return



def main():

    def downsample():
        return

    serial_port = '/dev/cu.usbmodem14101'  # Replace with your Arduino's port (e.g., 'COM3' on Windows)
    baud_rate = 115200

    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        print(f"Connected to {serial_port} at {baud_rate} baud")
        time.sleep(2)  # Wait for Arduino to reset
    except serial.SerialException as e:
        print(f"Error: {e}")
        exit()

    while True:
        if ser.in_waiting > 0:  # Check if data is available
            data = ser.readline().decode('utf-8').strip()  # Read and decode the data
            print(data)




    #
    # # initialise model
    # checkpoint = torch.load('../models/temporal_attention_model_full_augment_session_S6.pth')
    # model = TemporalAttentionModel()  # Update with any required parameters for your model initialization
    #
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()

if __name__ == '__main__':
    main()
