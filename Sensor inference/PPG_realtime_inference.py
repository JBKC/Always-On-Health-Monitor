'''
Passes PPG sensor & accelerometer input through trained model to produce real-time heart-rate inference
ppg sensor = DFRobot_Heartrate
accelerometer = Adafruit_MMA8451
'''

import torch
import serial
from Heartrate_training_eval.temporal_attention_model import TemporalAttentionModel
import asyncio
import collections
import time
import numpy as np


async def producer(ser, sliding_window):
    """
    Collects streaming data and appends to a sliding window
    """
    buffer_data = ""  # Temporary buffer for incomplete lines

    while True:
        if ser.in_waiting > 0:
            try:
                raw_data = ser.read(ser.in_waiting).decode('utf-8')
                buffer_data += raw_data
                lines = buffer_data.split("\n")
                buffer_data = lines[-1]  # Keep incomplete line

                for line in lines[:-1]:
                    if line.startswith("ppg:"):
                        ppg = float(line.split(":")[1])
                    elif line.startswith("accel:"):
                        accel = tuple(map(float, line.split(":")[1].split(",")))

                        sample = [ppg, *accel]
                        sliding_window.append(sample)

                        # Append the new sample to the sliding window
                        if len(sliding_window) > 256:
                            sliding_window.popleft()

                        print(sample)

            except Exception as e:
                print(f"Error: {e}")

        await asyncio.sleep(0.01)



async def consumer(window, output):
    '''
    Processes data on queue + passes through model to give HR prediction
    '''




async def main():
    '''
    Runs main asynchronous tasks
    '''

    # initialise model
    checkpoint = torch.load('../models/temporal_attention_model_full_augment_session_S6.pth')
    model = TemporalAttentionModel()  # Update with any required parameters for your model initialization
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # get stream from Arduino
    serial_port = '/dev/cu.usbmodem14101'
    baud_rate = 115200
    ser = serial.Serial(serial_port, baud_rate, timeout=1)

    # buffer (deque) to store data in an 8-second sliding window
    buffer = collections.deque(maxlen=256)
    # queue for HR predictions
    output = asyncio.Queue()

    # extract + process data concurrently
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(producer(ser, buffer))
        # task2 = tg.create_task(consumer(window))

    # run tasks
    await asyncio.gather(task1)


if __name__ == '__main__':
    asyncio.run(main())
