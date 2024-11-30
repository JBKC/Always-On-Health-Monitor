'''
Passes PPG sensor & accelerometer input through trained model to produce real-time heart-rate inference
ppg sensor = DFRobot_Heartrate
accelerometer = Adafruit_MMA8451
'''

import torch
import numpy as np
import serial
from Heartrate_training_eval.temporal_attention_model import TemporalAttentionModel
import realtime_eval
import asyncio
import collections
import time
import traceback


async def producer(ser, buffer, maxlen):
    """
    Collects streaming data and appends to a buffer (sliding window).
    """
    temp = ""  # Temporary buffer for incomplete lines

    while True:
        if ser.in_waiting > 0:

            # Read available data and append to temp
            raw_data = ser.read(ser.in_waiting).decode('utf-8')
            temp += raw_data

            # Split data into lines
            lines = temp.split("\n")
            temp = lines[-1]  # Keep incomplete line for the next iteration

            # Process complete lines
            for line in lines[:-1]:

                if line.startswith("ppg:"):
                    ppg = float(line.split(":")[1])

                elif line.startswith("accel:"):
                    accel = tuple(map(float, line.split(":")[1].split(",")))

                    sample = [ppg, *accel]  # Create sample
                    buffer.append(sample)
                    print(sample)

                    # Maintain sliding window size
                    if len(buffer) > maxlen:
                        buffer.popleft()


        await asyncio.sleep(0.01)

async def consumer(buffer, maxlen, model, output):
    '''
    Takes snapshot of queue & passes through model to give HR prediction
    '''

    ### need to create another queue that saves snapshots - for the case where inference of each window takes >2 seconds

    while True:
        if len(buffer) == maxlen:
            buffer = np.array(buffer)

            ### include artifact removal / pre-processing to get x_bvp

            # Process the data through the model
            pred = await asyncio.to_thread(realtime_eval.main(buffer, model))

            print(f"Heart rate prediction: {pred}")

            ### pin to output buffer

        await asyncio.sleep(0)



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

    maxlen = 320                # holds 2 windows

    # buffer (deque) to store data in 2 overlapping 8-second sliding windows
    buffer = collections.deque(maxlen=maxlen)
    # queue for HR predictions
    output = asyncio.Queue()

    # extract + process data concurrently
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(producer(ser, buffer, maxlen))
        task2 = tg.create_task(consumer(buffer, maxlen, model, output))

    # run tasks
    await asyncio.gather(task1, task2)


if __name__ == '__main__':
    asyncio.run(main())
