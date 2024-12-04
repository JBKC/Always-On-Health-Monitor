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


async def producer(ser, buffer, maxlen, counter):
    """
    Collects streaming data and appends to a buffer (sliding window).
    """

    print(f'Streaming data....')
    temp = ""  # temporary buffer for incomplete lines

    while True:
        if ser.in_waiting > 0:

            # read available data and append to temporary buffer
            raw_data = ser.read(ser.in_waiting).decode('utf-8')
            temp += raw_data

            lines = temp.split("\n")
            temp = lines[-1]                # save last (incomplete) part of buffer for next iteration

            # process complete lines from buffer
            for line in lines[:-1]:

                if line.startswith("ppg:"):
                    ppg = float(line.split(":")[1])

                elif line.startswith("accel:"):
                    accel = tuple(map(float, line.split(":")[1].split(",")))

                    sample = [ppg, *accel]  # Create sample
                    buffer.append(sample)

                    # Maintain sliding window size
                    if len(buffer) > maxlen:
                        buffer.popleft()

                    # increment counter
                    counter[0] += 1

        await asyncio.sleep(0.01)

async def consumer(buffer, maxlen, model, output, counter):
    '''
    Takes snapshot of queue & passes through model to give HR prediction
    '''

    while True:
        if len(buffer) == maxlen and counter[0] >= 64:
            # reset counter as full window received
            counter[0] = 0
            # take snapshot from buffer (2 windows) - shape (n_samples, n_channels) = (320, 4)
            snapshot = np.array(buffer)
            print(snapshot.shape)



            ### include artifact removal / pre-processing to get x_bvp

            # Process the data through the model
            pred = await asyncio.to_thread(realtime_eval.main, snapshot, model)

            ### pin to output buffer

        await asyncio.sleep(0)



async def main():
    '''
    Runs main asynchronous tasks
    '''

    # initialise model
    checkpoint_path = f'../models/temporal_attention_model_session_S7.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model = TemporalAttentionModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # get stream from Arduino
    serial_port = '/dev/cu.usbmodem14101'
    baud_rate = 115200
    ser = serial.Serial(serial_port, baud_rate, timeout=1)

    maxlen = 320                # holds 2 overlapping 8-second sliding windows (10 seconds)

    # buffer (deque) to store data in overlapping windows
    buffer = collections.deque(maxlen=maxlen)
    # queue for HR predictions
    output = asyncio.Queue()
    counter = [0]                   # counter to track 2 seconds (64 samples)

    # extract & process data concurrently
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(producer(ser, buffer, maxlen, counter))
        task2 = tg.create_task(consumer(buffer, maxlen, model, output, counter))

    # run tasks
    await asyncio.gather(task1, task2)


if __name__ == '__main__':
    asyncio.run(main())
