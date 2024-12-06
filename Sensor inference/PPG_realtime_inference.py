'''
Passes PPG sensor & accelerometer input through trained model
Produces real-time heart-rate prediction every 2 seconds, on a sliding 8-second window
ppg sensor = DFRobot_Heartrate
accelerometer = Adafruit_MMA8451
'''

import torch
import numpy as np
import serial
from Heartrate_training_eval.temporal_attention_model import TemporalAttentionModel
import realtime_processing
import realtime_eval
import asyncio
import collections
from bleak import BleakClient, BleakScanner
import uuid
import sys
import platform
import re

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

async def consumer(buffer, maxlen, window_queue, counter):
    '''
    Takes snapshot of queue & saves for processing
    '''

    while True:
        if len(buffer) == maxlen and counter[0] >= 64:
            counter[0] = 0                  # reset counter as full window received

            # take snapshot from buffer (2 windows) - shape (n_samples, n_channels) = (320, 4)
            snapshot = np.array(buffer)

            # add to window_queue to hold snapshots for concurrent processing
            await window_queue.put(snapshot)

        await asyncio.sleep(0)

async def processing(ser, window_queue, model, output):
    '''
    Processes snapshots from window_queue and appends predictions to the output
    '''
    while True:
        # get next snapshot from the queue
        snapshot = await window_queue.get()

        # motion artifact removal
        x_bvp = await asyncio.to_thread(realtime_processing.main, snapshot)

        # HR inference
        hr_pred = await asyncio.to_thread(realtime_eval.main, x_bvp, model)
        print(f"BPM: {hr_pred:.4f}")

        # pin prediction to output buffer
        output.append(hr_pred)
        print(f'BPM history: {output}')

        ser.write(f'{hr_pred:.4f}\n'.encode())

        # mark task as done
        window_queue.task_done()

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

    ## option 1 = USB
    serial_port = '/dev/cu.usbmodem14101'
    baud_rate = 9600
    ser = serial.Serial(serial_port, baud_rate, timeout=1)

    # Connection is confirmed; proceed with tasks
    print("Running model inference...")

    maxlen = 320                # holds 2 overlapping 8-second sliding windows (10 seconds)

    buffer = collections.deque(maxlen=maxlen)               # buffer (deque) to store data in overlapping windows
    window_queue = asyncio.Queue()                          # queue for storing windows for processing
    output = []                                             # list of HR predictions
    counter = [0]                                           # counter to track 2 seconds (64 samples)

    # extract & process data concurrently
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(producer(ser, buffer, maxlen, counter))
        task2 = tg.create_task(consumer(buffer, maxlen, window_queue, counter))
        task3 = tg.create_task(processing(ser, window_queue, model, output))

    # run tasks
    await asyncio.gather(task1, task2, task3)


if __name__ == '__main__':
    asyncio.run(main())
