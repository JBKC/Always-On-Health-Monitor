'''
Combining heartrate and activity predictions for a smartwatch interface
'''

import torch
import numpy as np
import serial
from Heartrate_training_eval.temporal_attention_model import TemporalAttentionModel
from Activity_detection_training_eval.activity_model_cnn2 import AccModel
import realtime_processing
import realtime_eval
import realtime_activity_processing
import realtime_activity_eval
import asyncio
import collections
import time


async def producer(ser, buffer, maxlen, counter):
    """
    Parses streaming data and appends to a buffer (sliding window).
    """

    print(f'Streaming data....')

    while True:
        if ser.in_waiting > 0:
            packet = ser.readline().decode('utf-8').strip()
            parts = packet.split(',')
            ppg = float(parts[1])
            accel = tuple((float(parts[2]), float(parts[3]), float(parts[4])))
            # print(parts)

            sample = [ppg, *accel]
            buffer.append(sample)

            # Maintain sliding window size
            if len(buffer) > maxlen:
                buffer.popleft()

            # increment counter
            counter[0] += 1

            await asyncio.sleep(0)

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

async def processing(ser, window_queue, hr_model, act_model, hr_output, act_output):
    '''
    Processes snapshots from window_queue
    Does concurrently for both hr & activity predictions
    '''
    while True:
        # get next snapshot from the queue
        snapshot = await window_queue.get()

        # concurrent processing for both models
        hr_processing = asyncio.create_task(asyncio.to_thread(realtime_processing.main, snapshot))
        act_processing = asyncio.create_task(asyncio.to_thread(realtime_activity_processing.main, snapshot, multi=1))
        # run tasks
        start = time.time()
        x_bvp, act = await asyncio.gather(hr_processing, act_processing)
        end = time.time()
        print(f"Total processing time: {end - start:.2f} seconds")

        # concurrent inference for both models
        hr_inference = asyncio.create_task(asyncio.to_thread(realtime_eval.main, x_bvp, hr_model))
        act_inference = asyncio.create_task(asyncio.to_thread(realtime_activity_eval.main, act, act_model))
        # run tasks
        start = time.time()
        hr_pred, act_int = await asyncio.gather(hr_inference, act_inference)
        end = time.time()
        print(f"Total inference time: {end - start:.2f} seconds")

        mapping = {
            0: "still",
            1: "stair",
            2: "foos",          # table football
            3: "cycle",
            4: "drive",
            5: "lunch",
            6: "walk",
            7: "desk"
        }
        act_pred = mapping.get(int(act_int), "Unknown activity")

        hr_output.put(hr_pred)
        act_output.put(act_pred)
        print(f"BPM: {hr_pred:.4f}")
        print(f"Activity: {act_pred}")

        try:
            await asyncio.to_thread(ser.write, f'h{hr_pred:.2f},a{act_pred}\n'.encode())
        except Exception as e:
            print(f"Error writing to serial: {e}")


        # mark task as done
        window_queue.task_done()

async def main():
    '''
    Runs main asynchronous tasks
    '''

    # initialise models
    hr_path = f'../models/temporal_attention_model_session_S7.pth'
    hr_checkpoint = torch.load(hr_path, map_location=torch.device('cpu'))
    hr_model = TemporalAttentionModel()
    hr_model.load_state_dict(hr_checkpoint['model_state_dict'])
    hr_model.eval()

    act_path = f'../models/activity_ppgnext_S5.pth'
    act_model = AccModel(in_channels=6, num_classes=8)
    act_model.load_state_dict(torch.load(act_path, map_location=torch.device('cpu')))
    act_model.eval()

    # bluetooth connection
    serial_port = '/dev/tty.ESP32-Classic-ESP32SPP'
    baud_rate = 115200

    print(f"Attempting to connect to Bluetooth device at {serial_port}...")
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        print("Successfully connected to the ESP32 via Bluetooth!")
    except serial.SerialException as e:
        print(f"Failed to connect to the Bluetooth device: {e}")
        return

    maxlen = 320                # holds 2 overlapping 8-second sliding windows (10 seconds)

    buffer = collections.deque(maxlen=maxlen)               # buffer (deque) to store raw data in overlapping windows
    window_queue = asyncio.Queue()                          # queue for storing windows for processing
    hr_output = asyncio.Queue()
    act_output = asyncio.Queue()
    counter = [0]                                           # counter to track 2 seconds (64 samples)

    # extract & process data concurrently
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(producer(ser, buffer, maxlen, counter))
        task2 = tg.create_task(consumer(buffer, maxlen, window_queue, counter))
        task3 = tg.create_task(processing(ser, window_queue, hr_model, act_model, hr_output, act_output))

    # run tasks
    await asyncio.gather(task1, task2, task3)


if __name__ == '__main__':
    asyncio.run(main())
