'''
Passes PPG sensor & accelerometer input through trained model
Produces prediction of current activity every 2 seconds
ppg sensor = DFRobot_Heartrate
accelerometer = Adafruit_MMA8451
'''

import torch
import numpy as np
import serial
from Activity_detection_training_eval.activity_model_cnn2 import AccModel
import realtime_activity_processing
import realtime_activity_eval
import asyncio
import collections

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

            sample = [ppg, *accel]
            buffer.append(sample)
            # print(sample)

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

            # take snapshot from buffer (2 windows) - shape (n_samples, n_channels) = (256, 4)
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

        # preprocessing: snapshot.shape = (256,4), out.shape = (1,6,256)
        out = await asyncio.to_thread(realtime_activity_processing.main, snapshot)

        # activity prediction - returns number from 0-7
        pred = await asyncio.to_thread(realtime_activity_eval.main, out, model)

        mapping = {
            0: "sitting still",
            1: "stairs",
            2: "table football",
            3: "cycling",
            4: "driving",
            5: "lunch break",
            6: "walking",
            7: "working at desk"
        }

        activity = mapping.get(int(pred), "Unknown activity")

        print(f"Activity: {activity}")

        # pin prediction to output buffer
        output.append(pred)
        print(f'Activity history: {output}')

        ser.write(f'{activity}\n'.encode())

        # mark task as done
        window_queue.task_done()

async def main():
    '''
    Runs main asynchronous tasks
    '''

    # initialise model
    checkpoint_path = f'../models/activity_ppgnext_S5.pth'

    model = AccModel(in_channels=6, num_classes=8)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()

    serial_port = '/dev/tty.ESP32-Classic-ESP32SPP'
    baud_rate = 115200

    # Check Bluetooth connection
    print(f"Attempting to connect to Bluetooth device at {serial_port}...")
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        print("Successfully connected to the ESP32 via Bluetooth!")
    except serial.SerialException as e:
        print(f"Failed to connect to the Bluetooth device: {e}")
        return

    maxlen = 256                # holds single 8 second window

    buffer = collections.deque(maxlen=maxlen)               # buffer (deque) to store data in overlapping windows
    window_queue = asyncio.Queue()                          # queue for storing windows for processing
    output = []                                             # list of predictions
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
