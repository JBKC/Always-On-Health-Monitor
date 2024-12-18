'''
Combining heartrate and activity predictions for a smartwatch interface
'''

import torch
import numpy as np
import serial_asyncio
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
import logging

serial_port = '/dev/tty.ESP32-Classic-ESP32SPP'
baud_rate = 115200
maxlen = 320  # holds 2 overlapping 8-second sliding windows (10 seconds)
overlap = 64 # 2-second window overlap

mapping = {
    0: "still",
    1: "stair",
    2: "foos",  # table football
    3: "cycle",
    4: "drive",
    5: "lunch",
    6: "walk",
    7: "desk"
}

class SerialProtocol(asyncio.Protocol):
    """
    Asynchronous Serial Protocol for reading raw data
    Appends new data to raw_queue
    """
    def __init__(self, raw_queue):
        super().__init__()
        self.raw_queue = raw_queue
        self.buffer = b''

    def connection_made(self, transport):
        print("Successfully connected to the ESP32 via Bluetooth!")

    def data_received(self, data):
        self.buffer += data
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            line = line.decode('utf-8').strip()
            if line:
                asyncio.create_task(self.raw_queue.put(line))

    def connection_lost(self, exc):
        if exc:
            print(f"Serial connection lost due to error: {exc}")
        else:
            print("Serial connection closed.")

class SampleCounter:
    """
    Simple counter for number of lines received (~31ms)
    """
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1

    def reset(self):
        self.value = 0

async def consumer(raw_queue, buffer, window_queue, counter):
    '''
    Combines producer & consumer roles from previous version
    Reads data as it's added to raw_queue, parses it, then adds to a deque and takes a snapshot for processing
    '''

    print(f'Streaming data....')

    while True:
        # waits for data to be appended to raw_queue
        packet = await raw_queue.get()
        parts = packet.split(',')
        # print(parts)
        buffer.append(tuple((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))))
        counter.increment()

        if counter.value >= overlap:
            if len(buffer) == maxlen:

                # take snapshot from buffer (2 windows) - shape (n_samples, n_channels) = (320, 4)
                snapshot = np.array(buffer)

                # add to window_queue to hold snapshots for concurrent processing
                await window_queue.put(snapshot)
                counter.reset()

        await asyncio.sleep(0)

async def processing(transport, window_queue, hr_model, act_model, hr_output, act_output):
    '''
    Processes snapshots from window_queue
    Does concurrently for both hr & activity predictions
    '''
    while True:
        # get next snapshot from the queue
        snapshot = await window_queue.get()

        # # concurrent processing for both models
        # hr_processing = asyncio.create_task(asyncio.to_thread(realtime_processing.main, snapshot))
        # act_processing = asyncio.create_task(asyncio.to_thread(realtime_activity_processing.main, snapshot, multi=1))
        # # run tasks
        # start = time.time()
        # x_bvp, act = await asyncio.gather(hr_processing, act_processing)
        # end = time.time()
        # print(f"Total processing time: {end - start:.2f} seconds")

        # Run HR Processing
        start_hr = time.time()
        x_bvp = await asyncio.to_thread(realtime_processing.main, snapshot)
        end_hr = time.time()
        print(f"HR processing time: {end_hr - start_hr:.2f} seconds")

        # Run Activity Processing
        start_act = time.time()
        act = await asyncio.to_thread(realtime_activity_processing.main, snapshot, multi=1)
        end_act = time.time()
        print(f"Activity processing time: {end_act - start_act:.2f} seconds")

        # concurrent inference for both models
        hr_inference = asyncio.create_task(asyncio.to_thread(realtime_eval.main, x_bvp, hr_model))
        act_inference = asyncio.create_task(asyncio.to_thread(realtime_activity_eval.main, act, act_model))
        # run tasks
        start = time.time()
        hr_pred, act_int = await asyncio.gather(hr_inference, act_inference)
        end = time.time()
        print(f"Total inference time: {end - start:.2f} seconds")


        act_pred = mapping.get(int(act_int), "Unknown activity")

        hr_output.put(hr_pred)
        act_output.put(act_pred)
        print(f"BPM: {hr_pred:.4f}")
        print(f"Activity: {act_pred}")

        # Write to serial
        output_str = f'h{hr_pred:.2f},a{act_pred}\n'
        transport.write(output_str.encode())

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


    raw_queue = asyncio.Queue()                             # stores raw data as it's received from sensor
    buffer = collections.deque(maxlen=maxlen)               # stores parsed data ready to be snapshotted and turned into model input windows
    window_queue = asyncio.Queue()

    counter = SampleCounter()
    hr_output = asyncio.Queue()
    act_output = asyncio.Queue()


    print(f"Attempting to connect to Bluetooth device at {serial_port}...")
    try:
        loop = asyncio.get_running_loop()
        transport, protocol = await serial_asyncio.create_serial_connection(
            loop,
            lambda: SerialProtocol(raw_queue),
            serial_port,
            baud_rate
        )
    except serial.SerialException as e:
        print(f"Failed to connect to the Bluetooth device: {e}")
        return

    #### extract & process data concurrently
    async with asyncio.TaskGroup() as tg:
        tg.create_task(consumer(raw_queue, buffer, window_queue, counter))
        tg.create_task(processing(transport, window_queue, hr_model, act_model, hr_output, act_output))

    # run tasks
    await asyncio.gather()


if __name__ == '__main__':
    asyncio.run(main())
