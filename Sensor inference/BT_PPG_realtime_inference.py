'''
Passes PPG sensor & accelerometer input through trained model
Produces real-time heart-rate prediction every 2 seconds, on a sliding 8-second window
ppg sensor = DFRobot_Heartrate
accelerometer = Adafruit_MMA8451
bluetooth module = DSD Tech HM-10
'''

import torch
import numpy as np
import asyncio
import collections
import time
from collections import deque
from bleak import BleakClient
from Heartrate_training_eval.temporal_attention_model import TemporalAttentionModel
import realtime_processing
import realtime_eval


# Global variables
packet_queue = deque()
current_packet = ""  # For accumulating fragments of a single packet

def notification_handler(sender, data):
    """
    Handles incoming BLE data, assembling fragments into complete packets.
    """
    global current_packet

    # Decode and strip incoming data
    fragment = data.decode('utf-8').strip()

    # Append new fragment to the current packet
    current_packet += fragment

    # Process packets only between '/'
    while '/' in current_packet:
        # Find the first '/'
        start_idx = current_packet.find('/')

        # Check if there's another '/' to mark the end of a packet
        end_idx = current_packet.find('/', start_idx + 1)
        if end_idx != -1:
            # Extract a full packet and append it to the queue
            full_packet = current_packet[start_idx + 1:end_idx]
            packet_queue.append(full_packet)

            # Remove the processed packet from `current_packet`
            current_packet = current_packet[end_idx:]
        else:
            # If no end '/', wait for the next fragment
            current_packet = current_packet[start_idx:]
            break


async def producer(buffer, maxlen, counter):
    """
    Processes complete packets from the queue and appends them to the buffer.
    """
    print('Streaming data....')

    while True:
        # Process packets from the queue
        if packet_queue:
            # Get the next complete packet
            packet = packet_queue.popleft()

            # Split the packet into parts
            parts = packet.split(',')

            try:
                # Parse the packet data
                ppg = float(parts[1])
                accel = tuple(float(parts[i]) for i in range(2, 5))
                sample = [ppg, *accel]

                # Append the sample to the buffer
                buffer.append(sample)

                # Maintain sliding window size
                if len(buffer) > maxlen:
                    buffer.popleft()

                # Increment counter
                counter[0] += 1
                # print(sample)

            except (IndexError, ValueError) as e:
                print(f"Error processing packet: {packet}, error: {e}")

        await asyncio.sleep(0)

async def consumer(buffer, maxlen, window_queue, counter):
    '''
    Takes snapshot of queue & saves for processing
    '''
    while True:
        if len(buffer) == maxlen and counter[0] >= 64:
            counter[0] = 0                  # reset counter as full window received

            # take snapshot from buffer (2 windows)
            snapshot = np.array(buffer)

            # add to window_queue to hold snapshots for processing
            await window_queue.put(snapshot)

        await asyncio.sleep(0)

async def processing(window_queue, model, output, send_queue):
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
        # print(f'BPM history: {output}')

        # write prediction back via BLE
        message = f"{hr_pred:.2f}\n".encode('ascii')
        # print(f"Sending raw bytes: {list(message)}")              # examine ASCII characters being sent
        await send_queue.put(message )

        # mark task as done
        window_queue.task_done()


async def send(client, send_queue, char_uuid):

    while True:
        # Get the next message to send
        message = await send_queue.get()
        try:
            await client.write_gatt_char(char_uuid, message)
            # print(f"Sent: {message.strip()}")
        except Exception as e:
            print(f"Failed to send over BLE: {message.strip()}, Error: {e}")
        finally:
            # Mark task as done
            send_queue.task_done()

        await asyncio.sleep(0.1)



async def main():
    '''
    Runs main asynchronous tasks
    '''
    # initialise model
    checkpoint_path = '../models/temporal_attention_model_session_S7.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model = TemporalAttentionModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    DEVICE_UUID = "B7693B48-2AB9-4B61-9E19-FA319FB63B25"
    HM10_CHAR_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

    maxlen = 320                # holds 2 overlapping 8-second windows (320 samples)
    buffer = collections.deque(maxlen=maxlen)   # buffer for data
    window_queue = asyncio.Queue()              # queue for windows
    output = []                                 # HR predictions
    send_queue = asyncio.Queue()                # queue for sending data back to BLE
    counter = [0]                               # sample counter

    async with BleakClient(DEVICE_UUID) as client:
        if client.is_connected:
            print("Bluetooth connection secured")
            await client.start_notify(HM10_CHAR_UUID, notification_handler)

            # while True:
            #     await asyncio.sleep(0)

            # Run producer, consumer, and processing concurrently
            async with asyncio.TaskGroup() as tg:
                task1 = tg.create_task(producer(buffer, maxlen, counter))
                task2 = tg.create_task(consumer(buffer, maxlen, window_queue, counter))
                task3 = tg.create_task(processing(window_queue, model, output, send_queue))
                task4 = tg.create_task(send(client, send_queue, HM10_CHAR_UUID))

            await asyncio.gather(task1, task2, task3, task4)
            # await asyncio.gather(task1)

if __name__ == '__main__':
    asyncio.run(main())
