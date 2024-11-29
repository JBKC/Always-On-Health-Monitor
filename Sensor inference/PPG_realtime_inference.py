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



async def producer(ser, window):
    '''
    Reads raw data and saves in dictionary
    '''


    return

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

    # create queue to store data (8 second windows)
    maxlen = 256
    window = collections.deque(maxlen=maxlen)

    async with asyncio.TaskGroup() as tg:
        #
        task1 = tg.create_task(producer(ser, window))
        task2 = ...

    # run tasks
    await asyncio.gather(task1, task2)




if __name__ == '__main__':
    asyncio.run(main())
