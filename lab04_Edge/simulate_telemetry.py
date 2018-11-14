# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import time
import os
import pandas as pd
import random
import numpy as np

# Using the Python Device SDK for IoT Hub:
#   https://github.com/Azure/azure-iot-sdk-python
from iothub_client import IoTHubClient, IoTHubClientError, IoTHubTransportProvider, IoTHubClientResult
from iothub_client import IoTHubMessage, IoTHubMessageDispositionResult, IoTHubError, DeviceMethodReturnValue

# The device connection string to authenticate the device with your IoT hub. You can get this from the Azure portal,
# in the device section of your IoT Hub
CONNECTION_STRING = "HostName=wopauliIotHub.azure-devices.net;DeviceId=data_generator;SharedAccessKey=vxOIZYn6VQUXbf5zmMY0USBCR2KWvNb0bTuC8JHxAXQ="

# Using the MQTT protocol.
PROTOCOL = IoTHubTransportProvider.MQTT
MESSAGE_TIMEOUT = 10000

# construct path to telemetry data
BASE_DIR = '..'
DATA_DIR = os.path.join(BASE_DIR, 'data')
DEVICE_FILE = os.path.join(DATA_DIR, 'telemetry.csv')
DEBUG = False

def send_confirmation_callback(message, result, user_context):
    if DEBUG:
        print("IoT Hub responded to message with status: %s" % (result))


def iothub_client_init():
    # Create an IoT Hub client
    client = IoTHubClient(CONNECTION_STRING, PROTOCOL)
    return client


def iothub_client_telemetry_sample_run():

    print("Reading data ... ")
    telemetry = pd.read_csv(DEVICE_FILE)
    telemetry = telemetry.rename(str, columns={'datetime': 'timestamp'})
    timestamps = telemetry['timestamp'].unique()
    print("Done.")

    try:
        client = iothub_client_init()
        print ("IoT Hub device sending periodic messages, press Ctrl-C to exit")

        for timestamp in timestamps:
            df = telemetry.loc[telemetry.loc[:, 'timestamp'] == timestamp, :]
            df = df.sample(frac=1).reset_index(drop=True)

            for r in range(0, df.shape[0]):
                row = df.iloc[r, :]
                message_txt = '{'
                for c, column in enumerate(row.index.values):
                    if column == 'timestamp':
                        message_txt += '\"%s\": \"%s\", ' % (column, row.loc[column])
                    else:
                        message_txt += '\"%s\": %s, ' % (column, row.loc[column])
                a_random_number = random.uniform(0,1)
                if a_random_number > .2:
                    message_txt += '\"%s\": %s, ' % ('volt_an', 'NaN')
                else:
                    message_txt += '\"%s\": %s, ' % ('volt_an', row.loc['volt'])
                message_txt = message_txt.strip(', ') + '}'
                message = IoTHubMessage(message_txt)
                if DEBUG:
                    print( "Sending message: %s" % message.get_string())
                client.send_event_async(message, send_confirmation_callback, None)
                time.sleep(1)

    except IoTHubError as iothub_error:
        print ("Unexpected error %s from IoTHub" % iothub_error)
        return
    except KeyboardInterrupt:
        print ( "IoTHubClient sample stopped" )


if __name__ == '__main__':
    print("IoT Hub Quickstart #1 - Simulated device")
    print("Press Ctrl-C to exit")
    iothub_client_telemetry_sample_run()
