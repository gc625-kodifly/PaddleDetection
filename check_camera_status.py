import smbus2
import argparse
import json, uuid, requests
from datetime import datetime

def send__status_to_server(status,cam_label,edge_id):
    payload = {
        "id": 1,
        "equipment": cam_label,
        "status": status,
    }
    DEFAULT_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

    REMOTE_SERVER_URL = (
        "https://pcs-backend.kodifly.com/api/ops/socket-message/"
    )

    try:

        message = {
            "message": payload,
            "message_id": str(uuid.uuid4()),
            "message_received_timestamp": datetime.now().isoformat(),
            "message_type": "EQUIPMENT_STATUS",
            "sender": f"PCS-EDGE-0{edge_id}"
        }
        message = json.dumps(message)
        print(f"trying to send message: {message}")
        response = requests.post(REMOTE_SERVER_URL, data=message, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        print("sent message")
        return True
    except Exception as e:
        print("Error")
        return f"Error: {e}"

# Define the I2C bus number (30 in your case)
I2C_BUS = 30

# Define the addresses you want to check
addresses_to_check = [0x42, 0x44]

orin_1_addresses = {
    'PCS-CAM-01': 0x42,
    'PCS-CAM-02': 0x44
}


orin_2_addresses = {
    'PCS-CAM-03': 0x42, # IDK YET
    'PCS-CAM-04': 0x44, # IDK YET
    'PCS-CAM-05': 0x44, # IDK YET
}






def check_i2c_device(bus, address):
    try:
        # Try to read a byte from the device
        bus.read_byte(address)
        return True
    except OSError:
        # Device is not found or not responding
        return False

def main(device_id):
    # Create an I2C bus instance
    bus = smbus2.SMBus(I2C_BUS)

    print(f"Checking devices for device_id: {device_id}")

    cur_addresses = orin_1_addresses if device_id == 1 else orin_2_addresses

    for cam_label,address in cur_addresses.items():
        if check_i2c_device(bus,address):
            print(f"camera {cam_label} connected at address 0x{address:02X}")
            send__status_to_server(True,cam_label,device_id)
        else:
            print(f"camera {cam_label} disconnected at address 0x{address:02X}")
            send__status_to_server(False,cam_label,device_id)
    # Close the bus when done
    bus.close()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Check I2C devices on the bus.")
    parser.add_argument('device_id', type=int, help='The device ID (string) to identify the check context')
    
    # Parse the arguments
    args = parser.parse_args()

    # Pass the parsed device_id to the main function
    main(args.device_id)
