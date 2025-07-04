import os
import sys
import trials.data.cyPyWinUSB as hid
import queue
from Crypto.Cipher import AES
import signal
import time


tasks = queue.Queue()

running = True
end_timestamp = 0

def signal_handler(sig, frame):
    global running
    global end_timestamp
    if end_timestamp != 0:
        print("force stop")
        sys.exit(0)
    end_timestamp = time.time_ns()
    print("stoped at: ", end_timestamp)
    running = False


signal.signal(signal.SIGINT, signal_handler)

class EEG(object):
    # Class variable to track all instances
    _instances = []
    
    def __init__(self):
        print("Initializing EEG device...")
        
        # Add this instance to the list BEFORE closing others
        if self not in EEG._instances:
            EEG._instances.append(self)
        
        # Close any existing instances to prevent multiple data handlers
        # BUT SKIP THIS INSTANCE (which we just added to the list)
        for instance in EEG._instances[:]:
            if instance != self and instance.hid:
                try:
                    print("Closing previous EEG instance to prevent duplicate data collection")
                    instance.close()
                    if instance in EEG._instances:
                        EEG._instances.remove(instance)
                except Exception as e:
                    print(f"Error closing previous EEG instance: {e}")
        
        # Clear the global tasks queue to ensure clean state
        with tasks.mutex:
            tasks.queue.clear()
        
        # Rest of initialization
        self.mask = {}
        self.mask[0] = [10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]
        self.mask[1] = [28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9]
        self.mask[2] = [46, 47, 32, 33, 34, 35, 36, 37, 38, 39, 24, 25, 26, 27]
        self.mask[3] = [48, 49, 50, 51, 52, 53, 54, 55, 40, 41, 42, 43, 44, 45]
        self.mask[4] = [66, 67, 68, 69, 70, 71, 56, 57, 58, 59, 60, 61, 62, 63]
        self.mask[5] = [84, 85, 86, 87, 72, 73, 74, 75, 76, 77, 78, 79, 64, 65]
        self.mask[6] = [102, 103, 88, 89, 90, 91, 92, 93, 94, 95, 80, 81, 82, 83]
        self.mask[7] = [140, 141, 142, 143, 128, 129, 130, 131, 132, 133, 134, 135, 120, 121]
        self.mask[8] = [158, 159, 144, 145, 146, 147, 148, 149, 150, 151, 136, 137, 138, 139]
        self.mask[9] = [160, 161, 162, 163, 164, 165, 166, 167, 152, 153, 154, 155, 156, 157]
        self.mask[10] = [178, 179, 180, 181, 182, 183, 168, 169, 170, 171, 172, 173, 174, 175]
        self.mask[11] = [196, 197, 198, 199, 184, 185, 186, 187, 188, 189, 190, 191, 176, 177]
        self.mask[12] = [214, 215, 200, 201, 202, 203, 204, 205, 206, 207, 192, 193, 194, 195]
        self.mask[13] = [216, 217, 218, 219, 220, 221, 222, 223, 208, 209, 210, 211, 212, 213]
        self.insight_1 = [0,8,14,22,28,36,42,50,56,64,70,78,84,92,98,106,112,120,126,134,140,148,154,162,168,176,182,190,196,204,210,218,224,232,238]
        self.insight_2 = [0,8,14,22,28,36,42,50,56,64]

        self.hid = None
        self.delimiter = ", "
        self.integer = True
        self.data_handler_set = False  # Track if data handler is set
        self.serial_number = None
        devicesUsed = 0
    
        # List all devices to help with debugging
        print("Scanning for HID devices...")
        all_devices = hid.find_all_hid_devices()
        print(f"Found {len(all_devices)} HID devices:")
        for idx, device in enumerate(all_devices):
            print(f"  Device {idx+1}: {device.product_name} (Vendor ID: {device.vendor_id}, Product ID: {device.product_id})")
    
        # Look for EEG device
        for device in all_devices:
            if device.product_name == 'EEG Signals':
                print(f"Found EEG device: {device.product_name} (Serial: {device.serial_number})")
                devicesUsed += 1
                try:
                    device.open()
                    self.hid = device
                    self.serial_number = device.serial_number
                    print("Successfully opened EEG device connection")
                except Exception as e:
                    print(f"Error opening EEG device: {e}")
                    continue
    
        if devicesUsed == 0:
            print("EEG CANNOT BE FOUND!! Make sure device is connected and powered on.")
            # Check if there's a similar device that might be the EEG
            for device in all_devices:
                if 'EEG' in device.product_name or 'emotiv' in device.product_name.lower():
                    print(f"Possible EEG device found with different name: {device.product_name}")
            
            self.hid = None  # Set hid to None if device not found
            return  # Exit the constructor

        if not self.hid:
            print("Failed to open connection to EEG device")
            return

        if not self.serial_number:
            print("Warning: EEG device found but serial number is missing")
            self.serial_number = "UNKNOWN"  # Provide a default to prevent crashes
        
        # Initialize encryption
        try:
            sn = bytearray()
            for i in range(0,len(self.serial_number)):
                sn += bytearray([ord(self.serial_number[i])])

            k = ['\0'] * 16 

            k = [sn[-1],00,sn[-2],72,sn[-1],00,sn[-2],84,sn[-3],16,sn[-4],66,sn[-3],00,sn[-4],80]
            self.samplingRate = 128
            self.channels = 40
            self.key = bytearray(k)
            self.cipher = AES.new(self.key, AES.MODE_ECB)
            
            # Set the data handler
            if self.hid:
                self.hid.set_raw_data_handler(self.dataHandler)
                self.data_handler_set = True
                print("EEG device initialized successfully and data handler set")
        except Exception as e:
            print(f"Error during EEG initialization: {e}")
            if self.hid:
                try:
                    self.hid.close()
                except:
                    pass
                self.hid = None

    def dataHandler(self, data):
        try:
            join_data = ''.join(map(chr, data[1:]))
            decrypted_data = self.cipher.decrypt(bytes(join_data,'latin-1')[0:32])
            tasks.put(decrypted_data)
        except Exception as e:
            print(f"Error in data handler: {e}")

    def convertEPOC(self, data, bits):
        
        level = 0
        for i in range(13, -1, -1):
            
            level <<= 1
            b = int((bits[i] / 8) + 0) # Added int() getting floats?
            o = bits[i] % 8
            level |= (data[b] >> o) & 1

        return level

    def convertEPOC_PLUS(self, value_1, value_2):
        edk_value = "%.8f" % (((int(value_1) * .128205128205129) + 4201.02564096001) + ((int(value_2) -128) * 32.82051289))
        if self.integer == True:
            return str(int(float(edk_value)))
        return edk_value
         
    def get_data(self):
        """Get data from the queue and convert it."""
        if not self.hid:
            return None
        
        if tasks.empty():
            return None
       
        data = tasks.get()
        packet_data = ""
        counter_data = str(data[0]) + self.delimiter
        try:
            for i in range(0,14):
                packet_data = packet_data + str(self.convertEPOC(data[1:], self.mask[i])) + self.delimiter
                          
            packet_data = packet_data[:-len(self.delimiter)]
            return str(counter_data + packet_data)

        except Exception as exception2:
            print(f"Error in get_data: {exception2}")
            return None

    def clear_data(self):
        """Clear all data from the queue."""
        print("Clearing EEG data queue")
        with tasks.mutex:
            tasks.queue.clear()

    def close(self):
        """Close the EEG connection and clean up resources."""
        print("Closing EEG connection...")
        if self.hid:
            # First remove the data handler to stop adding data to the queue
            if self.data_handler_set:
                try:
                    self.hid.set_raw_data_handler(None)
                    self.data_handler_set = False
                    print("Data handler removed")
                except Exception as e:
                    print(f"Error removing data handler: {e}")
            
            # Clear any remaining data in the queue
            with tasks.mutex:
                tasks.queue.clear()
            
            # Close the HID device
            try:
                self.hid.close()
                print("EEG connection closed successfully")
            except Exception as e:
                print(f"Error closing HID device: {e}")
            
            self.hid = None
            
            # Remove from instances list
            if self in EEG._instances:
                EEG._instances.remove(self)
        else:
            print("EEG connection was never opened")


#Don't use, experimental
    def get_data2(self):
        z = ''

        data = tasks.get()
        packet_data = ""
        
        for i in range(1, len(data)):
            z = z + format(data[i],'08b')
                                
        for i in range(2, len(self.insight_1),2):
            i_1 = self.insight_1[(i-2)]
            i_2 = self.insight_1[(i-1)]
                            
            if i_2 > len(z):
                i = len(self.insight_1)
                continue

            v_1 = '0b' + z[(i_1):(i_2)]
            v_2 = '0b' + z[(i_2):(i_2+6)]

            packet_data = packet_data + self.convertEPOC_PLUS(str(int(eval(v_2))), str(int(eval(v_1)))) + self.delimiter
            return str(packet_data)

    @classmethod
    def get_active_headset(cls):
        """Get an active EEG instance or create a new one."""
        # First look for any existing instance
        for instance in cls._instances:
            if instance.hid:
                print("Returning existing active EEG headset")
                return instance
        
        # If no active instance found, create a new one
        print("No active EEG headset found, creating new instance")
        return EEG()