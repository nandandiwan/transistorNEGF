from device import Device
from rgf import GreensFunction

device = Device()
rgf = GreensFunction(device_state=device)

rgf.rgf(0,0.1)