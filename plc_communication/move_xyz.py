import snap7
import struct
import snap7.client as client

# Constants for connection
PLC_IP = '192.168.0.1'  # Replace with your PLC's IP address
RACK = 0  # Rack number
SLOT = 1  # Slot number

plc = client.Client()
plc.connect(PLC_IP, RACK, SLOT)

def read_DB_number(db_number, data, length):
	reading = plc.db_read(db_number, data, length)
	value = struct.unpack('>f', reading)  # big-endian
	return value[0]

def write_DB_number(db_number, start_address, value):
	plc.db_write(db_number, start_address, value)

def writeBool_plc(db_number, start_offset, bit_offset, value):
    reading = plc.db_read(db_number, start_offset, 1)
    snap7.util.set_bool(reading, 0, bit_offset, value)
    plc.db_write(db_number, start_offset, reading)
    return None

def readBool_plc(db_number, start_offset, bit_offset):
    reading = plc.db_read(db_number, start_offset, 1)
    a = snap7.util.get_bool(reading, 0, bit_offset)
    return a

# Setup DB
STEPS_PER_REVOLUTION = 6400  # Number of steps per revolution

# DB Numbers in PLC
motor_x = 1
motor_y = 2
motor_z = 3

# Common Bool bit index
power = 0
move_Absolute = 1
move_Relative = 2
cong_tac = 3
jog_forward = 4
jog_back = 5
home = 6
is_done = 7

# Common data byte index
distance_Abs = 2
distance_Rel = 6
velocity = 10
real_position = 14

distance_Abs = 2

def convert_mm_x(motor, coordination_x):
    position_in_pulse = read_DB_number(motor, coordination_x, 4)
    position_in_mm = position_in_pulse * -2 / STEPS_PER_REVOLUTION
    return position_in_mm

def convert_mm_yz(motor, coordination_yz):
    position_in_pulse = read_DB_number(motor, coordination_yz, 4)
    position_in_mm = position_in_pulse * -5 / STEPS_PER_REVOLUTION
    return position_in_mm

def move_coordination_motor(coordination_x, coordination_y, coordination_z):
    coordination_x_pulse = convert_mm_x(motor_x, coordination_x)
    coordination_y_pulse = convert_mm_yz(motor_y, coordination_x)
    coordination_z_pulse = convert_mm_yz(motor_z, coordination_x)

    write_DB_number(motor_x, distance_Abs, coordination_x_pulse)
    write_DB_number(motor_y, distance_Abs, coordination_y_pulse)
    write_DB_number(motor_z, distance_Abs, coordination_z_pulse)
    
    writeBool_plc(motor_x, 0, move_Absolute, True)
    writeBool_plc(motor_y, 0, move_Absolute, True)
    writeBool_plc(motor_z, 0, move_Absolute, True)
    
    print("moved to (x,y,z):", coordination_x, coordination_y, coordination_z)
    
move_coordination_motor(motor_x, 5, 10, 15)

