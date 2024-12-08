import tkinter as tk
from tkinter import messagebox
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
    plc.db_write(db_number, start_address, bytearray(struct.pack('>f', value)))

def writeBool_plc(db_number, start_offset, bit_offset, value):
    reading = plc.db_read(db_number, start_offset, 1)
    snap7.util.set_bool(reading, 0, bit_offset, value)
    plc.db_write(db_number, start_offset, reading)

def readBool_plc(db_number, start_offset, bit_offset):
    reading = plc.db_read(db_number, start_offset, 1)
    return snap7.util.get_bool(reading, 0, bit_offset)

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

def convert_mm_to_pulse(coordination):
    return coordination / 5 * STEPS_PER_REVOLUTION

def move_coordination_motor(coordination_x, coordination_y, coordination_z):
    coordination_x_pulse = convert_mm_to_pulse(coordination_x)
    coordination_y_pulse = convert_mm_to_pulse(coordination_y)
    coordination_z_pulse = convert_mm_to_pulse(coordination_z)

    write_DB_number(motor_x, distance_Abs, coordination_x_pulse)
    write_DB_number(motor_y, distance_Abs, coordination_y_pulse)
    write_DB_number(motor_z, distance_Abs, coordination_z_pulse)
    
    writeBool_plc(motor_x, 0, move_Absolute, True)
    writeBool_plc(motor_y, 0, move_Absolute, True)
    writeBool_plc(motor_z, 0, move_Absolute, True)

def get_current_position():
    x = read_DB_number(motor_x, real_position, 4)
    y = read_DB_number(motor_y, real_position, 4)
    z = read_DB_number(motor_z, real_position, 4)
    return x, y, z

# Tkinter GUI
def run_motor():
    try:
        x = float(entry_x.get())
        y = float(entry_y.get())
        z = float(entry_z.get())
        move_coordination_motor(x, y, z)
        messagebox.showinfo("Success", f"Moved to position: X={x}, Y={y}, Z={z}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers for X, Y, Z.")

def update_position():
    try:
        x, y, z = get_current_position()
        label_position.config(text=f"Current Position: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
    except Exception as e:
        label_position.config(text="Error reading position")

# Create the main window
root = tk.Tk()
root.title("Motor Control Interface")

# Input fields
tk.Label(root, text="X Coordinate:").grid(row=0, column=0, padx=5, pady=5)
entry_x = tk.Entry(root)
entry_x.grid(row=0, column=1, padx=5, pady=5)

tk.Label(root, text="Y Coordinate:").grid(row=1, column=0, padx=5, pady=5)
entry_y = tk.Entry(root)
entry_y.grid(row=1, column=1, padx=5, pady=5)

tk.Label(root, text="Z Coordinate:").grid(row=2, column=0, padx=5, pady=5)
entry_z = tk.Entry(root)
entry_z.grid(row=2, column=1, padx=5, pady=5)

# Buttons
tk.Button(root, text="Run", command=run_motor).grid(row=3, column=0, columnspan=2, pady=10)

# Current position display
label_position = tk.Label(root, text="Current Position: X=0.00, Y=0.00, Z=0.00")
label_position.grid(row=4, column=0, columnspan=2, pady=5)

# Update position every 1 second
def periodic_update():
    update_position()
    root.after(1000, periodic_update)

periodic_update()

# Start the main loop
root.mainloop()

#tọa độ x, y, z tốt: 42.3, 54.1, 22.05