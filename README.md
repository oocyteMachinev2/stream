# stream : Oocyte maturation SOFTWARE project  🚀🚀🚀

Structure of project:

stream/
│
├── vision_system/		#space for develop algorithm related to camera
│   ├── ______init______.py                  	# use to call this folder from another
│   ├── models/
│   ├── training_data/
│   ├── evaluation/
│   └── vision_module.py	# main vision module
│
├── plc_communication/	# use for talking and control PLC, receive data from PLC
│   ├── __init__.py                   	# Marks this directory as a package
│   ├── plc_control.py
│   ├── plc_utils.py
│   └── logs/
│
├── uart_communication/
│   ├── __init__.py                   # Marks this directory as a package
│   ├── uart_interface.c
│   ├── uart_interface.h
│   └── sensor_data/
│
├── web_interface/
│   ├── __init__.py                   # Marks this directory as a package
│   ├── static/
│   ├── templates/
│   ├── js/
│   └── server.py
│
├── shared_file/
│   ├── __init__.py                   # Marks this directory as a package
│   ├── config.py
│   ├── utils.py
│   └── logger.py
│
├── lab_space/
│   ├── __init__.py                   # Marks this directory as a package
│   ├── test_ai.py
│   ├── test_plc.py
│   ├── test_uart.c
│   └── test_web.py
│
├── data_user/
│   ├── person1/
│   │	├── timeslapse_data/	# location to save timelapse image (.jpg combine with date and time collect image)

│   │	└── inputing data/ 	# information of this person when typing initial

│   │

│   ├── person1/
│   		├── timeslapse_data/	# location to save timelapse image (.jpg combine with date and time collect image)

│		└── inputing data/ 	# information of this person when typing initial

├── environment_logging/
│   ├── dd_mm_yy1/

│               ├──temp.csv

│               └──gas.csv
│   └── dd_mm_yy2/
│
└── launch.py
