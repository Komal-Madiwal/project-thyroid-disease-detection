import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" 
##The line of code you provided is used to generate a log file name based on the current date and time. 

# Get the absolute path of the current script and create 'logs' directory
log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "logs") ##_file__: This is a special variable in Python that represents the path to the current script. It is automatically defined by the Python interpreter.
## this line of code creates a path to a directory named "logs" that is located one level above the directory where the current script or module is located. The os.path.realpath is used to ensure that symbolic links are resolved and the absolute path is obtained.
os.makedirs(log_path, exist_ok=True)

# Create the full path for the log file
LOG_FILEPATH = os.path.join(log_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILEPATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

if __name__ == '__main__':
    logging.info("Testing log message")# Test logging at different levels


