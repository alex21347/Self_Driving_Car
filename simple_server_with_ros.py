#!/usr/bin/python3

import logging
import numpy as np
from numpysocket import NumpySocket

import rospy
from geometry_msgs.msg import Twist
import time
import numpy as np
#logger = logging.getLogger('simple server')
#logger.setLevel(logging.INFO)

npSocket = NumpySocket()
#publisher = rospy.Publisher('robot_1/cmd_vel', Twist)

#logger.info("starting server, waiting for client")
npSocket.startServer(1027)
time.sleep(1)
#while True:
npSocket.send(np.zeros((2,2)))
print('sent')
time.sleep(1)
#exit()
#msg = Twist()
#msg.linear.x = 0.15
#msg.angular.z = frame[0]
#publisher.send(msg)
    
try:
    npSocket.close()
except OSError as err:
    logging.error("server already disconnected")
