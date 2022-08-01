import numpy as np
import socket
import rospy
from geometry_msgs.msg import Twist
import time

publisher = rospy.Publisher("/robot_1/cmd_vel", Twist, queue_size=1)
last_omega = 0

def timer_callback(msg):
    global publisher
    global last_omega
    velocity_message = Twist()
    velocity_message.linear.x = 0.15
    velocity_message.angular.z = last_omega
    publisher.publish(velocity_message)

if __name__ == "__main__":
    rospy.init_node("velocity_udp_server")
    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    localIP     = "192.168.1.4"
    localPort   = 20000
    bufferSize  = 512
    UDPServerSocket.bind((localIP, localPort))
    #UDPServerSocket.listen()
    #conn, addr = UDPServerSocket.accept()
    time.sleep(1)
    timer = rospy.Timer(rospy.Duration(0.1), timer_callback)
    while(True):
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        #print(message)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]

        numpymsg = np.frombuffer(message, count = 1, dtype = np.float64)
        

        # Sending a reply to client
        omega = numpymsg[0] #?
        last_omega = omega
