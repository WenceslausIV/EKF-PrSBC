#!/usr/bin/env python
import matplotlib.pyplot as plt
import time

from __future__ import print_function
import rosnode
import tf_conversions
import threading

import roslib; roslib.load_manifest('teleop_twist_keyboard')
import rospy

from geometry_msgs.msg import Twist

import sys, select, termios, tty

import random
import math
import numpy as np
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse

import qpsolvers
from qpsolvers import solve_qp
from scipy import sparse as sparsed

import itertools
import numpy as np
from scipy.special import comb
from geometry_msgs.msg import TransformStamped, PoseStamped, Twist

rospy.init_node('teleop_twist_keyboard')
publisher = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)
rospy.sleep(2)
twist = Twist()
dt = 0.05
N = 1 # Number of robots
x = np.zeros((3, N))
#goal_points = np.array([[0., 0., 1., -1.], [-1., 1., 0., 0.], [math.pi / 2, -math.pi / 2, math.pi, 0.]])
dxu = np.zeros((2, N))
goal_points = np.array([[0.], [-1.], [math.pi / 2]])

p = np.zeros((3, N))

cov_list = []
cov_list2 = []
for i in range(N):
    cov = np.full((3, 3), 1.0, dtype=np.float64)
    cov_list.append(cov)
    cov_list2.append(cov)

prevtheta = np.zeros((1, N))
firstFlag = 1

def create_clf_unicycle_position_controller(linear_velocity_gain=0.8, angular_velocity_gain=3):

    assert isinstance(linear_velocity_gain, (int,
                                             float)), "In the function create_clf_unicycle_position_controller, the linear velocity gain (linear_velocity_gain) must be an integer or float. Recieved type %r." % type(
        linear_velocity_gain).__name__
    assert isinstance(angular_velocity_gain, (int,
                                              float)), "In the function create_clf_unicycle_position_controller, the angular velocity gain (angular_velocity_gain) must be an integer or float. Recieved type %r." % type(
        angular_velocity_gain).__name__

    # Check user input ranges/sizes
    assert linear_velocity_gain >= 0, "In the function create_clf_unicycle_position_controller, the linear velocity gain (linear_velocity_gain) must be greater than or equal to zero. Recieved %r." % linear_velocity_gain
    assert angular_velocity_gain >= 0, "In the function create_clf_unicycle_position_controller, the angular velocity gain (angular_velocity_gain) must be greater than or equal to zero. Recieved %r." % angular_velocity_gain

    def position_uni_clf_controller(states, positions):

        # Check user input types
        assert isinstance(states,
                          np.ndarray), "In the function created by the create_clf_unicycle_position_controller function, the single-integrator robot states (xi) must be a numpy array. Recieved type %r." % type(
            states).__name__
        assert isinstance(positions,
                          np.ndarray), "In the function created by the create_clf_unicycle_position_controller function, the robot goal points (positions) must be a numpy array. Recieved type %r." % type(
            positions).__name__

        # Check user input ranges/sizes
        assert states.shape[
                   0] == 3, "In the function created by the create_clf_unicycle_position_controller function, the dimension of the unicycle robot states (states) must be 3 ([x;y;theta]). Recieved dimension %r." % \
                            states.shape[0]
        assert positions.shape[
                   0] == 2, "In the function created by the create_clf_unicycle_position_controller function, the dimension of the robot goal positions (positions) must be 2 ([x_goal;y_goal]). Recieved dimension %r." % \
                            positions.shape[0]
        assert states.shape[1] == positions.shape[
            1], "In the function created by the create_clf_unicycle_position_controller function, the number of unicycle robot states (states) must be equal to the number of robot goal positions (positions). Recieved a current robot pose input array (states) of size %r states %r and desired position array (positions) of size %r states %r." % (
        states.shape[0], states.shape[1], positions.shape[0], positions.shape[1])

        _, N = np.shape(states)
        dxu = np.zeros((2, N))

        pos_error = positions - states[:2][:]
        rot_error = np.arctan2(pos_error[1][:], pos_error[0][:])
        dist = np.linalg.norm(pos_error, axis=0)

        dxu[0][:] = linear_velocity_gain * dist * np.cos(rot_error - states[2][:])
        dxu[1][:] = angular_velocity_gain * dist * np.sin(rot_error - states[2][:])

        return dxu

    return position_uni_clf_controller

unicycle_position_controller = create_clf_unicycle_position_controller

def transition_model(i, p, dt):

    x_dot = dxu[0,i] * dt * math.cos(p[2,i]) 
    y_dot = dxu[0,i] * dt * math.sin(p[2,i]) 
    
    p[0,i] = p[0,i] + x_dot 
    p[1,i] = p[1,i] + y_dot 
    p[2,i] = p[2,i] + dxu[1,i] * dt

    return p

def getG(i, p, dt):
    '''
    Parameters:
        x (Pose): The previous pose
        u (numpy array): The latest control input

    '''
    # Get displacement and orientation change

    # Create the matrix
    G = np.array([[1,0,-1],[0,1,-1],[0,0,1]], dtype=np.float32)
    G[0,2] = -dxu[0,i] * dt * math.sin(p[2,i]) 
    G[1,2] = dxu[0,i] * dt * math.cos(p[2,i])

def getV(i, p, dt):
    '''
    Parameters:
        x (Pose): The previous pose
        u (numpy.array): The latest control input vector
    '''

    # Create the matrix
    V = np.array([[-1,-1],[-1,-1],[0, 1]],dtype=np.float64)

    # -- new values
    V[0,0] = math.cos(p[2,i]) * dt
    V[0,1] = 0
    V[1,0] = math.sin(p[2,i]) * dt
    V[1,1] = 0
    V[2,0] = 0
    V[2,1] = dt
    return V

def updateCov(i, p, dt):
    global cov_list, m
    g = getG(i, p, dt)
    v = getV(i, p, dt)
    ahh = 0.05
    # set velocity noise covariance CHANGE LATER!!!
    m = np.array([[ahh**2 * dxu[0,i],0],[0,ahh**2*dxu[1,i]]])
    cov_list[i] = g @ cov_list[i] @ g.T + v @ m @ v.T
    return cov_list[i]

def update_cov2(i):
    global cov_list, cov_list2
    ahh = 0.05
    # set camera measurement noise covariance CHANGE LATER!!!
    Q = np.array([[ahh**2 ,0,0],[0,ahh**2,0],[0,0,ahh**2]])

    h = np.array([[1,0,0],[0,1,0],[0,0,1]])
    cov_list2[i] = h @ cov_list[i] @ h.T + Q
    return cov_list2[i]

def predict(i):
    global p, cov

    p = transition_model(i, p, 0.05)
    cov_list[i] = updateCov(i, p, 0.05)

def measure(i):
    global x, p, cov_list
    h = np.array([[1,0,0],[0,1,0],[0,0,1]])
    K = cov_list[i] @ h.T @ np.inv(update_cov2(i))
    p[0,i] = p[0,i] + K[0,0] @ (x[0,i] - p[0,i])
    p[1,i] = p[1,i] + K[1,1] @ (x[1,i] - p[1,i])
    p[2,i] = p[2,i] + K[2,2] @ (x[2,i] - p[2,i])
    eye = np.eye((3,3))
    cov_list[i] = (eye - K @ h) @ cov_list[i]
    

def callback(data, args):
    global firstFlag, x, p
    i = args

    #do i have to wrap the angle?
    theta = tf_conversions.transformations.euler_from_quaternion(
        [data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])[2]
    
    x[0, i] = data.pose.position.x
    x[1, i] = data.pose.position.y
    x[2, i] = theta

    if firstFlag == 1:
        p[0,i] = data.pose.position.x
        p[1,i] = data.pose.position.y
        p[2,i] = theta
        firstFlag = 0



def control_callback(event):

    for i in range(N):
        t = np.array([[0],[0],[0]])
        t[0, 0] = p[0,i]
        t[1, 0] = p[1,i]
        t[2, 0] = p[2,i]
        dxu = unicycle_position_controller(t, goal_points)
        #dxu = uni_barrier_cert(dxu, x)dxu = uni_barrier_cert(dxu, x)
        twist.linear.x = dxu[0,p]/5.
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = dxu[1,p]/5.
        publisher.publish(twist)


def ekf_update_function(event):
    predict(0)
    measure(0)

def gaussian_graph(event):
    mean = 0
    variance = cov_list[i][0][0]
    sigma = np.sqrt(variance)
    x = np.linspace(mean - 3*sigma, mean + 3*sigma, 100)
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp( - (x - mean)**2 / (2 * variance))
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label=f'Mean = {mean}, Variance = {variance}')
    plt.title('Gaussian Distribution')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()
    rospy.spin()

def central():
    rospy.Subscriber('/vrpn_client_node/Hus117' + '/pose', PoseStamped, callback, 0)

    timer = rospy.Timer(rospy.Duration(0.05), control_callback)
    ekf_timer = rospy.Timer(rospy.Duration(0.05), ekf_update_function)
    graph_timer = rospy.Timer(rospy.Duration(1), gaussian_graph)


if __name__ == '__main__':

    try:
        central()
    except rospy.ROSInterruptException:
        print(rospy.ROSInterruptException)
