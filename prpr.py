#!/usr/bin/env python3
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time
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
import scipy.stats as stats
from scipy.special import comb
import itertools
import numpy as np
from geometry_msgs.msg import TransformStamped, PoseStamped, Twist
from std_msgs.msg import Float64MultiArray

rospy.init_node('teleop_twist_keyboard')
publisher = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)
array_pub = rospy.Publisher('mean_std_array_0', Float64MultiArray, queue_size=1)
array_msg = Float64MultiArray()
array_msg.data = [0, 0]

rospy.sleep(2)
twist = Twist()
dt = 0.06
N = 2 # Number of robots
x = np.zeros((3, N))

nx = np.array([[0],[0],[0]])

#goal_points = np.array([[0., 0., 1., -1.], [-1., 1., 0., 0.], [math.pi / 2, -math.pi / 2, math.pi, 0.]])
dxu = np.zeros((2, N))
dxi = np.zeros((2, N))
goal_points = np.array([[0., 1.0], [0., 0.], [math.pi / 2, -math.pi/2]])


p = np.zeros((3, N))
URandSpan = 0. * np.ones((2, N))  # setting up velocity error range for each robot

x_rand_span_x = 0. * np.random.rand(1, N)  # setting up position error range for each robot,
x_rand_span_y = 0. * np.random.rand(1, N)
XRandSpan = np.concatenate((x_rand_span_x, x_rand_span_y))

Kp = 10
Vmax = 200
Wmax = np.pi
safety_radius = 0.2
barrier_gain = 1000
projection_distance = 0.05
magnitude_limit = 0.2
confidence_level = 0.95

cov_list = []
cov_list2 = []
for i in range(N):
    cov = np.full((3, 3), 0.0001, dtype=np.float64)
    cov_list.append(cov)
    cov_list2.append(cov)

#*****************************fix cov later for robot 1*****************
robotGaussianDistInfox = np.array([[p[0,0], x[0,1]],[math.sqrt(abs(cov_list[0][0,0])), math.sqrt(abs(cov_list[0][0,0])) ]]) #all the robots' x pos distribution info including mean and std
robotGaussianDistInfoy = np.array([[p[1,0], x[1,1]],[math.sqrt(abs(cov_list[0][1,1])), math.sqrt(abs(cov_list[0][1,1])) ]]) #all the robots' y pos distribution info including mean and std

firstFlag = 1

def create_clf_unicycle_pose_controller(approach_angle_gain=1, desired_angle_gain=2.7, rotation_error_gain=0.3):
	"""Returns a controller ($u: \mathbf{R}^{3 \times N} \times \mathbf{R}^{3 \times N} \to \mathbf{R}^{2 \times N}$)
    that will drive a unicycle-modeled agent to a pose (i.e., position & orientation). This control is based on a control
    Lyapunov function.

    approach_angle_gain - affects how the unicycle approaches the desired position
    desired_angle_gain - affects how the unicycle approaches the desired angle
    rotation_error_gain - affects how quickly the unicycle corrects rotation errors.


    -> function
    """

	gamma = approach_angle_gain
	k = desired_angle_gain
	h = rotation_error_gain

	def R(theta):
		return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

	def pose_uni_clf_controller(states, poses):
		N_states = states.shape[1]
		dxu = np.zeros((2, N_states))

		for i in range(N_states):
			translate = R(-poses[2, i]).dot((poses[:2, i] - states[:2, i]))
			e = np.linalg.norm(translate)
			theta = np.arctan2(translate[1], translate[0])
			alpha = theta - (states[2, i] - poses[2, i])
			alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

			ca = np.cos(alpha)
			sa = np.sin(alpha)
			dxu[0, i] = gamma * e * ca
#			print(alpha)
			dxu[1, i] = k * alpha + gamma * ((ca * sa) / alpha) * (alpha + h * theta)

		return dxu

	return pose_uni_clf_controller

def create_si_to_uni_mapping(projection_distance=0.05, angular_velocity_limit=np.pi):
    """Creates two functions for mapping from single integrator dynamics to
    unicycle dynamics and unicycle states to single integrator states.

    This mapping is done by placing a virtual control "point" in front of
    the unicycle.

    projection_distance: How far ahead to place the point
    angular_velocity_limit: The maximum angular velocity that can be provided

    -> (function, function)
    """

    # Check user input types
    assert isinstance(projection_distance, (int,
                                            float)), "In the function create_si_to_uni_mapping, the projection distance of the new control point (projection_distance) must be an integer or float. Recieved type %r." % type(
        projection_distance).__name__
    assert isinstance(angular_velocity_limit, (int,
                                               float)), "In the function create_si_to_uni_mapping, the maximum angular velocity command (angular_velocity_limit) must be an integer or float. Recieved type %r." % type(
        angular_velocity_limit).__name__

    # Check user input ranges/sizes
    assert projection_distance > 0, "In the function create_si_to_uni_mapping, the projection distance of the new control point (projection_distance) must be positive. Recieved %r." % projection_distance
    assert projection_distance >= 0, "In the function create_si_to_uni_mapping, the maximum angular velocity command (angular_velocity_limit) must be greater than or equal to zero. Recieved %r." % angular_velocity_limit

    def si_to_uni_dyn(dxi, poses):
        """Takes single-integrator velocities and transforms them to unicycle
        control inputs.

        dxi: 2xN numpy array of single-integrator control inputs
        poses: 3xN numpy array of unicycle poses

        -> 2xN numpy array of unicycle control inputs
        """

        # Check user input types
        assert isinstance(dxi,
                          np.ndarray), "In the si_to_uni_dyn function created by the create_si_to_uni_mapping function, the single integrator velocity inputs (dxi) must be a numpy array. Recieved type %r." % type(
            dxi).__name__
        assert isinstance(poses,
                          np.ndarray), "In the si_to_uni_dyn function created by the create_si_to_uni_mapping function, the current robot poses (poses) must be a numpy array. Recieved type %r." % type(
            poses).__name__

        # Check user input ranges/sizes
        assert dxi.shape[
                   0] == 2, "In the si_to_uni_dyn function created by the create_si_to_uni_mapping function, the dimension of the single integrator velocity inputs (dxi) must be 2 ([x_dot;y_dot]). Recieved dimension %r." % \
                            dxi.shape[0]
        assert poses.shape[
                   0] == 3, "In the si_to_uni_dyn function created by the create_si_to_uni_mapping function, the dimension of the current pose of each robot must be 3 ([x;y;theta]). Recieved dimension %r." % \
                            poses.shape[0]
        assert dxi.shape[1] == poses.shape[
            1], "In the si_to_uni_dyn function created by the create_si_to_uni_mapping function, the number of single integrator velocity inputs must be equal to the number of current robot poses. Recieved a single integrator velocity input array of size %r x %r and current pose array of size %r x %r." % (
        dxi.shape[0], dxi.shape[1], poses.shape[0], poses.shape[1])

        M, N = np.shape(dxi)

        cs = np.cos(poses[2, :])
        ss = np.sin(poses[2, :])

        dxu = np.zeros((2, N))
        dxu[0, :] = (cs * dxi[0, :] + ss * dxi[1, :])
        dxu[1, :] = (1 / projection_distance) * (-ss * dxi[0, :] + cs * dxi[1, :])

        # Impose angular velocity cap.
        dxu[1, dxu[1, :] > angular_velocity_limit] = angular_velocity_limit
        dxu[1, dxu[1, :] < -angular_velocity_limit] = -angular_velocity_limit

        return dxu

    def uni_to_si_states(poses):
        """Takes unicycle states and returns single-integrator states

        poses: 3xN numpy array of unicycle states

        -> 2xN numpy array of single-integrator states
        """

        _, N = np.shape(poses)

        si_states = np.zeros((2, N))
        si_states[0, :] = poses[0, :] + projection_distance * np.cos(poses[2, :])
        si_states[1, :] = poses[1, :] + projection_distance * np.sin(poses[2, :])

        return si_states

    return si_to_uni_dyn, uni_to_si_states


def create_uni_to_si_dynamics(projection_distance=0.05):
    """Creates two functions for mapping from unicycle dynamics to single
    integrator dynamics and single integrator states to unicycle states.

    This mapping is done by placing a virtual control "point" in front of
    the unicycle.

    projection_distance: How far ahead to place the point

    -> function
    """

    # Check user input types
    assert isinstance(projection_distance, (int,
                                            float)), "In the function create_uni_to_si_dynamics, the projection distance of the new control point (projection_distance) must be an integer or float. Recieved type %r." % type(
        projection_distance).__name__

    # Check user input ranges/sizes
    assert projection_distance > 0, "In the function create_uni_to_si_dynamics, the projection distance of the new control point (projection_distance) must be positive. Recieved %r." % projection_distance

    def uni_to_si_dyn(dxu, poses):
        """A function for converting from unicycle to single-integrator dynamics.
        Utilizes a virtual point placed in front of the unicycle.

        dxu: 2xN numpy array of unicycle control inputs
        poses: 3xN numpy array of unicycle poses
        projection_distance: How far ahead of the unicycle model to place the point

        -> 2xN numpy array of single-integrator control inputs
        """

        # Check user input types
        assert isinstance(dxu,
                          np.ndarray), "In the uni_to_si_dyn function created by the create_uni_to_si_dynamics function, the unicycle velocity inputs (dxu) must be a numpy array. Recieved type %r." % type(
            dxi).__name__
        assert isinstance(poses,
                          np.ndarray), "In the uni_to_si_dyn function created by the create_uni_to_si_dynamics function, the current robot poses (poses) must be a numpy array. Recieved type %r." % type(
            poses).__name__

        # Check user input ranges/sizes
        assert dxu.shape[
                   0] == 2, "In the uni_to_si_dyn function created by the create_uni_to_si_dynamics function, the dimension of the unicycle velocity inputs (dxu) must be 2 ([v;w]). Recieved dimension %r." % \
                            dxu.shape[0]
        assert poses.shape[
                   0] == 3, "In the uni_to_si_dyn function created by the create_uni_to_si_dynamics function, the dimension of the current pose of each robot must be 3 ([x;y;theta]). Recieved dimension %r." % \
                            poses.shape[0]
        assert dxu.shape[1] == poses.shape[
            1], "In the uni_to_si_dyn function created by the create_uni_to_si_dynamics function, the number of unicycle velocity inputs must be equal to the number of current robot poses. Recieved a unicycle velocity input array of size %r x %r and current pose array of size %r x %r." % (
        dxu.shape[0], dxu.shape[1], poses.shape[0], poses.shape[1])

        M, N = np.shape(dxu)

        cs = np.cos(poses[2, :])
        ss = np.sin(poses[2, :])

        dxi = np.zeros((2, N))
        dxi[0, :] = (cs * dxu[0, :] - projection_distance * ss * dxu[1, :])
        dxi[1, :] = (ss * dxu[0, :] + projection_distance * cs * dxu[1, :])

        return dxi

    return uni_to_si_dyn

def find_b(a, c, delta):
    if a > 0.2:
        a == 0.2
    elif c > 0.2:
        c == 0.2
    
  
    sigma = 1
    # returns list of b2, b1, sigma
    b2 = delta
    b1 = delta

    # a and c should be positive

    if a > c:  # [-A, A] is the large one, and[-C, C] is the smaller one
        A = a
        C = c
    else:
        A = c
        C = a

    if A == 0 and C == 0:
        return b2, b1, sigma

    # O_vec = [-(A + C), -(A - C), (A - C), (A + C)] # vector of vertices on the trap distribution cdf

    h = 1 / (2 * A)  # height of the trap distribution
    area_seq = [1 / 2 * 2 * C * h, 2 * (A - C) * h, 1 / 2 * 2 * C * h]
    area_vec = [area_seq[0], sum(area_seq[:2])]

    if abs(A - C) < 1e-5:  # then is triangle
        # assuming sigma > 50
        b1 = (A + C) - 2 * C * np.sqrt((1 - sigma) / (1 - area_vec[1]))  # 1 - area_vec[1] should be very close to 0.5
        b2 = -b1

        b1 = b1 + delta
        b2 = b2 + delta  # apply shift here due to xi - xj

    else:  # than is trap
        b1 = (A + C) - 2 * C * np.sqrt((1 - sigma) / (1 - area_vec[1]))
        b2 = -(A + C) + 2 * C * np.sqrt((1 - sigma) / (1 - area_vec[1]))
        b1 = b1 + delta
        b2 = b2 + delta  # apply shift here due to xi - xj
    print(b1)
    print('b1 above')
    return b2, b1, sigma

def create_si_pr_barrier_certificate_centralized(gamma=100, safety_radius=0.2, magnitude_limit=0.2, confidence_level=0.95):

    def barrier_certificate(dxi, x):
        global URandSpan, XRandSpan, robotGaussianDistInfox, robotGaussianDistInfoy
    
        num_constraints = int(comb(N, 2))
        A = np.zeros((num_constraints, 2 * N))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2 * np.identity(2 * N)))

        count = 0
        if len(XRandSpan) == 1:
            XRandSpan = np.zeros(2, N)
        if len(URandSpan) == 1:
            URandSpan = np.zeros(2, N)
        for i in range(N - 1):
            for j in range(i + 1, N):

                max_dvij_x = np.linalg.norm(URandSpan[0, i] + URandSpan[0, j])
                max_dvij_y = np.linalg.norm(URandSpan[1, i] + URandSpan[1, j])
                max_dxij_x = np.linalg.norm(x[0, i] - x[0, j]) + np.linalg.norm(XRandSpan[0, i] + XRandSpan[0, j])
                max_dxij_y = np.linalg.norm(x[1, i] - x[1, j]) + np.linalg.norm(XRandSpan[1, i] + XRandSpan[1, j])
                BB_x = -safety_radius ** 2 - 2 / gamma * max_dvij_x * max_dxij_x
                BB_y = -safety_radius ** 2 - 2 / gamma * max_dvij_y * max_dxij_y
                #siwon added
                z_value = stats.norm.ppf(1 - (1 - confidence_level) / 2) #if confidence level is 95% then z_value is about 1.96
                print(z_value)
                XRandSpan[0, i] = (z_value * robotGaussianDistInfox[1,i])
                XRandSpan[0, j] = (z_value * robotGaussianDistInfox[1,j])
                #print(robotGaussianDistInfox[0,i]) 
                XRandSpan[1, i] = (z_value * robotGaussianDistInfoy[1,i])
                XRandSpan[1, j] = (z_value * robotGaussianDistInfoy[1,j])
                #print(XRandSpan[0,i])
                b2_x, b1_x, sigma = find_b(XRandSpan[0, i], XRandSpan[0, j], x[0, i] - x[0, j])
                b2_y, b1_y, sigma = find_b(XRandSpan[1, i], XRandSpan[1, j], x[1, i] - x[1, j])

                if (b2_x < 0 and b1_x > 0) or (b2_x > 0 and b1_x < 0):
                    # print('WARNING: distance between robots on x smaller than error bound!')
                    b_x = 0
                elif (b1_x < 0) and (b2_x < b1_x) or (b2_x < 0 and b2_x > b1_x):
                    b_x = b1_x
                elif (b2_x > 0 and b2_x < b1_x) or (b1_x > 0 and b2_x > b1_x):
                    b_x = b2_x
                else:
                    b_x = b1_x
                    # print('WARNING: no uncertainty or sigma = 0.5 on x')  # b1 = b2 or no uncertainty

                if (b2_y < 0 and b1_y > 0) or (b2_y > 0 and b1_y < 0):
                    # print('WARNING: distance between robots on y smaller than error bound!')
                    b_y = 0
                elif (b1_y < 0 and b2_y < b1_y) or (b2_y < 0 and b2_y > b1_y):
                    b_y = b1_y
                elif (b2_y > 0 and b2_y < b1_y) or (b1_y > 0 and b2_y > b1_y):
                    b_y = b2_y
                else:
                    b_y = b1_y

                A[count, (2 * i)] = -2 * b_x  # matlab original: A(count, (2*i-1):(2*i)) = -2*([b_x;b_y]);
                A[count, (2 * i + 1)] = -2 * b_y

                A[count, (2 * j)] = 2 * b_x  # matlab original: A(count, (2*j-1):(2*j)) =  2*([b_x;b_y])';
                A[count, (2 * j + 1)] = 2 * b_y

                t1 = np.array([[b_x], [0.0]])
                t2 = np.array([[max_dvij_x], [0]])
                t3 = np.array([[max_dxij_x], [0]])
                t4 = np.array([[0], [b_y]])
                t5 = np.array([[0], [max_dvij_y]])
                t6 = np.array([[0], [max_dxij_y]])


                h1 = np.linalg.norm(t1) ** 2 - safety_radius ** 2 - 2 * np.linalg.norm(
                    t2) * np.linalg.norm(t3) / gamma
                h2 = np.linalg.norm(t4) ** 2 - safety_radius ** 2 - 2 * np.linalg.norm(
                    t5) * np.linalg.norm(t6) / gamma  # h_y

                h = h1 + h2

                b[count] = gamma * h ** 3  # matlab original: b(count) = gamma*h^3
                count += 1

        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, 0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] = dxi[:, idxs_to_normalize] * (magnitude_limit / norms[idxs_to_normalize])

        f_mat = -2 * np.reshape(dxi, 2 * N, order='F')
        f_mat = f_mat.astype('float')
        result = qp(H, matrix(f_mat), matrix(A), matrix(b))['x']

        return np.reshape(result, (2, -1), order='F')

    return barrier_certificate


def create_pr_unicycle_barrier_certificate_cent(barrier_gain=100, safety_radius=0.12, projection_distance=0.05,
                                                magnitude_limit=0.2, confidence_level=0.95):

    # Check user input types
    assert isinstance(barrier_gain, (int,
                                     float)), "In the function create_pr_unicycle_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(
        barrier_gain).__name__
    assert isinstance(safety_radius, (int,
                                      float)), "In the function create_pr_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(
        safety_radius).__name__
    assert isinstance(projection_distance, (int,
                                            float)), "In the function create_pr_unicycle_barrier_certificate, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be an integer or float. Recieved type %r." % type(
        projection_distance).__name__
    assert isinstance(magnitude_limit, (int,
                                        float)), "In the function create_pr_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(
        magnitude_limit).__name__
    assert isinstance(confidence_level,
                      float), "In the function create_pr_unicycle_barrier_certificate, the confidence level must be a float. Recieved type %r." % type(
        confidence_level).__name__

    # Check user input ranges/sizes

    assert barrier_gain > 0, "In the function create_pr_unicycle_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert safety_radius >= 0.12, "In the function create_pr_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m). Recieved %r." % safety_radius
    assert projection_distance > 0, "In the function create_pr_unicycle_barrier_certificate, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be positive. Recieved %r." % projection_distance
    assert magnitude_limit > 0, "In the function create_pr_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_pr_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit
    assert confidence_level <= 1, "In the function create_pr_unicycle_barrier_certificate, the confidence level must be less than 1. Recieved %r." % confidence_level
    assert confidence_level >= 0, "In the function create_pr_unicycle_barrier_certificate, the confidence level must be positive (greater than 0). Recieved %r." % confidence_level

    si_barrier_cert = create_si_pr_barrier_certificate_centralized(gamma=barrier_gain,
                                                                   safety_radius=safety_radius + projection_distance,
                                                                   confidence_level=confidence_level)

    si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping(projection_distance=projection_distance)

    uni_to_si_dyn = create_uni_to_si_dynamics(projection_distance=projection_distance)

    def f(dxu, x):
        global XRandSpan, URandSpan



        # Check user input types
        assert isinstance(dxu,
                          np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the unicycle robot velocity command (dxu) must be a numpy array. Recieved type %r." % type(
            dxu).__name__
        assert isinstance(x,
                          np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(
            x).__name__

        # Check user input ranges/sizes
        assert x.shape[
                   0] == 3, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the unicycle robot states (x) must be 3 ([x;y;theta]). Recieved dimension %r." % \
                            x.shape[0]
        assert dxu.shape[
                   0] == 2, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the robot unicycle velocity command (dxu) must be 2 ([v;w]). Recieved dimension %r." % \
                            dxu.shape[0]
        assert x.shape[1] == dxu.shape[
            1], "In the function created by the create_unicycle_barrier_certificate function, the number of robot states (x) must be equal to the number of robot unicycle velocity commands (dxu). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (
        x.shape[0], x.shape[1], dxu.shape[0], dxu.shape[1])

        x_si = uni_to_si_states(x)
        # Convert unicycle control command to single integrator one
        dxi = uni_to_si_dyn(dxu, x)
        # Apply single integrator barrier certificate
        dxi = si_barrier_cert(dxi, x_si)
        # Return safe unicycle command
        return si_to_uni_dyn(dxi, x)

    return f



def findDistanceBetweenAngles(a, b):
    '''
    Get the smallest orientation difference in range [-pi,pi] between two angles 
    Parameters:
        a (double): an angle in radians
        b (double): an angle in radians
    
    Returns:
        double: smallest orientation difference in range [-pi,pi]
    '''
    result = 0
    difference = b - a
    
    if difference > math.pi:
      difference = math.fmod(difference, math.pi)
      result = difference - math.pi
 
    elif(difference < -math.pi):
      difference = math.fmod(difference, math.pi)
      result = difference + math.pi
 
    else:
      result = difference
 
    return result
 
 
 
def displaceAngle(a1, a2):
    '''
    Displace an orientation by an angle and stay within [-pi,pi] range
    Parameters:
        a1 (double): an angle
        a2 (double): an angle
    
    Returns:
        double: The resulting angle in range [-pi,pi] after displacing a1 by a2
    '''
    a2 = a2 % (2.0*math.pi)
 
    if a2 > math.pi:
        a2 = (a2 % math.pi) - math.pi
 
    return findDistanceBetweenAngles(-a1,a2)


def transition_model(i, p, dt):
    x_dot = (dxu[0,i]) * dt * math.cos(p[2,i]) 
    y_dot = (dxu[0,i]) * dt * math.sin(p[2,i]) 
    
    p[0,i] = p[0,i] + x_dot 
    p[1,i] = p[1,i] + y_dot
    t = displaceAngle(p[2,i], (dxu[1,i]*dt))
    p[2,i] = t

    return p

def getG(i,p, dt):
    '''
    Parameters:
        x (Pose): The previous pose
        u (numpy array): The latest control input

    '''
    # Get displacement and orientation change

    # Create the matrix
    G = np.array([[1,0,-1],[0,1,-1],[0,0,1]], dtype=np.float32)
    G[0,2] = -(dxu[0,i]) * dt * math.sin(p[2,i]) 
    G[1,2] = (dxu[0,i]) * dt * math.cos(p[2,i])
    return G

def getV(i,p, dt):
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

def updateCov(i,p, dt):
    global cov_list, m
    g = getG(i,p, dt)
    v = getV(i,p, dt)
    ahh = 10
    # set velocity noise covariance CHANGE LATER!!!
    m = np.array([[ahh * (dxu[0,i]),0],[0,ahh*(dxu[1,i])]])
    cov_list[i] = g @ cov_list[i] @ g.T + v @ m @ v.T
    return cov_list[i]

def update_cov2(i):
    global cov_list, cov_list2
    ahh = 0.02
    # set camera measurement noise covariance CHANGE LATER!!!
    Q = np.array([[0.001 ,0,0],[0,0.001,0],[0,0,ahh**2]])

    h = np.array([[1,0,0],[0,1,0],[0,0,1]])
    cov_list2[i] = h @ cov_list[i] @ h.T + Q
    return cov_list2[i]

def predict(i):
    global p, cov, dt

    p = transition_model(i,p, dt)
    cov_list[i] = updateCov(i,p, dt)

def measure(i):
    global x, cov_list, p
    h = np.array([[1,0,0],[0,1,0],[0,0,1]])
    K = cov_list[i] @ h.T @ np.linalg.inv(update_cov2(i))
    #maybe fix?
    
    k = np.array([ [p[0,i]],[p[1,i]],[p[2,i]] ])
    q = np.array([ [nx[0,i]],[nx[1,i]],[nx[2,i]] ])

    t = k + K @ (q - k)
    p[0,i] = t[0,0]
    p[1,i] = t[1,0]
    p[2,i] = t[2,0]
    eye = np.eye(3, dtype=float)
    cov_list[i] = (eye - K @ h) @ cov_list[i]
    

uni_controller = create_clf_unicycle_pose_controller()
uni_barrier_cert = create_pr_unicycle_barrier_certificate_cent(safety_radius=safety_radius, confidence_level=confidence_level)

def callback(data, args):
    global firstFlag, x, p, nx
    i = args

    #do i have to wrap the angle?
    theta = tf_conversions.transformations.euler_from_quaternion(
        [data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])[2]
    
    x[0,i] = data.pose.position.x
    x[1,i] = data.pose.position.y
    x[2,i] = theta
    l = np.array([[0],[0],[0]])
    l[0,0] = x[0,i]
    l[1,0] = x[1,i]
    l[2,0] = x[2,i]
    nx = x +  np.random.normal(0, 0.1,l.shape)
    if args == 1:
        p[0,1] = data.pose.position.x
        p[1,1] = data.pose.position.y
        p[2,1] = theta



    if firstFlag == 1:
        p[0,i] = nx[0,0]
        p[1,i] = nx[1,0]
        p[2,i] = .0
        firstFlag = 0



def control_callback(event):
    global p, dxu
    dxu = uni_controller(p, goal_points)
    dxu = uni_barrier_cert(dxu, p)
    print(dxu)
    twist.linear.x = dxu[0,0]
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0
    twist.angular.y = 0
    twist.angular.z = dxu[1,0]
    publisher.publish(twist)
    #predict(0)
    #measure(0)
    #print(p)

def ekf_update_function(event):
    global p,robotGaussianDistInfox, robotGaussianDistInfoy
   
    predict(0)
    measure(0)
    robotGaussianDistInfox[0,0] = p[0,0]
    #print(cov_list[0][0,0])
    robotGaussianDistInfox[1,0] = math.sqrt(abs(cov_list[0][0,0]))
    robotGaussianDistInfoy[0,0] = p[1,0]
    robotGaussianDistInfoy[1,0] = math.sqrt(abs(cov_list[0][1,1]))
    #**********change dist info for robot 1 later*********
    robotGaussianDistInfox[0,1] = x[0,1]
    robotGaussianDistInfox[1,1] = math.sqrt(abs(cov_list[0][0,0]))
    robotGaussianDistInfoy[0,1] = x[1,1]
    robotGaussianDistInfoy[1,1] = math.sqrt(abs(cov_list[0][1,1]))
    array_msg.data = [robotGaussianDistInfox[0,0], robotGaussianDistInfox[1,0], robotGaussianDistInfoy[0,0],robotGaussianDistInfoy[1,0]]  
    array_pub.publish(array_msg)

def central():
    rospy.Subscriber('/vrpn_client_node/Hus188' + '/pose', PoseStamped, callback, 0)
    rospy.Subscriber('/vrpn_client_node/Hus138' + '/pose', PoseStamped, callback, 1)

    timer = rospy.Timer(rospy.Duration(0.05), control_callback)
    ekf_timer = rospy.Timer(rospy.Duration(0.06), ekf_update_function)
    #pos_compare_timer = rospy.Timer(rospy.Duration(50), pos_compare)
    #pos_compare()
    rospy.spin()

if __name__ == '__main__':

    try:
        central()
    except rospy.ROSInterruptException:
        print(rospy.ROSInterruptException)
