#!/usr/bin/env python3
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import time
import rosnode
import tf_conversions
import threading
import roslib;

roslib.load_manifest('teleop_twist_keyboard')
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
from matplotlib.patches import Circle
from khepera_communicator.msg import K4_controls, SensorReadings

rospy.init_node('teleop_twist_keyboard')
# publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

publisher131 = rospy.Publisher('/cmd_vel_131', Twist, queue_size=1)
publisher139 = rospy.Publisher('/cmd_vel_139', Twist, queue_size=1)

rospy.sleep(2)

twist131 = Twist()
twist139 = Twist()
dt = 0.05
N = 2  # Number of robots
x = np.zeros((3, N))
p_0 = np.zeros((3, 1))
p_1 = np.zeros((3, 1))
nx = np.zeros((3, N))

# goal_points = np.array([[0., 0., 1., -1.], [-1., 1., 0., 0.], [math.pi / 2, -math.pi / 2, math.pi, 0.]])
dxu = np.zeros((2, N))
dxi = np.zeros((2, N))
# change below line according to your N and its goals
goal_points = np.array([[0., 0.], [-2.0, 2.0], [math.pi / 2, -math.pi / 2]])

URandSpan = 0.01 * np.ones((2, N))  # setting up velocity error range for each robot

x_rand_span_x = 0. * np.random.rand(1, N)  # setting up position error range for each robot,
x_rand_span_y = 0. * np.random.rand(1, N)
XRandSpan = np.concatenate((x_rand_span_x, x_rand_span_y))

Kp = 10
Vmax = 200
Wmax = np.pi
safety_radius = 0.3
barrier_gain = 1000
projection_distance = 0.05
magnitude_limit = 0.2
confidence_level = 0.9


cov_list = []
cov_list2 = []
firstFlag = []
for i in range(N):
    cov = np.full((3, 3), 0.0001, dtype=np.float64)
    cov_list.append(cov)
    cov_list2.append(cov)
    firstFlag.append(1)


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


def create_si_pr_barrier_certificate_centralized(gamma=100, safety_radius=0.3, magnitude_limit=0.2,
                                                 confidence_level=0.9):
    def barrier_certificate(dxi, xp):
        global URandSpan, XRandSpan

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
                max_dxij_x = np.linalg.norm(xp[0, i] - xp[0, j]) + np.linalg.norm(XRandSpan[0, i] + XRandSpan[0, j])
                max_dxij_y = np.linalg.norm(xp[1, i] - xp[1, j]) + np.linalg.norm(XRandSpan[1, i] + XRandSpan[1, j])

                # siwon added
                z_value = stats.norm.ppf(confidence_level)

                twobytwo_cov = np.array(
                    [[cov_list[i][0][0], cov_list[i][0][1]], [cov_list[i][1][0], cov_list[i][1][1]]])

                eigenvalues, eigenvectors = np.linalg.eig(twobytwo_cov)
                #print(twobytwo_cov)
                major_axis_length_i = 2 * np.sqrt(np.abs(np.max(eigenvalues)))

                # major_axis_length_i = 0

                circle_i_r = major_axis_length_i / 2
                circle_i_std = circle_i_r


                # currently using i's cov for robot j
                twobytwo_cov = np.array(
                    [[cov_list[j][0][0], cov_list[j][0][1]], [cov_list[j][1][0], cov_list[j][1][1]]])
                eigenvalues, eigenvectors = np.linalg.eig(twobytwo_cov)
                major_axis_length_j = 2 * np.sqrt(np.abs(np.max(eigenvalues)))

                circle_j_r = major_axis_length_j / 2
                circle_j_std = circle_j_r
                # UNCOMMENT BELOW LINE IF THE RUNNING AWAY BEHAVIOR IS OBSERVED
                # circle_j_std = 0

                new_gaus_std = np.sqrt((circle_i_std ** 2) + (circle_j_std ** 2))
                b1_x = new_gaus_std * z_value

                #print(b1_x)

                b1_y = b1_x
                b2_y = -b1_y

                b2_x = -b1_x
                b1_x = b1_x + (xp[0, i] - xp[0, j])
                b2_x = b2_x + (xp[0, i] - xp[0, j])

                b1_y = b1_y + (xp[1, i] - xp[1, j])
                b2_y = b2_y + (xp[1, i] - xp[1, j])
                '''
                b1_x = 0.03
                b2_x = -0.03
                b1_y = 0.03
                b2_y = -0.03
                '''
                '''
                b2_x, b1_x = find_b(XRandSpan[0, i], XRandSpan[0, j], x[0, i] - x[0, j])
                b2_y, b1_y = find_b(XRandSpan[1, i], XRandSpan[1, j], x[1, i] - x[1, j])
                '''

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

                # max_dxij_x = np.linalg.norm(x[0, i] - x[0, j]) + np.linalg.norm(XRandSpan[0, i] + XRandSpan[0, j])
                # max_dxij_y = np.linalg.norm(x[1, i] - x[1, j]) + np.linalg.norm(XRandSpan[1, i] + XRandSpan[1, j])

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


def create_pr_unicycle_barrier_certificate_cent(barrier_gain=100, safety_radius=0.3, projection_distance=0.05,
                                                magnitude_limit=0.2, confidence_level=0.9):
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

    elif (difference < -math.pi):
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
    a2 = a2 % (2.0 * math.pi)

    if a2 > math.pi:
        a2 = (a2 % math.pi) - math.pi

    return findDistanceBetweenAngles(-a1, a2)


def transition_model(i, p_0, dt):
    x_dot = (dxu[0, i]) * dt * math.cos(p_0[2, 0])
    y_dot = (dxu[0, i]) * dt * math.sin(p_0[2, 0])

    p_0[0, 0] = p_0[0, 0] + x_dot
    p_0[1, 0] = p_0[1, 0] + y_dot
    t = displaceAngle(p_0[2, 0], (dxu[1, i] * dt))
    p_0[2, 0] = t

    return p_0


def getG(i, p_0, dt):
    '''
    Parameters:
        x (Pose): The previous pose
        u (numpy array): The latest control input

    '''
    G = np.array([[1, 0, -1], [0, 1, -1], [0, 0, 1]], dtype=np.float32)
    G[0, 2] = -(dxu[0, i]) * dt * math.sin(p_0[2, 0])
    G[1, 2] = (dxu[0, i]) * dt * math.cos(p_0[2, 0])
    return G


def getV(i, p_0, dt):
    '''
    Parameters:
        x (Pose): The previous pose
        u (numpy.array): The latest control input vector
    '''

    # Create the matrix
    V = np.array([[-1, -1], [-1, -1], [0, 1]], dtype=np.float64)

    # -- new values
    V[0, 0] = math.cos(p_0[2, 0]) * dt
    V[0, 1] = 0
    V[1, 0] = math.sin(p_0[2, 0]) * dt
    V[1, 1] = 0
    V[2, 0] = 0
    V[2, 1] = dt
    return V


def updateCov(i, p_0, dt):
    global cov_list
    g = getG(i, p_0, dt)
    v = getV(i, p_0, dt)
    ahh = 0.05
    # set velocity noise covariance CHANGE LATER!!!
    m = np.array([[0.005 * (dxu[0, i]), 0], [0, 0.005 * (dxu[1, i])]])
    cov_list[i] = g @ cov_list[i] @ g.T + v @ m @ v.T
    return cov_list[i]


def update_cov2(i):
    global cov_list, cov_list2
    ahh = 0.01
    # **********************************************************
    # MAYBE CHANGE TO 0.1, 0.1, 0.001
    Q = np.array([[0.05, 0, 0], [0, 0.05, 0], [0.0005, 0, 0]])

    h = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cov_list2[i] = h @ cov_list[i] @ h.T + Q
    return cov_list2[i]


def predict(i):
    global p_0, cov_list, dt

    p_0 = transition_model(i, p_0, dt)
    cov_list[i] = updateCov(i, p_0, dt)


def measure(i):
    global x, cov_list, p_0
    h = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    K = cov_list[i] @ h.T @ np.linalg.inv(update_cov2(i))
    # maybe fix?

    k = np.array([[p_0[0, 0]], [p_0[1, 0]], [p_0[2, 0]]])
    q = np.array([[nx[0, i]], [nx[1, i]], [nx[2, i]]])

    t = k + K @ (q - k)
    p_0[0, 0] = t[0, 0]
    p_0[1, 0] = t[1, 0]
    p_0[2, 0] = t[2, 0]
    eye = np.eye(3, dtype=float)
    cov_list[i] = (eye - K @ h) @ cov_list[i]


def transition_model1(i, p_1, dt):
    x_dot = (dxu[0, i]) * dt * math.cos(p_1[2, 0])
    y_dot = (dxu[0, i]) * dt * math.sin(p_1[2, 0])

    p_1[0, 0] = p_1[0, 0] + x_dot
    p_1[1, 0] = p_1[1, 0] + y_dot
    t = displaceAngle(p_1[2, 0], (dxu[1, i] * dt))
    p_1[2, 0] = t

    return p_1


def getG1(i, p_1, dt):
    '''
    Parameters:
        x (Pose): The previous pose
        u (numpy array): The latest control input

    '''
    G = np.array([[1, 0, -1], [0, 1, -1], [0, 0, 1]], dtype=np.float32)
    G[0, 2] = -(dxu[0, i]) * dt * math.sin(p_1[2, 0])
    G[1, 2] = (dxu[0, i]) * dt * math.cos(p_1[2, 0])
    return G


def getV1(i, p_1, dt):
    '''
    Parameters:
        x (Pose): The previous pose
        u (numpy.array): The latest control input vector
    '''

    # Create the matrix
    V = np.array([[-1, -1], [-1, -1], [0, 1]], dtype=np.float64)

    # -- new values
    V[0, 0] = math.cos(p_1[2, 0]) * dt
    V[0, 1] = 0
    V[1, 0] = math.sin(p_1[2, 0]) * dt
    V[1, 1] = 0
    V[2, 0] = 0
    V[2, 1] = dt
    return V


def updateCov1(i, p_1, dt):
    global cov_list, m
    g = getG(i, p_1, dt)
    v = getV(i, p_1, dt)
    ahh = 0.1
    # set velocity noise covariance CHANGE LATER!!!
    m = np.array([[0.05 * (dxu[0, i]), 0], [0, 0.05 * (dxu[1, i])]])
    cov_list[i] = g @ cov_list[i] @ g.T + v @ m @ v.T
    return cov_list[i]


def update_cov21(i):
    global cov_list, cov_list2
    ahh = 0.01
    # **********************************************************
    # MAYBE CHANGE TO 0.1, 0.1, 0.001
    Q = np.array([[0.05, 0, 0], [0, 0.05, 0], [0.0005, 0, 0]])

    h = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cov_list2[i] = h @ cov_list[i] @ h.T + Q
    return cov_list2[i]


def predict1(i):
    global p_1, cov_list, dt

    p_1 = transition_model1(i, p_1, dt)
    cov_list[i] = updateCov1(i, p_1, dt)


def measure1(i):
    global x, cov_list, p_1
    h = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    K = cov_list[i] @ h.T @ np.linalg.inv(update_cov21(i))
    # maybe fix?

    k = np.array([[p_1[0, 0]], [p_1[1, 0]], [p_1[2, 0]]])
    q = np.array([[nx[0, i]], [nx[1, i]], [nx[2, i]]])

    t = k + K @ (q - k)
    p_1[0, 0] = t[0, 0]
    p_1[1, 0] = t[1, 0]
    p_1[2, 0] = t[2, 0]
    eye = np.eye(3, dtype=float)
    cov_list[i] = (eye - K @ h) @ cov_list[i]


uni_controller = create_clf_unicycle_pose_controller()
uni_barrier_cert = create_pr_unicycle_barrier_certificate_cent(safety_radius=safety_radius,
                                                               confidence_level=confidence_level)

def callback(data, args):
    global firstFlag, x, nx, p_0, p_1
    i = args

    # do i have to wrap the angle?
    theta = tf_conversions.transformations.euler_from_quaternion(
        [data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])[2]

    x[0, i] = data.pose.position.x
    x[1, i] = data.pose.position.y
    x[2, i] = theta

    l = np.array([[0], [0], [0]])

    noise = np.random.normal(0, 0.05, l.shape)
    # cancel noise for theta
    nx[0, i] = x[0, i] + noise[0, 0]
    nx[1, i] = x[1, i] + noise[1, 0]
    nx[2, i] = x[2, i]

    if firstFlag[i] == 1 and i == 0:
        p_0[0, 0] = nx[0, i]
        p_0[1, 0] = nx[1, i]
        p_0[2, 0] = nx[2, i]
        firstFlag[i] = 0

    if firstFlag[i] == 1 and i == 1:
        p_1[0, 0] = nx[0, i]
        p_1[1, 0] = nx[1, i]
        p_1[2, 0] = nx[2, i]
        firstFlag[i] = 0

    if i == 1:
        p_1[0, 0] = x[0, i]
        p_1[1, 0] = x[1, i]
        p_1[2, 0] = x[2, i]


    print("p_0's x", p_0[0,0])
    print("x's x", x[0,0])





def control_callback(event):
    global p_0, p_1, dxu
    #p = np.array([ [p_0[0,0],p_1[0,0]] , [p_0[1,0],p_1[1,0]] , [p_0[2,0],p_1[2,0]]  ])
    p = np.hstack((p_0, p_1))
    #dxu = uni_controller(p, goal_points)
    dxu = uni_barrier_cert(uni_controller(p, goal_points), p)

    twist131.linear.x = dxu[0, 0]
    twist131.linear.y = 0.0
    twist131.linear.z = 0.0
    twist131.angular.x = 0
    twist131.angular.y = 0
    twist131.angular.z = dxu[1, 0]

    twist139.linear.x = dxu[0, 1]
    twist139.linear.y = 0.0
    twist139.linear.z = 0.0
    twist139.angular.x = 0
    twist139.angular.y = 0
    twist139.angular.z = dxu[1, 1]
    publisher131.publish(twist131)
    publisher139.publish(twist139)




def ekf_update_function131(event):
    predict(0)
    measure(0)


def ekf_update_function139(event):
    predict1(1)
    measure1(1)


def central():
    rospy.Subscriber('/vrpn_client_node/Hus131' + '/pose', PoseStamped, callback, 0)
    rospy.Subscriber('/vrpn_client_node/Hus139' + '/pose', PoseStamped, callback, 1)

    timer = rospy.Timer(rospy.Duration(0.05), control_callback)
    ekf_timer131 = rospy.Timer(rospy.Duration(0.05), ekf_update_function131)
    ekf_timer139 = rospy.Timer(rospy.Duration(0.05), ekf_update_function139)
    # pos_compare_timer = rospy.Timer(rospy.Duration(50), pos_compare)
    # pos_compare()
    rospy.spin()


if __name__ == '__main__':

    try:
        central()
    except rospy.ROSInterruptException:
        print(rospy.ROSInterruptException)
