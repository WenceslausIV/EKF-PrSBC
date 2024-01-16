#!/usr/bin/env python3
from __future__ import print_function
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
import itertools
import numpy as np
from scipy.special import comb
from geometry_msgs.msg import TransformStamped, PoseStamped, Twist

rospy.sleep(2)
twist = Twist()
dt = 0.06
N = 1  # Number of robots
x = np.zeros((3, N))
goal_points = np.array([[0], [0], [0]])
dxu = np.zeros((2, N))
tdxu = np.zeros((2, N))
boundary_points = np.array([-1.0, -1.0, 1.0, 1.0])


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

def create_single_integrator_barrier_certificate_with_boundary(barrier_gain=100, safety_radius=0.17, magnitude_limit=0.2, boundary_points = np.array([-1.6, 1.6, -1.0, 1.0])):
    """Creates a barrier certificate for a single-integrator system with a rectangular boundary included.  This function
    returns another function for optimization reasons.

    barrier_gain: double (controls how quickly agents can approach each other.  lower = slower)
    safety_radius: double (how far apart the agents will stay)
    magnitude_limit: how fast the robot can move linearly.

    -> function (the barrier certificate function)
    """

    #Check user input types
    assert isinstance(barrier_gain, (int, float)), "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(safety_radius, (int, float)), "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_radius).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert safety_radius >= 0.12, "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m) plus the distance to the look ahead point used in the diffeomorphism if that is being used. Recieved %r." % safety_radius
    assert magnitude_limit > 0, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit


    def f(dxi, x):
        #Check user input types
        assert isinstance(dxi, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the single-integrator robot velocity command (dxi) must be a numpy array. Recieved type %r." % type(dxi).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the single integrator robot states (x) must be 2 ([x;y]). Recieved dimension %r." % x.shape[0]
        assert dxi.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the robot single integrator velocity command (dxi) must be 2 ([x_dot;y_dot]). Recieved dimension %r." % dxi.shape[0]
        assert x.shape[1] == dxi.shape[1], "In the function created by the create_single_integrator_barrier_certificate function, the number of robot states (x) must be equal to the number of robot single integrator velocity commands (dxi). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        
        # Initialize some variables for computational savings
        N = dxi.shape[1]
        num_constraints = int(comb(N, 2)) + 4*N
        A = np.zeros((num_constraints, 2*N))
        b = np.zeros(num_constraints)
        #H = sparse(matrix(2*np.identity(2*N)))
        H = 2*np.identity(2*N)

        count = 0
        for i in range(N-1):
            for j in range(i+1, N):
                error = x[:, i] - x[:, j]
                h = (error[0]*error[0] + error[1]*error[1]) - np.power(safety_radius, 2)

                A[count, (2*i, (2*i+1))] = -2*error
                A[count, (2*j, (2*j+1))] = 2*error
                b[count] = barrier_gain*np.power(h, 3)

                count += 1
        
        for k in range(N):
            #Pos Y
            A[count, (2*k, 2*k+1)] = np.array([0,1])
            b[count] = 0.4*barrier_gain*(boundary_points[3] - safety_radius/2 - x[1,k])**3;
            count += 1

            #Neg Y
            A[count, (2*k, 2*k+1)] = -np.array([0,1])
            b[count] = 0.4*barrier_gain*(-boundary_points[2] - safety_radius/2 + x[1,k])**3;
            count += 1

            #Pos X
            A[count, (2*k, 2*k+1)] = np.array([1,0])
            b[count] = 0.4*barrier_gain*(boundary_points[1] - safety_radius/2 - x[0,k])**3;
            count += 1

            #Neg X
            A[count, (2*k, 2*k+1)] = -np.array([1,0])
            b[count] = 0.4*barrier_gain*(-boundary_points[0] - safety_radius/2 + x[0,k])**3;
            count += 1
        
        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, 0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

        f = -2*np.reshape(dxi, (2*N,1), order='F')
        b = np.reshape(b, (count,1), order='F')
        result = qp(matrix(H), matrix(f), matrix(A), matrix(b))['x']
        #result = solver2.solve_qp(H, f, A, b, 0)[0]

        return np.reshape(result, (2, N), order='F')

    return f

def create_unicycle_barrier_certificate_with_boundary(barrier_gain=100, safety_radius=0.12, projection_distance=0.05, magnitude_limit=0.2, boundary_points = np.array([-1.6, 1.6, -1.0, 1.0])):
    """ Creates a unicycle barrier cetifcate to avoid collisions. Uses the diffeomorphism mapping
    and single integrator implementation. For optimization purposes, this function returns 
    another function.

    barrier_gain: double (how fast the robots can approach each other)
    safety_radius: double (how far apart the robots should stay)
    projection_distance: double (how far ahead to place the bubble)

    -> function (the unicycle barrier certificate function)
    """

    #Check user input types
    assert isinstance(barrier_gain, (int, float)), "In the function create_unicycle_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(safety_radius, (int, float)), "In the function create_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_radius).__name__
    assert isinstance(projection_distance, (int, float)), "In the function create_unicycle_barrier_certificate, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be an integer or float. Recieved type %r." % type(projection_distance).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_unicycle_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert safety_radius >= 0.12, "In the function create_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m). Recieved %r." % safety_radius
    assert projection_distance > 0, "In the function create_unicycle_barrier_certificate, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be positive. Recieved %r." % projection_distance
    assert magnitude_limit > 0, "In the function create_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit


    si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary(barrier_gain=barrier_gain, safety_radius=safety_radius+projection_distance, boundary_points=boundary_points)

    si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping(projection_distance=projection_distance)

    uni_to_si_dyn = create_uni_to_si_dynamics(projection_distance=projection_distance)

    def f(dxu, x):
        #Check user input types
        assert isinstance(dxu, np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the unicycle robot velocity command (dxu) must be a numpy array. Recieved type %r." % type(dxu).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 3, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the unicycle robot states (x) must be 3 ([x;y;theta]). Recieved dimension %r." % x.shape[0]
        assert dxu.shape[0] == 2, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the robot unicycle velocity command (dxu) must be 2 ([v;w]). Recieved dimension %r." % dxu.shape[0]
        assert x.shape[1] == dxu.shape[1], "In the function created by the create_unicycle_barrier_certificate function, the number of robot states (x) must be equal to the number of robot unicycle velocity commands (dxu). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxu.shape[0], dxu.shape[1])


        x_si = uni_to_si_states(x)
        #Convert unicycle control command to single integrator one
        dxi = uni_to_si_dyn(dxu, x)
        #Apply single integrator barrier certificate
        dxi = si_barrier_cert(dxi, x_si)
        #Return safe unicycle command
        return si_to_uni_dyn(dxi, x)

    return f


unicycle_position_controller = create_clf_unicycle_pose_controller()
unicycle_barrier_cert =  create_unicycle_barrier_certificate(safety_radius = 0.1)



msg = """
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

For Holonomic mode (strafing), hold down the shift key:
---------------------------
   U    I    O
   J    K    L
   M    <    >

t : up (+z)
b : down (-z)

anything else : stop

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%

CTRL-C to quit
"""

moveBindings = {
        'i':(1,0,0,0),
        'o':(1,0,0,-1),
        'j':(0,0,0,1),
        'l':(0,0,0,-1),
        'u':(1,0,0,1),
        ',':(-1,0,0,0),
        '.':(-1,0,0,1),
        'm':(-1,0,0,-1),
        'O':(1,-1,0,0),
        'I':(1,0,0,0),
        'J':(0,1,0,0),
        'L':(0,-1,0,0),
        'U':(1,1,0,0),
        '<':(-1,0,0,0),
        '>':(-1,-1,0,0),
        'M':(-1,1,0,0),
        't':(0,0,1,0),
        'b':(0,0,-1,0),
    }

speedBindings={
        'q':(1.1,1.1),
        'z':(.9,.9),
        'w':(1.1,1),
        'x':(.9,1),
        'e':(1,1.1),
        'c':(1,.9),
    }

class PublishThread(threading.Thread):
    def __init__(self, rate):
        super(PublishThread, self).__init__()
        self.publisher = rospy.Publisher('cmd_vel', Twist, queue_size = 1)
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.th = 0.0
        self.speed = 0.0
        self.turn = 0.0
        self.condition = threading.Condition()
        self.done = False

        # Set timeout to None if rate is 0 (causes new_message to wait forever
        # for new data to publish)
        if rate != 0.0:
            self.timeout = 1.0 / rate
        else:
            self.timeout = None

        self.start()

    def wait_for_subscribers(self):
        i = 0
        while not rospy.is_shutdown() and self.publisher.get_num_connections() == 0:
            if i == 4:
                print("Waiting for subscriber to connect to {}".format(self.publisher.name))
            rospy.sleep(0.5)
            i += 1
            i = i % 5
        if rospy.is_shutdown():
            raise Exception("Got shutdown request before subscribers connected")

    def update(self, x, y, z, th, speed, turn):
        self.condition.acquire()
        self.x = x
        self.y = y
        self.z = z
        self.th = th
        self.speed = speed
        self.turn = turn
        # Notify publish thread that we have a new message.
        self.condition.notify()
        self.condition.release()

    def stop(self):
        self.done = True
        self.update(0, 0, 0, 0, 0, 0)
        self.join()


unicycle_position_controller = create_clf_unicycle_pose_controller()
uni_barrier_cert = create_unicycle_barrier_certificate_with_boundary(barrier_gain=100, safety_radius=0.2, projection_distance=0.05, magnitude_limit=0.2, boundary_points = boundary_points)

    def run(self):
        twist = Twist()
        while not self.done:
            self.condition.acquire()
            # Wait for a new message or timeout.
            self.condition.wait(self.timeout)
            # Copy state into twist message.
            '''
            twist.linear.x = self.x * self.speed
            twist.linear.y = self.y * self.speed
            twist.linear.z = self.z * self.speed
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = self.th * self.turn
            '''
            tdxu[0,0] = self.x * self.speed
            tdxu[1,0] = self.th * self.turn
            dxu = uni_barrier_cert(tdxu, x)
            twist.linear.x = 
            self.condition.release()

            # Publish.
            self.publisher.publish(twist)

        # Publish stop message when thread exits.
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        self.publisher.publish(twist)


def getKey(key_timeout):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], key_timeout)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def vels(speed, turn):
    return "currently:\tspeed %s\tturn %s " % (speed,turn)

def callback(data, args):
    global x
    i = args

    # do i have to wrap the angle?
    theta = tf_conversions.transformations.euler_from_quaternion(
        [data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])[2]
  
    x[0, i] = data.pose.position.x
    x[1, i] = data.pose.position.y
    x[2, i] = theta
    
force_on = 0
def control_callback(event):
    dxu = unicycle_barrier_cert(tdxu, x)
    
    

def central():
    rospy.Subscriber('/vrpn_client_node/Limo' + '/pose', PoseStamped, callback, 0)
    timer = rospy.Timer(rospy.Duration(0.05), control_callback)
    rospy.spin()

if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('teleop_twist_keyboard')

    speed = rospy.get_param("~speed", 0.5)
    turn = rospy.get_param("~turn", 1.0)
    repeat = rospy.get_param("~repeat_rate", 0.0)
    key_timeout = rospy.get_param("~key_timeout", 0.0)
    if key_timeout == 0.0:
        key_timeout = None

    pub_thread = PublishThread(repeat)

    x = 0
    y = 0
    z = 0
    th = 0
    status = 0

    try:
        pub_thread.wait_for_subscribers()
        pub_thread.update(x, y, z, th, speed, turn)

        print(msg)
        print(vels(speed,turn))
        while(1):
            key = getKey(key_timeout)
            if key in moveBindings.keys():
                x = moveBindings[key][0]
                y = moveBindings[key][1]
                z = moveBindings[key][2]
                th = moveBindings[key][3]
            elif key in speedBindings.keys():
                speed = speed * speedBindings[key][0]
                turn = turn * speedBindings[key][1]

                print(vels(speed,turn))
                if (status == 14):
                    print(msg)
                status = (status + 1) % 15
            else:
                # Skip updating cmd_vel if key timeout and robot already
                # stopped.
                if key == '' and x == 0 and y == 0 and z == 0 and th == 0:
                    continue
                x = 0
                y = 0
                z = 0
                th = 0
                if (key == '\x03'):
                    break
 
            pub_thread.update(x, y, z, th, speed, turn)
            print(x,y,z,th)

    except Exception as e:
        print(e)

    finally:
        pub_thread.stop()

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
