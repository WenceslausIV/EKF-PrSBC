import numpy as np

N = 4 # Number of robots
x = np.zeros((3, N))
goal_points = np.array([[0., 0., 1., -1.], [-1., 1., 0., 0.], [math.pi / 2, -math.pi / 2, math.pi, 0.]])
dxu = x = np.zeros((2, N))
p = np.zeros((3, N))
cov_list = []
cov_list2 = []
for i in range(N):
    cov = np.full((3, 3), 1.0, dtype=np.float64)
    cov_list.append(cov)
    cov_list2.append(cov)

prevtheta = np.zeros((1, N))
firstFlag = 1
def transition_model(i, p, dt):

    x_dot = dxu[0,i] * dt * math.cos(p[2,i] + (dxu[1,i] * dt/ 2)) # do i still have to divide dxu[1] by 2?
    y_dot = dxu[0,i] * math.sin(p[2,i] + (dxu[1,i] / 2)) # do i still have to divide dxu[1] by 2?
    
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
    G[0,2] = -dxu[0,i] * dt * math.sin(p[2,i] + dxu[1,i]) 
    G[1,2] = dxu[0,i] * dt * math.cos(p[2,i] + dxu[1,i])

def getV(i, p, dt):
    '''
    Parameters:
        x (Pose): The previous pose
        u (numpy.array): The latest control input vector
    '''

    # Create the matrix
    V = np.array([[-1,-1],[-1,-1],[0, 1]],dtype=np.float64)

    # -- new values
    V[0,0] = cos(p[2,i] + dxu[2,i] / 2) * dt
    V[0,1] = -dxu[0,i] * sin(p[2,i] + dxu[2,i] / 2) * (dt/2)
    V[1,0] = cos(p[2,i] + dxu[2,i] / 2) * dt
    V[1,1] = dxu[0,i] * cos(p[2,i] + dxu[2,i] / 2) * (dt/2)
    V[2,0] = 0
    V[2,1] = dt
    return V

def updateCov(i, p, dt):
    global cov_list
    g = getG(i, p, dt)
    v = getV(i, p, dt)
    ahh = 0.05
    m = np.array([[ahh**2 * dxu[0,i],0],[0,ahh**2*dxu[1,i]]])
    cov_list[i] = g @ cov_list[i] @ g.T + v @ m @ v.T
    return cov_list[i]

def update_cov2(i):
    global cov_list, cov_list2
    ahh = 0.05
    Q = np.array([[ahh**2 ,0,0],[0,ahh**2,0],[0,0,ahh**2]])

    h = np.array([[1,0,0],[0,1,0],[0,0,1]])
    cov_list2[i] = h @ cov_list[i] @ h.T + Q
    return cov_list2[i]

def predict(i):
    global p, cov
    p = transition_model(i, p, dt)
    cov_list[i] = updateCov(i, p, dt)

def measure(i):
    global p, cov_list
    h = np.array([[1,0,0],[0,1,0],[0,0,1]])
    K = cov_list[i] @ h.T @ np.inv(update_cov2())
    p[0,i] = p[0,i] + K[0,0] @ (x[0,i] - p[0,i])
    p[1,i] = p[1,i] + K[1,1] @ (x[1,i] - p[1,i])
    p[2,i] = p[2,i] + K[2,2] @ (x[2,i] - p[2,i])
    eye = np.eye((3,3))
    cov_list[i] = (eye - K @ h) @ cov_list[i]

def callback(data, args):
    global prevtheta
    i = args

    #do i have to wrap the angle?
    theta = tf_conversions.transformations.euler_from_quaternion(
        [data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])[2]
    
    x[0, i] = data.pose.position.x
    x[1, i] = data.pose.position.y
    x[2, i] = theta
    prevtheta[1,i] = theta
    if firstFlag == 1:
        p[0,i] = data.pose.position.x
        p[1,i] = data.pose.position.y
        p[2,i] = theta
        firstFlag = 0

def control_callback(event):

    for i in range(N):
        dxu = unicycle_position_controller(x, goal_points)
        dxu = uni_barrier_cert(dxu, x)
        twist.linear.x = dxu[0,p]/5.
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = dxu[1,p]/5.
        publisher.publish(twist)


def central():
    rospy.Subscriber('/vrpn_client_node/Hus117' + '/pose', PoseStamped, callback, 0)
    rospy.Subscriber('/vrpn_client_node/Hus137' + '/pose', PoseStamped, callback, 1)
    rospy.Subscriber('/vrpn_client_node/Hus138' + '/pose', PoseStamped, callback, 2)
    rospy.Subscriber('/vrpn_client_node/Hus188' + '/pose', PoseStamped, callback, 3)
    timer = rospy.Timer(rospy.Duration(0.01), control_callback)
    rospy.spin()


if __name__ == '__main__':

    try:
        central()
    except rospy.ROSInterruptException:
        print(rospy.ROSInterruptException)
