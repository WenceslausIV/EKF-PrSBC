import numpy as np
import scipy.stats as stats
'''
x_rand_span_x = 0.085 * np.random.randint(1, 2, (1, N))  # setting up position error range for each robot,
x_rand_span_y = 0.085 * np.random.randint(1, 2, (1, N))
XRandSpan = np.concatenate((x_rand_span_x, x_rand_span_y))
'''
# I assumed that we have x pos error distribution(gaussian) and y pos error distribution(gaussian), and they are independent.
# this code includes a portion of create_si_pr_barrier_certificate_centralized and the whole trap_cdf_inv function to visualize my idea in one file
N = 4
robotGaussianDistInfox = np.array([[mean1, mean2, mean3, mean4],[std1, std2, std3, std4]]) #all the robots' x pos distribution info including mean and std
robotGaussianDistInfoy = np.array([[mean1, mean2, mean3, mean4],[std1, std2, std3, std4]]) #all the robots' y pos distribution info including mean and std
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
        XRandSpan[0, i] = robotGaussianDistInfox[0,i] - (z_value * robotGaussianDistInfox[1,i])
        XRandSpan[0, j] = robotGaussianDistInfox[0,j] - (z_value * robotGaussianDistInfox[1,j])
      
        XRandSpan[1, i] = robotGaussianDistInfoy[0,i] - (z_value * robotGaussianDistInfoy[1,i])
        XRandSpan[1, j] = robotGaussianDistInfoy[0,j] - (z_value * robotGaussianDistInfoy[1,j])
        #until here
        
        b2_x, b1_x, sigma = trap_cdf_inv(XRandSpan[0, i], XRandSpan[0, j], x[0, i] - x[0, j])
        b2_y, b1_y, sigma = trap_cdf_inv(XRandSpan[1, i], XRandSpan[1, j], x[1, i] - x[1, j])
        '''
        To sum up
        XRandSpan[0,i] is robot i's x error gaussian distribution's confidence level % value, for example if confidence level is 95% then its the 95% value of the distribution
        so this way, trap_cdf_inv() sigma can be 1 and keep the code
        '''
              


def trap_cdf_inv(a, c, delta):
    
    
  
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
        if sigma > area_vec[1]:  # right triangle area
            b1 = (A + C) - 2 * C * np.sqrt((1 - sigma) / (1 - area_vec[1]))
            b2 = -(A + C) + 2 * C * np.sqrt((1 - sigma) / (1 - area_vec[1]))

            b1 = b1 + delta
            b2 = b2 + delta  # apply shift here due to xi - xj

        elif sigma > area_vec[0] and sigma <= area_vec[1]:  # in between the triangle part
            b1 = -(A - C) + (sigma - area_vec[0]) / h  # assuming > 50%, then b1 should > 0
            b2 = -b1

            b1 = b1 + delta
            b2 = b2 + delta  # apply shift here due to xi - xj

            # note that b1 could be > or < b2, depending on whether sigma > or < .5

        elif sigma <= area_vec[0]:
            b1 = -(A + C) + 2 * C * np.sqrt(sigma / area_vec[0])  # assuming > 50%, then b1 should > 0
            b2 = -b1

            b1 = b1 + delta
            b2 = b2 + delta  # apply shift here due to xi - xj

        else:
            print('first triangle, which is not allowed as long as we assume sigma > 50%')

    return b2, b1, sigma
