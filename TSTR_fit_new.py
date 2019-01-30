# TSTR_fit

# specular follows a trowbridge-reitz model (both P and G)
# diffuse follows a trowbridge-reitz distribution of surface normals combined with lambertian diffusion for calculating correction factor N
# other distributions can be chosen as well

# described in Reflectance of Polytetrafluoroethylene (PTFE) for Xenon_Scintillation Light, LIP-Coimbra; some changes made here

# called geometrical optical approximation (GOA) model in other Coimbra paper,
# but this doesn't include the correction factor N

import numpy as np
import scipy.optimize
import scipy.special
import copy
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.colors

# following equations could be refined for mu !~ mu_0, but teflon has mu~mu_0


# this function is used in calculating the total reflectance
def G_calc(theta_r, phi_r, theta_i, n_0, polarization, parameters):
    rho_L = parameters[0]
    n = parameters[1]
    gamma = parameters[2]

    theta_i_prime = 0.5 * np.arccos(np.cos(theta_i) * np.cos(theta_r) -
                                    np.sin(theta_i) * np.sin(theta_r) * np.cos(phi_r))

    theta_r_prime = 0.5 * np.arccos(np.cos(theta_i) * np.cos(theta_r) -
                                    np.sin(theta_i) * np.sin(theta_r))

    def G_prime(theta):
        return 2 / (1 + np.sqrt(1 + np.power(gamma * np.tan(theta), 2)))

        # this has negative of the inside of the H functions from the paper, I think it was a typo

    G = H(np.pi / 2 - theta_i_prime) * H(np.pi / 2 - theta_r_prime) * \
        G_prime(theta_i) * G_prime(theta_r)

    return G

def F_s(theta_i, n_0, n):
    if np.size(theta_i)>1: return F_s_array(theta_i, n_0, n)
    else: return F_s_single(theta_i, n_0, n)
    
# polarized electric field perpendicular to plane of incidence (vertical)
def F_s_single(theta_i, n_0, n):
    if np.abs(n_0 / n * np.sin(theta_i)) > 1:
        # total internal reflection
        return 1
    theta_t = np.arcsin(n_0 / n * np.sin(theta_i))
    return np.power((n_0 * np.cos(theta_i) - n * np.cos(theta_t)) /
                    (n_0 * np.cos(theta_i) + n * np.cos(theta_t)), 2)
    
# polarized electric field perpendicular to plane of incidence (vertical)
# designed to work when theta_i is an array; more efficient for arrays, but much 
# slower for single values due to masked arrays being slow
def F_s_array(theta_i, n_0, n):
    mask = np.abs(n_0 / n * np.sin(theta_i)) > 1
    #print(type(mask))
    theta_i_mask = np.ma.masked_array(theta_i, mask)
    theta_t = np.ma.arcsin(n_0 / n * np.sin(theta_i_mask))
    F_calc = np.power((n_0 * np.cos(theta_i) - n * np.cos(theta_t)) /
                    (n_0 * np.cos(theta_i) + n * np.cos(theta_t)), 2)
    #print(F_calc)
    return np.ma.filled(F_calc,fill_value=1)

def F_p(theta_i, n_0, n):
    if np.size(theta_i)>1: return F_p_array(theta_i, n_0, n)
    else: return F_p_single(theta_i, n_0, n)
    
# polarized electric field parallel to plane of incidence (horizontal)
def F_p_single(theta_i, n_0, n):
    if np.abs(n_0 / n * np.sin(theta_i)) > 1:
        # total internal reflection
        return 1
    theta_t = np.arcsin(n_0 / n * np.sin(theta_i))
    return np.power((n_0 * np.cos(theta_t) - n * np.cos(theta_i)) /
                    (n_0 * np.cos(theta_t) + n * np.cos(theta_i)), 2)

# polarized electric field parallel to plane of incidence (horizontal)
# designed to work when theta_i is an array; more efficient for arrays, but much 
# slower for single values due to masked arrays being slow
def F_p_array(theta_i, n_0, n):
    # Have to check for arcsin errors; using masked array to handle multiple theta_i values
    mask = np.abs(n_0 / n * np.sin(theta_i)) > 1
    #print(type(mask))
    theta_i_mask = np.ma.masked_array(theta_i, mask)
    theta_t = np.ma.arcsin(n_0 / n * np.sin(theta_i_mask))
    #print(type(theta_t))
    F_calc = np.power((n_0 * np.cos(theta_t) - n * np.cos(theta_i)) /
                    (n_0 * np.cos(theta_t) + n * np.cos(theta_i)), 2)
    #print(type(F_calc))
    return np.ma.filled(F_calc,fill_value=1)


def F_unpolarized(theta_i, n_0, n):
    return 0.5 * (F_s(theta_i, n_0, n) + F_p(theta_i, n_0, n))


def F(theta_i, n_0, n, polarization):
    return polarization * F_p(theta_i, n_0, n) + (1 - polarization) * F_s(theta_i, n_0, n)


# heaviside step function
def H(x):
    return 0.5 * (np.sign(x) + 1)


# BRIDF as described in paper
# takes an array, easy to plot
# parameters has form [rho_L, n, gamma]; optional fourth parameter K
# All angles are in degrees
# average_angle sets aperture size (diameter) to average over (no avg if 0)
# precision sets grid spacing for 2D averaging (1D average used if < 0)
def BRIDF_plotter(theta_r_in_degrees_array, phi_r_in_degrees, theta_i_in_degrees, n_0, polarization, parameters, average_angle=0, precision=-1, sigma_theta_i=2):
    phi_r = phi_r_in_degrees * np.pi / 180
    theta_i = theta_i_in_degrees * np.pi / 180
    precision_rad = precision * np.pi / 180
    if precision <= 0: # Skip 2D averaging
        return_array = []
        for theta_r_in_degrees in theta_r_in_degrees_array:
            theta_r = np.pi * theta_r_in_degrees / 180
            return_array.append(BRIDF(theta_r, phi_r, theta_i, n_0, polarization, parameters, precision=precision_rad))
        if average_angle <= 0:
            return return_array
        else: # Just average adjacent points in plane
            return average_by_angle(theta_r_in_degrees_array, return_array, average_angle)
    else: # Do a full 2D average across both theta_r and phi_r, with grid spacing given by precision
        avgs = []
        log_rho_L = np.log(parameters[0])
        log_n_minus_one = np.log(parameters[1]-1)
        log_gamma = np.log(parameters[2])
        if len(parameters)>3: 
            if parameters[3]>0: log_K = np.log(parameters[3])
            else: log_K = None
        else: log_K = None
        for theta_r_in_degrees in theta_r_in_degrees_array: # For each point, calculate the BRIDF along a grid nearby and average it
            avgs.append(average_BRIDF([theta_r_in_degrees,phi_r_in_degrees,theta_i_in_degrees,n_0,polarization], log_rho_L, log_n_minus_one, log_gamma, log_K, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
        return avgs

# Takes a set of viewing angles and intensities and returns the average, for each point, of adjacent points w/in +/- angle/2
# Assumes angles are sorted; if arrays include multiple runs, will not average across runs
def average_by_angle(theta_r_in_degrees_array, intensity_array, angle):
    #print("avg angle: "+str(angle))
    #print("theta array: "+str(theta_r_in_degrees_array))
    grouped = [[] for i in intensity_array]
    angles = [[] for i in intensity_array]
    for i in range(len(intensity_array)):
        for j in range(i,len(intensity_array)):
            if np.abs(theta_r_in_degrees_array[j] - theta_r_in_degrees_array[i]) < angle / 2.:
                grouped[i].append(intensity_array[j])
                angles[i].append(theta_r_in_degrees_array[j])
            else: break # if we've left the averaging angle range, stop looking forward to avoid combining runs
        for j in range(i,0,-1):
            if np.abs(theta_r_in_degrees_array[j] - theta_r_in_degrees_array[i]) < angle / 2.:
                grouped[i].append(intensity_array[j])
                angles[i].append(theta_r_in_degrees_array[j])
            else: break # if we've left the averaging angle range, stop looking backward to avoid combining runs
    averaged = [np.average(grouped[i]) for i in range(len(intensity_array))]
    return averaged

# Not updated yet to allow averaging
def BRIDF_radiance_plotter(theta_r_in_degrees_array, phi_r_in_degrees, theta_i_in_degrees, n_0, polarization, parameters):
    phi_r = phi_r_in_degrees * np.pi / 180
    theta_i = theta_i_in_degrees * np.pi / 180
    return_array = []
    for theta_r_in_degrees in theta_r_in_degrees_array:
        theta_r = np.pi * theta_r_in_degrees / 180
        return_array.append(BRIDF(theta_r, phi_r, theta_i, n_0, polarization, parameters) / np.cos(theta_r))
    return return_array

# Not updated yet to allow averaging
def BRIDF_specular_plotter(theta_r_in_degrees_array, phi_r_in_degrees, theta_i_in_degrees,
                           n_0, polarization, parameters):
    phi_r = phi_r_in_degrees * np.pi / 180
    theta_i = theta_i_in_degrees * np.pi / 180
    return_array = []
    for theta_r_in_degrees in theta_r_in_degrees_array:
        theta_r = np.pi * theta_r_in_degrees / 180
        return_array.append(BRIDF_specular(theta_r, phi_r, theta_i, n_0, polarization, parameters))
    return return_array

# Not updated yet to allow averaging
def BRIDF_diffuse_plotter(theta_r_in_degrees_array, phi_r_in_degrees, theta_i_in_degrees,
                          n_0, polarization, parameters):
    phi_r = phi_r_in_degrees * np.pi / 180
    theta_i = theta_i_in_degrees * np.pi / 180
    return_array = []
    for theta_r_in_degrees in theta_r_in_degrees_array:
        theta_r = np.pi * theta_r_in_degrees / 180
        return_array.append(BRIDF_diffuse(theta_r, phi_r, theta_i, n_0, polarization, parameters))
    return return_array

# Analytical calculation of BRIDF function, using Coimbra model
# If parameters includes K, specular spike is included; precision sets length scale of delta function in specular spike
# All angles should be in radians at this stage
def BRIDF_pair(theta_r, phi_r, theta_i, n_0, polarization, parameters, precision=-1):

    t0=time.time()
    rho_L = parameters[0]
    n = parameters[1]
    gamma = parameters[2]

    # Local angle, relative to microfacet; from page 85 of Claudio's thesis
    theta_prime = 0.5 * np.arccos(np.cos(theta_i) * np.cos(theta_r) -
        np.sin(theta_i) * np.sin(theta_r) * np.cos(phi_r))

    theta_i_prime = theta_prime
    theta_r_prime = theta_prime

    # Trowbridge-Reitz micro-facet angle distribution, p. 88 of Claudio's thesis; normalized w/o shadowing/masking factor G in integral
    def P(alpha_):
        return np.power(gamma, 2) / \
               (np.pi * np.power(np.cos(alpha_), 4) *
                np.power(np.power(gamma, 2) + np.power(np.tan(alpha_), 2), 2))

    # Cook-Torrance distribution, p. 87
    # def P(alpha_):
        # return np.exp(-np.tan(alpha_)**2/gamma**2) / (np.pi * gamma**2 * np.cos(alpha_)**4)
		
    # New empirical distribution, based off of Lorentzian distribution in tan(alpha)
	# Factor in front ensures normalization; calculated numerically, as integral doesn't have a closed form
	# Normalization is correct to w/in 1% for gamma_prime from 0.05-0.3, 4% for gamma_prime from 0.03-0.05
    # def P(alpha_):
        # gamma_prime = gamma/2 # scale so that this gamma is similar to gamma in CT and TR distributions
        # return (1.3981*gamma_prime+0.13) / (np.pi*gamma_prime**2*(1+(np.tan(alpha_)/gamma_prime)**2))
			
    # Bivariate-Cauchy distribution, from Claudio; distribution is normalized
    # def P(alpha_):
        # gamma_prime = gamma*0.74 # scale so that this gamma is similar to gamma in CT and TR distributions
        # return gamma_prime*(gamma_prime+1)/(2*np.pi*np.cos(alpha_)**3*(gamma_prime**2+np.tan(alpha_)**2)**(3/2))
			
    # Check probably unnecessary; if cos errors pop up, will have to fix to work for arrays 
    # if theta_i == theta_r:
        # alpha_specular = 0
    # else:
        # alpha_specular = np.arccos((np.cos(theta_i) + np.cos(theta_r)) / (2 * np.cos(theta_i_prime)))
    # Micro-facet angle for reflection into angle theta_r, also from p. 85
    alpha_specular = np.arccos((np.cos(theta_i) + np.cos(theta_r)) / (2 * np.cos(theta_i_prime)))

    P_ = P(alpha_specular)

    t1=time.time()
    #print("theta_prime, alpha_specular, P calculation time: {0}".format(t1-t0))
	
    # Wolff correction factor, p. 114 of thesis; uses global angles
	# Breaking up into two parts: incident Fresnel factor (always scalar since theta_i is) and outgoing (can be an array)
	# First deal w/ incident angle part of W: W=0 if theta_i > theta_crit = arcsin(n/n0) and n0>n; fraction making through should be integral of P*G*W
	# but N is already integral of P*G; fraction of angles below theta_crit is integral of P(alpha)*sin(alpha)*cos(alpha)*dalpha*dphi_alpha
	# 1-F(theta_i) is very close to 1 below theta_crit, so can just use integral above (skip G)
	# Difficulty is in 2D integral: bounds of integration of microfacet angles (alpha, phi_alpha) are where theta_prime < theta_crit;
	# theta_prime = arccos(cos(theta_i)*cos(alpha)+sin(theta_i)*sin(alpha)*cos(phi_alpha)) 
	# Algorithm:
	# Calculate phi_alpha_max using eqn below (above some inclination, no alpha achieves required condition)		
	# Then break up [-phi_alpha_max, phi_alpha_max] into N parts and for each discrete phi_alpha:
	#		calculate alpha_crit (where theta_prime = theta_crit)
	#		sum up P(alpha)*sin(alpha)*cos(alpha)*dalpha*dphi_alpha for alpha in range [alpha_crit, pi/2]
	# Add up all sums
    def W_crit(theta, theta_c, polarization=0.5, n_phi_alpha=25, n_alpha=100): 
		# Assumes theta is a scalar
        phi_alpha_m = phi_alpha_max(theta, theta_c) # Max phi_alpha where there is an angle alpha that can give theta_prime<theta_crit
        dphi_alpha = np.pi/n_phi_alpha
        d_alpha = np.pi/2/n_alpha
        n_phi=round(2*phi_alpha_m/dphi_alpha)
        # if np.size(theta)>1:
            # phi_alpha_range = [np.linspace(-phi_i,phi_i,n_phi) for phi_i in phi_alpha_m]#= np.empty((np.size(theta),n_phi))
            # for ii in arange(np.size(theta)):
                # phi_alpha_range[ii,:] = np.linspace(-phi_alpha_m,phi_alpha_m,n_phi)
        # else:
        phi_alpha_range = np.linspace(-phi_alpha_m,phi_alpha_m,n_phi)
        #print(dphi_alpha, phi_alpha_range[1]-phi_alpha_range[0])
        W_sum = 0
        for phi_alpha in phi_alpha_range:
            alpha_crit = alpha_calc(theta, theta_c, phi_alpha)
            alpha_range = np.linspace(alpha_crit, np.pi/2, round((np.pi/2-alpha_crit)/d_alpha))
            #print(d_alpha, alpha_range[1]-alpha_range[0])
            #th_p = theta_prime_calc(theta, alpha_range, phi_alpha)
            #W_list = 1 - F(th_p, n_0, n, polarization) # Can calculate Fresnel factor, multiply in sum, but slows down immensely, doesn't change value much
            W_sum += np.sum(P(alpha_range)*np.sin(alpha_range)*np.cos(alpha_range)*d_alpha*dphi_alpha)
        return W_sum
	
	# Incident part of W, W_i
    if np.size(theta_i) > 1:
        # W_i = 1 - F(theta_i, n_0, n, polarization) # Use this if skipping W_crit calculation 
        theta_i_mean = np.mean(theta_i)
        if n_0 > n and np.abs(n_0 / n * np.sin(theta_i_mean)) >= 1: # Above critical angle, just calculate W_crit for average angle
            theta_crit = np.arcsin(n/n_0)
            W_i = W_crit(theta_i_mean, theta_crit, polarization)
        else:
            W_i = 1 - F(theta_i, n_0, n, polarization) # could speed up using theta_i_mean
    else:
        if n_0 > n and np.abs(n_0 / n * np.sin(theta_i)) >= 1:
            theta_crit = np.arcsin(n/n_0)
            W_i = W_crit(theta_i, theta_crit, polarization)
            #W_i = 1 - F(theta_i, n_0, n, polarization)
        else:
           W_i = 1 - F(theta_i, n_0, n, polarization) 
	
    # Operating on masked arrays is slow; only use them if needed (inputs are arrays)
    if np.size(theta_r) > 1: # Have to use masks
        theta_r_mean = np.mean(theta_r)
        if n_0 > n and np.abs(n_0 / n * np.sin(theta_r_mean)) >= 1: # Above critical angle, just calculate W_crit for average angle
            theta_crit = np.arcsin(n/n_0)
            W_o = W_crit(theta_r_mean, theta_crit, polarization=0.5)
        else:
            mask = np.abs(n_0 / n * np.sin(theta_r)) >= 1
            theta_r_mask = np.ma.masked_array(theta_r, mask)
            W_o = (1 - F_unpolarized(np.ma.arcsin(n_0 / n * np.sin(theta_r_mask)), n, n_0))
            W_o = np.ma.filled(W_o, fill_value=0)
    else:
		# theta_r is a single value
        if n_0 > n and np.abs(n_0 / n * np.sin(theta_r)) >= 1:
			# the W expression would have an arcsin error
            theta_crit = np.arcsin(n/n_0)
            W_o = W_crit(theta_r, theta_crit, polarization=0.5) 
            #print("W_o critical calculation")
            #W_o = 0
        else:
            # There was a typo here (?), I changed by moving parentheses and taking a reciprocal
            W_o = (1 - F_unpolarized(np.arcsin(n_0 / n * np.sin(theta_r)), n, n_0))
    W = W_i*W_o 

    # def omega(n,n_p,th): # Solid angle factor from beams expanding due to Snell's law
        # return np.cos(th)/np.sqrt(1-(n_p/n)**2*np.sin(th)**2)	
    # W=W*omega(n,n_0,theta_r)
	
    # theta_r_list = np.linspace(0.,np.pi/2,102)
    # #W_list = [(1 - F(theta_i, n_0, n, polarization)) * (1 - F_unpolarized(np.arcsin(n_0 / n * np.sin(t_r)), n, n_0)) for t_r in theta_r_list]
    # W_list = [(1 - F_unpolarized(np.arcsin(n_0 / n * np.sin(t_r)), n, n_0)) for t_r in theta_r_list]
    # plt.plot(theta_r_list*180./np.pi,W_list )
    # plt.show()
			
    t2=time.time()
    #print("W calculation time: {0}".format(t2-t1))
			
	# Shadowing and masking, for Trowbridge-Reitz distribution; p. 108 of thesis
    def G_prime(theta):
        return 2 / (1 + np.sqrt(1 + np.power(gamma * np.tan(theta), 2)))
			
    # Shadowing and masking, for Cook-Torrance (also called Beckman?) distribution
	# From eq. 4.80, p. 108 of thesis, itself taken from https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    # def G_prime(theta):
        # th = np.abs(theta) # Should be symmetric (definition of theta assumes positive)
        # # if theta=0, tan(theta)=0 and m_t -> inf, gauss_m=>0, erf(m_t)->1, G->1; if theta=pi/2, tan(theta)=>inf, m_t=>0, erf(m_t)=>0,gauss_m=>inf, G->0
        # if np.size(th) > 1: # For many angles, use masks to account for special cases
            # mask0 = np.tan(th)==0 # Mask if th=0
            # theta_mask = np.ma.masked_array(th, mask0)
            # m_t = 1/(gamma*np.ma.tan(theta_mask)) # m_t, if finite
            # m_t = np.ma.filled(m_t, fill_value=np.inf) 
            # gauss_m = gamma*np.ma.tan(theta_mask)*np.exp(-m_t**2)/np.sqrt(np.pi)# mask if m_t = 0			
            # gauss_m = np.ma.filled(gauss_m, fill_value=0)
            # return 2/(1 + scipy.special.erf(m_t) + gauss_m)
        # elif th==0: return 1 # Special cases that give infinities
        # elif th==np.pi/2: return 0
        # else:
            # m_t = 1/(gamma*np.tan(th))
            # gauss_m = np.exp(-m_t**2)/(m_t*np.sqrt(np.pi))
            # return 2/(1 + scipy.special.erf(m_t) + gauss_m)
			
    # this has negative of the inside of the H functions from the paper, I think it was a typo
    G = H(np.pi / 2 - theta_i_prime) * H(np.pi / 2 - theta_r_prime) * \
        G_prime(theta_i) * G_prime(theta_r)
			
    t3=time.time()
    #print("G calculation time: {0}".format(t3-t2))
			
    # Oren-Nayar correction factor(s) (1-A+B)*cos(theta_i);
    # From p. 8 of arXiv:0910.1056v1 (Reflectance of PTFE for Xenon Scintillation Light), using Torrance-Sparrow (not Trowbridge-Reitz)
    # (For comparison, Claudio's thesis p. 117 has Oren-Nayar correction factor, N, for Trowbridge-Reitz)
    theta_m = np.minimum(theta_i, theta_r)

    theta_M = np.maximum(theta_i, theta_r)
	
    use_N_TR=True
    # Model from Claudio's thesis, p. 117; calculated using Trowbridge-Reitz distribution
    if use_N_TR:
        gamma_squared = gamma**2
        script_n_0 = 1 / (1 - gamma_squared) - \
            gamma_squared / (1 - gamma_squared) * \
            np.arctanh(np.sqrt(1 - gamma_squared)) / np.sqrt(1 - gamma_squared)

        script_n = gamma_squared / (2 * (1 - gamma_squared)) - \
            gamma_squared * (2 - gamma_squared) / (1 - gamma_squared) * \
            np.arctanh(np.sqrt(1 - gamma_squared)) / np.sqrt(1 - gamma_squared)

        N = G_prime(theta_i) * G_prime(theta_r) * \
            (script_n_0 - np.tan(theta_i) * np.tan(theta_r) * np.cos(phi_r) * script_n)
    # Model from p. 8 of arXiv:0910.1056v1 (Reflectance of PTFE for Xenon Scintillation Light)
	# Actually uses Torrance-Sparrow distribution!
    # Note that this is the same as eq. 4.98 from Claudio's thesis, but with C=0 and gamma proportional to sigma_alpha
    # also Claudio's thesis has some typos, e.g. sign of B and inside H is wrong
	# Can compare to Oren-Nayar paper (DOI 10.1145/192161.192213), eq. 27 or 30; here we choose coordinates so phi_i=0
	# The simplified version used in the paper is eq. 30 
    else:
        A = 0.5 * np.power(gamma, 2) / (np.power(gamma, 2) + 0.92)

        B = 0.45 * np.power(gamma, 2) / (np.power(gamma, 2) + 0.25) * \
            H(np.cos(phi_r)) * np.cos(phi_r) * np.sin(theta_M) * np.tan(theta_m)
        
        N = 1 - A + B
        use_second_order = False # allows reflected light a second reflection; has very little effect on shape
        if use_second_order:
            C = 0.17 * np.power(gamma, 2) / (np.power(gamma, 2) + 0.36) * \
				(1-(2*theta_m/np.pi)**2*np.cos(phi_r))
            N = N + rho_L*C

    t4=time.time()
    #print("N calculation time: {0}".format(t4-t3))
			
    # Use Fresnel factor relative to local angle theta_i_prime
    F_ = F(theta_i_prime, n_0, n, polarization)

    t5=time.time()
    #print("F calculation time: {0}".format(t5-t4))
	
    # add specular spike, if parameters length >3
    specular_delta=0
    if len(parameters)>3:    
        K = parameters[3]
        # this is where it differs from semi-empirical fit, Lambda was slightly different 
        # see Silva Oct. 2009 "A model of the reflection distribution in the vacuum ultra violet region"
        C = specular_spike_norm(theta_r, theta_i, K)
        # integrated over the photodiode solid angle, the delta functions go away; make sure spike isn't double counted
        is_theta_spec = np.logical_or(np.abs(theta_r - theta_i) < precision/2., theta_r==theta_i+precision/2.) 
        is_phi_spec = np.logical_or(np.abs(phi_r) < precision/2.,phi_r==precision/2.) 
        is_specular = np.logical_and(is_theta_spec, is_phi_spec)
        #print(is_specular)
        specular_delta = is_specular/precision**2 # only one point in grid is non-zero; magnitude scales w/ grid size to preserve average
    else:
        C = 0
    
    # Empirical addition of error function to specular lobe, for data in LXe
    # theta_cutoff = 67
    # beta = 1
    # erf_factor = 0.5*(1+scipy.special.erf((theta_r-theta_cutoff)/beta))
    #print("W: {0}, N: {1}".format(W,(1-A+B)))
    
    # Semi-empirical formula from p. 121
    # specular component is split into specular lobe w/ normalization=1-C and specular spike w/ normalization=C
	# P, G, and (1-A+B) or N depend on microfacet distribution
    return [ (1-C) * F_ * G * P_ / (4 * np.cos(theta_i)) + C * F_ * G * specular_delta,
        rho_L / np.pi * W * N * np.cos(theta_r)]


def BRIDF_specular(theta_r, phi_r, theta_i, n_0, polarization, parameters, precision=-1):
    return BRIDF_pair(theta_r, phi_r, theta_i, n_0, polarization, parameters, precision=precision)[0]


def BRIDF_diffuse(theta_r, phi_r, theta_i, n_0, polarization, parameters, precision=-1):
    return BRIDF_pair(theta_r, phi_r, theta_i, n_0, polarization, parameters, precision=precision)[1]


def BRIDF(theta_r, phi_r, theta_i, n_0, polarization, parameters, precision=-1):
    pair = BRIDF_pair(theta_r, phi_r, theta_i, n_0, polarization, parameters, precision=precision)
    return pair[0] + pair[1]

def theta_prime_calc(th_i, al, phi_al):
	# Local angle for microfacet at angle al, inclination phi_al, with global incident angle th_i
	# Note that because of the definition of arccos, this is always > 0, so should only be used when expected to be positive
	# Works for array inputs
	return np.arccos(np.cos(th_i)*np.cos(al)+np.sin(th_i)*np.sin(al)*np.cos(phi_al))

def alpha_calc(th_i, th_p, phi_al):
	# Microfacet angle alpha needed so that local angle is th_p, for global incident angle th_i and microfacet inclination phi_al
	# By the angle definitions, alpha_tmp has the range [-pi, +pi]
	# But the plane is unchanged by rotations by pi, so shift makes the range [-pi/2, +pi/2]
	# Works for array inputs
    alpha_tmp = np.pi/2-np.arccos(np.cos(th_p)/np.sqrt(np.cos(th_i)**2+np.sin(th_i)**2*np.cos(phi_al)**2))\
            -np.arctan2(np.cos(th_i),np.sin(th_i)*np.cos(phi_al))
    shift = (alpha_tmp < -np.pi/2)*np.pi-(alpha_tmp > np.pi/2)*np.pi
    return alpha_tmp + shift

def phi_alpha_max(th_i, th_p):
	# Max microfacet inclination phi_alpha such that local angle th_p can be achieved for some microfacet angle alpha, 
	# given incident angle th_i
	# Works for array inputs
	return np.arccos(np.sqrt(np.cos(th_p)**2-np.cos(th_i)**2)/np.sin(th_i)**2)
	
def specular_spike_norm(theta_r, theta_i, K):
    return np.exp(-K / 2 * (np.cos(theta_i) + np.cos(theta_r))) # Version from Coimbra paper
    # return np.exp(-K *np.cos(theta_i)) # Version from Claudio's thesis
	# Choice of version doesn't affect specular spike bc of delta function, but does affect specular lobe
	# Version w/ theta_r effectively enhances specular lobe at viewing angles above specular, reduces below
    
# Calculates BRIDF as a function of parameters; inputs are in degrees, converted to radians; can be arrays
# independent variables has the form [theta_r_in_degrees, phi_r_in_degrees, theta_i_in_degrees, n_0, polarization]
def unvectorized_fitter(independent_variables, log_rho_L, log_n_minus_one, log_gamma, log_K=None, precision=-1):
    theta_r = independent_variables[0] * np.pi / 180
    phi_r = independent_variables[1] * np.pi / 180
    theta_i = independent_variables[2] * np.pi / 180
    n_0 = independent_variables[3]
    polarization = independent_variables[4]
    precision = precision * np.pi / 180
    rho_L = np.exp(log_rho_L)
    n = np.exp(log_n_minus_one) + 1
    gamma = np.exp(log_gamma)
    parameters = [rho_L, n, gamma]
    if log_K is not None:
        parameters.append(np.exp(log_K))
    return BRIDF(theta_r, phi_r, theta_i, n_0, polarization, parameters, precision=precision)


# independent variables without angle has the form [theta_r_in_degrees, phi_r_in_degrees, n_0, polarization]
def unvectorized_fitter_with_angle(independent_variables_without_angle, theta_i, log_rho_L, log_n_minus_one, log_gamma, log_K=None, precision=-1):
    theta_r = independent_variables_without_angle[0] * np.pi / 180
    phi_r = independent_variables_without_angle[1] * np.pi / 180
    n_0 = independent_variables_without_angle[2]
    polarization = independent_variables_without_angle[2]
    precision = precision * np.pi / 180
    rho_L = np.exp(log_rho_L)
    n = np.exp(log_n_minus_one) + 1
    gamma = np.exp(log_gamma)
    parameters = [rho_L, n, gamma]
    if log_K is not None:
        parameters.append(np.exp(log_K))
    return BRIDF(theta_r, phi_r, theta_i, n_0, polarization, parameters, precision=precision)


# gets weights to use for the gaussian distribution of incident angles
# x is the array of incident angles to get weights for
# outputs weights that should correspond to a list of N points separated by 
def get_relative_gaussian_weights(x, mu, sigma):
    return [np.exp(-(x_ - mu) * (x_ - mu) / (2 * sigma * sigma)) for x_ in x]


def average_BRIDF(independent_variables, log_rho_L, log_n_minus_one, log_gamma, log_K, average_angle, precision=0.25, sigma_theta_i=2.0):
    # For a single point, calculate BRIDF values on a grid within a circle of diameter average_angle from center
    # and average them all; grid spacing is given by precision; all angles in degrees
    # Timing tests suggest that there's almost no speed up between a precision of 0.5 and 1 (for average_angle=4), and
    # only a slight speed up for 0.5 vs 0.25, so a precision of 0.5 or 0.25 seems reasonable
    # If sigma_theta_i>0, also add smearing in theta_i with width sigma_theta_i, to simulate the finite width of the incoming light
    # 0.14" collimator ID, 7" length -> arcsin(0.14/7) ~= 1.1 deg full opening angle for cone (if all light starts at a point before collimator)
	# Or 1.1 deg half opening angle if light comes from full range of angles at the entrance
    theta_r0 = independent_variables[0]
    phi_r0 = independent_variables[1]
    theta_i = independent_variables[2]
    theta_i_scalar = theta_i
    
    #print(theta_i,theta_r0,phi_r0)
    if sigma_theta_i > 0: # Expand theta_i to multiple values, drawn from a distribution centered on theta_i with width +/- sigma_theta_i
        n_rand = 20 # number of random values to draw
        #theta_i_prime = np.repeat(theta_i, n_rand)
        #delta_theta_i = np.random.normal(scale=sigma_theta_i, size=n_rand)
        eps=0.0001
        delta_theta_i = np.linspace(-3 * sigma_theta_i-eps,sigma_theta_i * 3+eps,n_rand) # offset by eps to avoid errors related to floating point precision
        #print(delta_theta_i)
        theta_i = theta_i+delta_theta_i
    


    # Make a grid of points in theta_r and phi_r
    #print(average_angle)
    theta_r_list = np.linspace(theta_r0-average_angle/2.,theta_r0+average_angle/2.,average_angle/precision+1)
    #print(theta_r_list)
    phi_r_list = np.linspace(phi_r0-average_angle/2.,phi_r0+average_angle/2.,average_angle/precision+1)
    if np.size(theta_i)>1: # If smearing theta_i, make the grid 3D
        tt, pp, ti = np.meshgrid(theta_r_list, phi_r_list, theta_i)
    else:
        tt, pp = np.meshgrid(theta_r_list, phi_r_list)
    tt=tt.flatten()
    pp=pp.flatten()
    # Include only points w/in circle
    in_circle=(tt-theta_r0)**2+(pp-phi_r0)**2 <= (average_angle/2)**2
    tt = tt[in_circle]
    pp = pp[in_circle]
    if np.size(theta_i)>1:
        ti=ti.flatten()
        theta_i=ti[in_circle]
        #print(theta_i[:25])
        weights = get_relative_gaussian_weights(theta_i, theta_i_scalar, sigma_theta_i)
    else:
        weights = [1.0]*len(tt)
    #print(tt[:25],pp[:25])
    return np.sum(weights * unvectorized_fitter([tt,pp,theta_i]+list(independent_variables[3:]),log_rho_L, log_n_minus_one, log_gamma, log_K, precision=precision)) / np.sum(weights)

    
# takes arrays of independent variable lists; last two independent variables are precision and average_angle (assumed to be the same for all points)
def fitter(independent_variables_array, log_rho_L, log_n_minus_one, log_gamma, log_K=None):
    average_angle = independent_variables_array[0][-1]
    precision = independent_variables_array[0][-2]
    sigma_theta_i = independent_variables_array[0][-3]
    # Could be fitting many runs, w/ different params like incident angle or medium index

    if precision <= 0: 
        arr = []
        for independent_variables in independent_variables_array:
            arr.append(unvectorized_fitter(independent_variables, log_rho_L, log_n_minus_one, log_gamma, log_K, precision=precision))
        if average_angle > 0: # Just average adjacent points in plane
            theta_r_in_degrees_array = [ind_vars[0] for ind_vars in independent_variables_array]
            # need to have theta_r in degrees! ind. vars is a list, each entry of which contains a list including theta_r (first)
            return average_by_angle(theta_r_in_degrees_array, arr, average_angle)
        else: # No averaging
            return arr
    else: # Do a full 2D average across both theta_r and phi_r, with grid spacing given by precision
        # t1=time.time()
        # print("Average by angle time: {0}".format(t1-t0))
        # t2=time.time()
        avgs = []
        for independent_variables in independent_variables_array: # For each point, calculate the BRIDF along a grid nearby and average it
            avgs.append(average_BRIDF(independent_variables, log_rho_L, log_n_minus_one, log_gamma, log_K, average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
        # t3=time.time()
        # print("Average BRIDF time: {0}".format(t3-t2))
        return avgs

def fitter_with_angle(independent_variables_without_angle_array, theta_i, log_rho_L, log_n_minus_one, log_gamma, log_K=None, precision=-1):
    arr = []
    for independent_variables_without_angle in independent_variables_without_angle_array:
        fit_val = unvectorized_fitter_with_angle(independent_variables_without_angle, theta_i, log_rho_L, log_n_minus_one, log_gamma, log_K, precision=precision)
        arr.append(fit_val)
    return arr

# independent variables array has as elements lists of independent variables for each point
# at each point, it has the form [theta_r_in_degrees, phi_r_in_degrees, theta_i_in_degrees, n_0, polarization]
# average_angle sets the angle in degrees over which to average the BRIDF for fitting to the data
# if precision is positive, the averaging is a full 2D average over theta_r, phi_r with grid size given by precision; else just a 1D average in plane
# if sigma_theta_i is positive and doing a 2D average in theta_r, phi_r, sets Gaussian sigma for smearing in incident angle
# use_errs sets whether to use the run's relative_std values to weight the fitting by or to use uniform errors
# use_spike sets whether to include the specular spike in the BRIDF model
def fit_parameters(independent_variables_array_intensity_array, p0=[0.5, 1.5, 0.05], average_angle=0, precision=-1, sigma_theta_i=-1, use_errs=True, use_spike=False, bounds=([-np.inf],[np.inf])):
    independent_variables_array = independent_variables_array_intensity_array[0]
    independent_variables_array = [list+[sigma_theta_i,precision,average_angle] for list in independent_variables_array]
    #print(independent_variables_array)
    intensity_array = independent_variables_array_intensity_array[1]
    if use_errs: std_array = independent_variables_array_intensity_array[2]
    else: std_array = None
    p0_log = [np.log(p0[0]), np.log(p0[1]-1), np.log(p0[2])]
    if use_spike:
        if len(p0)>3: p0_log.append(np.log(p0[3]))
        else: p0_log.append(np.log(2.0)) # default K value
    # initial parameters are the ones found in the paper
	# Set parameter bounds, transformed to log space; if lower or upper bounds are length <3, use default unbounded in that direction
    bounds_min=bounds[0]
    bounds_max=bounds[1]
    if len(bounds_min)>2:
        bounds_min_log=[np.log(bounds_min[0]), np.log(bounds_min[1]-1), np.log(bounds_min[2])]
        if len(bounds_min)>3 and use_spike: bounds_min_log.append(np.log(bounds_min[3]))
    else:
        bounds_min_log=-np.inf
    if len(bounds_max)>2:
        bounds_max_log=[np.log(bounds_max[0]), np.log(bounds_max[1]-1), np.log(bounds_max[2])]
        if len(bounds_max)>3 and use_spike: bounds_max_log.append(np.log(bounds_max[3]))
    else:
        bounds_max_log=np.inf
    bounds_log=(bounds_min_log, bounds_max_log)
    # fitter_p0 = fitter(independent_variables_array, p0_log[0], p0_log[1], p0_log[2])
    # res_fit_p0 = np.sum((np.array(intensity_array) - np.array(fitter_p0))**2)
    fit_results = scipy.optimize.curve_fit(fitter, np.array(independent_variables_array), np.array(intensity_array),
                                          p0=p0_log, sigma=std_array, absolute_sigma=True, bounds=bounds_log, ftol=1e-7, xtol=1e-7)
    fit_params_log = fit_results[0]
    cov_matrix = fit_results[1]
    #print("Covariance matrix (log): \n", cov_matrix)
    errs = np.sqrt(np.diag(cov_matrix))
    print("Errors (log): ",errs)
    errs_inv = 1./errs
    errs_inv_mat = np.diag(errs_inv)
    corr_mat = np.matmul(np.matmul(errs_inv_mat,cov_matrix),errs_inv_mat)
    print("Correlation matrix (log): \n",corr_mat)
    
    fitter_popt = fitter(independent_variables_array, *fit_params_log)
    if std_array is not None:
        chi2_fit_popt = np.sum(((np.array(intensity_array) - np.array(fitter_popt))/std_array)**2)/(len(intensity_array)-len(fit_params_log))
        print("Fit reduced chi^2: ",chi2_fit_popt)
    # tmp = (np.array(intensity_array) - np.array(fitter_popt))/std_array
    # top_diffs = np.argpartition(tmp**2,-10)[-10:]
    # print([(independent_variables_array[ind][0],independent_variables_array[ind][2]) for ind in top_diffs])
    # print([tmp[ind]**2 for ind in top_diffs])
    
    min_rho = np.exp(fit_params_log[0]-errs[0])
    max_rho = np.exp(fit_params_log[0]+errs[0])
    min_n = np.exp(fit_params_log[1]-errs[1])+1
    max_n = np.exp(fit_params_log[1]+errs[1])+1
    min_gamma = np.exp(fit_params_log[2]-errs[2])
    max_gamma = np.exp(fit_params_log[2]+errs[2])
    param_ranges = [[min_rho,max_rho],[min_n,max_n],[min_gamma,max_gamma]]
    if len(fit_params_log)>3: param_ranges.append([np.exp(fit_params_log[3]-errs[3]),np.exp(fit_params_log[3]+errs[3])])
    print("Min and max values: \n",param_ranges)
    
    params = [np.exp(fit_params_log[0]), np.exp(fit_params_log)[1] + 1, np.exp(fit_params_log[2])]
    if len(fit_params_log)>3: params.append(np.exp(fit_params_log[3]))
    return params


def change_theta_i(vars_intensity_array, new_theta_i):
    new_points = copy.deepcopy(vars_intensity_array)
    for point in new_points[0]:
        point[2] = new_theta_i
    return new_points


# returns [fitted_theta_i, rho_L, n, gamma]
# not intended for running data with more than one incident angle at a time
def fit_parameters_and_angle(independent_variables_array_intensity_array, p0=[0.5, 1.5, 0.05], average_angle=0, precision=-1, sigma_theta_i=-1, use_errs=True, use_spike=False):
    independent_variables_array = independent_variables_array_intensity_array[0]
    independent_variables_array = [list+[sigma_theta_i, precision, average_angle] for list in independent_variables_array]
    intensities = independent_variables_array_intensity_array[1]
    
    # determine if there is a sharp peak
    max_ind = intensities.index(max(intensities))
    theta_r_peak = independent_variables_array[max_ind][0]
    theta_i_peak = independent_variables_array[max_ind][2]
    # peak is within 15 degrees of where it's expected
    if np.abs(theta_r_peak - theta_i_peak) < 15:
        # use peak as theta_i
        points_1 = change_theta_i(independent_variables_array_intensity_array, theta_r_peak)
        parameters_with_theta_i_peak = fit_parameters(points_1,p0,average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=use_errs, use_spike=use_spike)
        theta_r_in_degrees_array = [point[0] for point in independent_variables_array]
        point0 = points_1[0][0]
        fit_array = BRIDF_plotter(theta_r_in_degrees_array, *point0[1:], parameters_with_theta_i_peak)
        fit_peak_index = fit_array.index(max(fit_array))
        fit_peak = independent_variables_array[fit_peak_index][0]
        peak_offset = fit_peak - theta_r_peak
        points_2 = change_theta_i(independent_variables_array_intensity_array, theta_r_peak - peak_offset)
        return [theta_r_peak - peak_offset] + fit_parameters(points_2,p0,average_angle, use_errs=use_errs, use_spike=use_spike)

    else:
        # use experimental as theta_i
        return [theta_i_peak] + fit_parameters(independent_variables_array_intensity_array,p0,average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_spike=use_spike)


# returns [fitted_theta_i, rho_L, n, gamma]
# not intended for running data with more than one incident angle at a time
# Note: not yet updated for newer data format (Run)
def fit_parameters_and_angle_with_starting_theta_i(points, starting_theta_i, average_angle=0):
    independent_variables_without_angle_array = []
    intensity_array = []
    for point in points:
        independent_variables_without_angle_array.append([point.theta_r_in_degrees, point.phi_r_in_degrees,
                                            point.n_0, point.polarization, average_angle])
        intensity_array.append(point.intensity)
    # initial parameters are the ones found in the paper
    fit_params = scipy.optimize.curve_fit(fitter_with_angle, independent_variables_without_angle_array, intensity_array,
                                          p0=[starting_theta_i, np.log(0.5), np.log(1.5 - 1), np.log(0.05)])[0]
    return [fit_params[0], np.exp(fit_params[1]), np.exp(fit_params)[2] + 1, np.exp(fit_params[3])]

# Calculates hemispherical reflectance for a given incident angle
# Note that specular lobe depends on whether specular spike is included or not
# Warning! 2D integration of specular component seems to fail for models in LXe
# Use reflectance_specular separately in that case
def reflectance(theta_i_in_degrees, n_0, polarization, parameters):

    theta_i = theta_i_in_degrees * np.pi / 180

    a = lambda x: 0
    b = lambda x: np.pi / 2

    def BRIDF_int(theta_r, phi_r):
        # solid angle not used here
        G = G_calc(theta_r, phi_r, theta_i, n_0, polarization, parameters)
        # precision=-1 ensures that specular spike isn't included, though it may modify specular lobe normalization if parameters include K
        # will add specular spike by hand, since integration won't have a fixed grid, so ensuring a real delta function is difficult
        # sin(theta_r) is from solid angle, G is to correct for the fact that shadowed/masked light ends up reflected, see eq. 16 of Silva 2009 Xe scintillation
		# Ideally, P is normalized with G in the integral, but none of the models we use do this 
        return BRIDF(theta_r, phi_r, theta_i, n_0, polarization, parameters, precision=-1) * np.sin(theta_r) / G 

    spec_spike = 0
    if len(parameters)>3:
        n=parameters[1]
        K=parameters[3]
        C = specular_spike_norm(theta_i, theta_i, K)
        F_ = F(theta_i, n_0, n, polarization)
        spec_spike = C * F_
        
    return scipy.integrate.dblquad(BRIDF_int, -np.pi, np.pi, a, b)[0] + spec_spike

# Calculates diffuse hemispherical reflectance for a given incident angle
def reflectance_diffuse(theta_i_in_degrees, n_0, polarization, parameters):

    theta_i = theta_i_in_degrees * np.pi / 180

    a = lambda x: 0 # Limits for theta_r
    b = lambda x: np.pi / 2
    if n_0 > parameters[1]:
        th_crit = lambda x: np.arcsin(parameters[1]/n_0)

    def BRIDF_int(theta_r, phi_r):
        # solid angle not used here
        G = G_calc(theta_r, phi_r, theta_i, n_0, polarization, parameters)
        # sin(theta_r) is from solid angle, G is to correct for the fact that shadowed/masked light ends up reflected, see eq. 16 of Silva 2009 Xe scintillation
		# Ideally, P is normalized with G in the integral, but none of the models we use do this 
        return BRIDF_diffuse(theta_r, phi_r, theta_i, n_0, polarization, parameters) * np.sin(theta_r) / G

    #theta_r_list = np.linspace(0.,np.pi/2,100)
    #BRIDF_list = [BRIDF_diffuse(theta_r, 0., theta_i, n_0, polarization, parameters, precision=-1) * np.sin(theta_r) / G_calc(theta_r, 0., theta_i, n_0, polarization, parameters) for theta_r in theta_r_list]
    #G_list = [G_calc(theta_r, 0., theta_i, n_0, polarization, parameters) for theta_r in theta_r_list]
    # plt.plot(theta_r_list,BRIDF_list )
    # plt.plot(theta_r_list,G_list )
    # plt.show()
    if n_0 > parameters[1]: # If we have total internal reflection, break up integral into part below and above critical angle
        # int_1d=scipy.integrate.quad(BRIDF_int,0,np.pi/2,0,full_output=1,epsrel=1e-5,epsabs=1e-5,limit=10)[0]*2*np.pi
        # print("Integral based off of phi_r=0: ",int_1d)
        integral_results_below = scipy.integrate.dblquad(BRIDF_int, -np.pi, np.pi, a, th_crit)
        #integral_results_above = scipy.integrate.dblquad(BRIDF_int, -np.pi, np.pi, th_crit_plus, b) #dblquad doesn't like W_crit
        phi_r_list = np.linspace(-np.pi,np.pi,30) # Not a strong function of phi_r, so few points are needed
        d_phi = np.pi*2/30
        one_d_int_results_above = [scipy.integrate.quad(BRIDF_int,np.arcsin(parameters[1]/n_0),np.pi/2,phi,full_output=1,epsrel=1e-5,epsabs=1e-5,limit=10) for phi in phi_r_list]
        one_d_ints_above = [int_result[0] for int_result in one_d_int_results_above]
        #print("Integrals for each slice in phi_r: ", one_d_ints_above)
        # print("Number of function evaluations in first integral: ",one_d_int_results_above[0][2]['neval'])
        # print("Number of subintervals: ",one_d_int_results_above[0][2]['last'])
        sum_integral_above = sum(one_d_ints_above)*d_phi
        integral = integral_results_below[0] + sum_integral_above# + integral_results_above[0]
    else:
        integral_results = scipy.integrate.dblquad(BRIDF_int, -np.pi, np.pi, a, b)
        integral = integral_results[0]
    #print("2D integral result: ",integral_results[0])
    #print("2D integral error: ",integral_results[1])
    return integral

# Calculates diffuse hemispherical reflectance for a given incident angle
# Note that specular lobe depends on whether specular spike is included or not
# Warning! 2D integration of specular component seems to fail for models in LXe
# Uses sum of 1D integrals in that case
def reflectance_specular(theta_i_in_degrees, n_0, polarization, parameters):

    theta_i = theta_i_in_degrees * np.pi / 180

    a = lambda x: 0
    b = lambda x: np.pi / 2

    def BRIDF_int(theta_r, phi_r):
        G = G_calc(theta_r, phi_r, theta_i, n_0, polarization, parameters)
        # precision=-1 ensures that specular spike isn't included, though it may modify specular lobe normalization if parameters include K
        # will add specular spike by hand, since integration won't have a fixed grid, so ensuring a real delta function is difficult
        # sin(theta_r) is from solid angle, G is to correct for the fact that shadowed/masked light ends up reflected, see eq. 16 of Silva 2009 Xe scintillation
		# Ideally, P is normalized with G in the integral, but none of the models we use do this 
        return BRIDF_specular(theta_r, phi_r, theta_i, n_0, polarization, parameters, precision=-1) * np.sin(theta_r) / G 

    spec_spike = 0
    if len(parameters)>3:
        n=parameters[1]
        K=parameters[3]
        C = specular_spike_norm(theta_i, theta_i, K)
        F_ = F(theta_i, n_0, n, polarization)
        spec_spike = C * F_

    if n_0>parameters[1]: # If medium index is higher than sample, use 1D integrals instead
        phi_r_list = np.linspace(-np.pi,np.pi,500)
        d_phi = np.pi*2/500
        # theta_r_list = np.linspace(0.,np.pi/2,100)
        # BRIDF_list = [BRIDF_specular(theta_r, 0., theta_i, n_0, polarization, parameters, precision=-1) * np.sin(theta_r) / G_calc(theta_r, 0., theta_i, n_0, polarization, parameters) for theta_r in theta_r_list]
        # G_list = [G_calc(theta_r, 0., theta_i, n_0, polarization, parameters) for theta_r in theta_r_list]
        # plt.plot(theta_r_list,BRIDF_list )
        # plt.plot(theta_r_list,G_list )
        # plt.show()
        one_d_int_results = [scipy.integrate.quad(BRIDF_int,0.,np.pi/2,phi,full_output=1) for phi in phi_r_list]
        one_d_ints = [int_result[0] for int_result in one_d_int_results]
        #print("Integrals for each slice in phi_r: ", one_d_ints)
        sum_integral = sum(one_d_ints)*d_phi
        # print("Summed integrals, times d_phi: ",sum_integral)
        # print("Summed integral error, times d_phi: ",sum([int_result[1] for int_result in one_d_int_results])*d_phi)
        return sum_integral + spec_spike
    else:
        integral_results = scipy.integrate.dblquad(BRIDF_int, -np.pi, np.pi, a, b)
        #print("2D integral error: ",integral_results[1])
        #print("2D integral result: ",integral_results[0])
        #print("Integral evaluation points: ", integral_results[2]['alist'])
        return integral_results[0] + spec_spike
    

# uses 3d grid of parameters
# independent variables array has as elements lists of independent variables for each point
# at each point, it has the form [theta_r_in_degrees, phi_r_in_degrees, theta_i_in_degrees, n_0, polarization]
# also included in first argument are the intensity values and their errors
def fit_parameters_grid(independent_variables_array_intensity_array, rho_start, rho_end, rho_num, n_start, n_end, n_num, gamma_start, gamma_end, gamma_num, average_angle=0, precision=-1, sigma_theta_i=-1, plot=True, show=True):
    independent_variables_array = independent_variables_array_intensity_array[0]
    #print(independent_variables_array)
    intensity_array = independent_variables_array_intensity_array[1]
    std_array = independent_variables_array_intensity_array[2]
    
    rho_array = np.linspace(rho_start, rho_end, rho_num)
    n_array = np.linspace(n_start, n_end, n_num)
    gamma_array = np.linspace(gamma_start, gamma_end, gamma_num)

    grid_points = []
    for rho in rho_array:
        for n in n_array:
            for gamma in gamma_array:
                grid_points.append([rho, n, gamma])

    length = len(grid_points)

    chi2 = []
    for i in range(len(grid_points)):
        grid_point = grid_points[i]
        chi2.append(chi_squared(independent_variables_array, intensity_array, std_array, grid_point, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
        if i%10 == 0:
            print(str(i) + "/" + str(length))

    min_index = np.argmin(chi2)
    min_params = grid_points[min_index]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        p0 = []
        p1 = []
        p2 = []
        color = []
        for i in range(len(grid_points)):
            p0.append(grid_points[i][0])
            p1.append(grid_points[i][1])
            p2.append(grid_points[i][2])
            color.append(chi2[i])

        a = ax.scatter(p0, p1, p2, c=color, norm=matplotlib.colors.LogNorm(), s=15)

        fig.colorbar(a)

        ax.text2D(0., 0.95, "optimal parameters: rho_L = " + str(round(min_params[0], 5)) + ", n = " + str(round(min_params[1], 5)) + ", gamma = " + str(round(min_params[2], 5)), transform=ax.transAxes)

        ax.set_xlabel('rho_L')
        ax.set_ylabel('n')
        ax.set_zlabel('gamma')

        if show:
            plt.show()

    return grid_points[min_index]

# Returns chi^2/degrees of freedom for the given data set and fit parameters
def chi_squared(independent_variables_array, intensity_array, std_array, parameters, average_angle=0, precision=-1, sigma_theta_i=-1):
    params_log = [np.log(parameters[0]), np.log(parameters[1]-1), np.log(parameters[2])]
    if len(parameters)>3: 
        if parameters[3]>0: params_log.append(np.log(parameters[3]))
    
    independent_variables_array = [list+[sigma_theta_i,precision,average_angle] for list in independent_variables_array]
    
    fitter_popt = fitter(independent_variables_array, *params_log)
    chi2 = np.sum(((np.array(intensity_array) - np.array(fitter_popt))/std_array)**2)/(len(intensity_array)-len(params_log))
    #print("Fit reduced chi^2: ",chi2_fit_popt)

    return chi2






