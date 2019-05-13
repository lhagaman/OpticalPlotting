import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
from file_reader import Run, get_independent_variables_and_relative_intensities
from plotting import plot_runs, plot_TSTR_fit
from TSTR_fit_new import fit_parameters, fit_parameters_and_angle, fit_parameters_grid, fitter, BRIDF_plotter, reflectance_diffuse, reflectance_specular, BRIDF, BRIDF_specular, chi_squared, G_calc, F_wavelength_range
import time
from scipy.interpolate import interp1d
from scipy.integrate import quad

# Plot style settings
# 14 pt font seems to work for default figure size of [6.4, 4.8] (half column width figure)
# saving figure as *.eps makes figure less blurry when zoomed in, but lines become quite thick when zoomed out
matplotlib.rcParams['font.size'] = 14
#matplotlib.rcParams['figure.figsize'] = [8.0, 6.0] # Default is [6.4, 4.8]
#print(matplotlib.rcParams['figure.figsize'])
matplotlib.rcParams['legend.fontsize'] = 'small'
matplotlib.rcParams['figure.autolayout'] = True

# Fitter params for s9 in vacuum after LXe, 2.15 solid angle
fit_params_vacuum=[0.68114103391512837, 1.6259581862678401, 0.11282346636450034, 8.5642437795248174]
# Fitter parameters for s9 in LXe, 2.15 solid angle factor, Lorentz model
# fit_params_lowp=[0.9466040398353657, 1.5542353537306548, 0.09079476300181971]
# fit_params_hip=[0.9871481402574648, 1.5715034469520426, 0.09586940402324681]
# Fitter parameters for s9 in LXe, 2.00 solid angle factor, BC model
fit_params_lowp=[1.1077461844574992, 1.5547594445581758, 0.10054542669454647]
fit_params_lowp_nrange_0_12=[1.1036187692365016, 1.5556531315657196, 0.10075277940099392]
fit_params_hip2=[1.1999650882272368, 1.5436916548263535, 0.20783614159557665]
fit_params_hip=[1.1525410041990918, 1.5691617861938227, 0.11806736015270886, 11.114872654849078]
fit_params_s8=[1.30037846657779, 1.5481537596293447, 0.060365408007835346, 4.900748469692683]
fit_params_s9_400nm=[0.6916903132118614, 1.7685668238585814, 0.052664172718245646]
fit_params_s9_400nm_nrange_0_15=[0.6414663099143563, 2.0285416875437416, 0.2071409460435925]
# fit_params=[0.5884392151835587, 1.568625421661115, 0.13151331842089983, 2.591901372978062]
# fit_params=fit_params_lowp_nrange_0_12
# fit_params=[0.795, 1.586, 0.106]
# Fitter parameters for s9 in LXe, 2.00 soid angle, BC, power correction 0.753
fit_params_s9_lowp_corr_nrange_0_15=[0.8098340929026058, 1.5651721053377226, 0.1117946993003091]
# Fitter parameters for s9 in LXe, 2.00 soid angle, TR, power correction 0.753
fit_params_s9_lowp_corr_nrange_0_22_TR=[0.7954599045509312, 1.5856455123704216, 0.10588126510921678]
fit_params_s9_lowp_corr_F_sigmoid_nrange_0_22 = [0.7863406181428023, 1.575379194134467, 0.1159731109468702, 0.058296968852107584]
fit_params_s9_lowp_corr_F_sigmoid=[0.821156242689131, 1.5764822034015689, 0.11532489637813395, 0.05942653556417246]
# fit_params=fit_params_s9_lowp_corr_F_sigmoid_nrange_0_22
fit_params_s9_lowp2_corr_nrange_0_15=[0.9001033317655373, 1.5675280902284154, 0.11370369845973141]
# Fitter parameters for s8 in LXe, 2.00 soid angle, BC, power correction 0.753
fit_params_s8_lowp_corr_nrange_0_15=[0.9427456100375428, 1.5664201944697242, 0.08352285569818733, 3.9498103310586443]
fit_params_s8_lowp_corr_nrange_0_15_avg_5deg=[0.9402613669832427, 1.5664520623940574, 0.06270405048987444, 4.025893124659275]
# fit_params=fit_params_s8_lowp_corr_nrange_0_15_avg_5deg
# Fitter parameters for s9 in LXe, 2.00 solid angle, TR, power correction 0.753, with wavelength range for n_LXe_178
fit_params_s9_lowp_corr_n_LXe_wavelength=[0.7759585422536841, 1.5859038386290387, 0.1152682251245288, 5.607571663996526]
# fit_params=[0.7759585422536841, 1.5859038386290387, 0.057, 5.607571663996526] # like s9 but w/ a smoother surface; gamma from fit to molded, unpolished PTFE from Claudio's paper
# Fitter parameters for s9 in LXe, 3rd run, 2.00 solid angle factor, BC model
fit_params_r3_s9_lowp= [1.0216459850600514, 1.5467069707623793, 0.1101614104957136]
# fit_params=fit_params_r3_s9_lowp
# Fitter parameters for s2 in LXe, 3rd run, 2.00 solid angle factor, BC model
fit_params_r3_s2_hip=[1.0720924015263558, 1.5425690292388028, 0.20472829546890117] # error on n from fit is 1.542-1.543; correlation is 2% w/ rho, -48% w/ gamma (gamma and rho are -0.8%, i.e. very uncorrelated); not much difference in profile for 20 pts in rho, gamma vs 10
# Fit parameters using power correction, TR dist, 2.00 solid angle factor, Gaussian distribution in n
fit_params_s9_lowp_gauss_n= [0.7617591236933063, 1.581667834504223, 0.10965539347694214, 1.2535811658618297]
fit_params_s9_lowp2_gauss_n= [0.855702300814262, 1.5823928228219537, 0.1174943642524602, 1.375597480899577]
fit_params_s9_hip_gauss_n= [0.7951797306837846, 1.6003118297015986, 0.1096203149254032, 1.219900829304097]
fit_params_s9_hip2_gauss_n= [0.8226683156062747, 1.5974576635618738, 0.11905775599090765, 1.3981072921827302]
fit_params_s9_medp_gauss_n= [0.8090807492966514, 1.601324395321314, 0.11048727271620085, 1.4948435916434195]
fit_params_s9_medp2_gauss_n= [0.8314357200774386, 1.596290134519927, 0.1133753088729935, 1.390219480578202]
fit_params_s9_bubbles_gauss_n= [0.7969293740674858, 1.573971098600423, 0.12385855127165646, 1.4524567768049776]
fit_params_s9_nobubbles_gauss_n= [0.8296345869549955, 1.5755593111871824, 0.12215387161041028, 1.3373494975065916]
fit_params_s9_getter_gauss_n= [0.8318405791997575, 1.5810629481820473, 0.11767447597690964, 1.3769551555713786]
fit_params_s9_lowp_above_gauss_n= [0.8477739823593516, 1.5815242729575472, 0.11821105282357888, 1.3590210508898195]
fit_params_s9_165nm_gauss_n= [0.217183610197179, 1.9181849064913858, 0.1536796098999006, 5.850296346659101]
fit_params_s9_300nm_gauss_n=  [0.9447069486729521, 1.4374429468203447, 0.1070442097507485, 1.5680079162288003]
fit_params_s9_400nm_gauss_n= [0.6021432485069854, 1.5440956423913457, 0.13516753727577357, 2.6357777432157024]
fit_params_s5_lowp_gauss_n= [0.7676075426046298, 1.5751319649534214, 0.11329673174987098, 1.5458560015702658]
fit_params_s8_lowp_gauss_n= [0.9144536241966592, 1.5793399801859325, 0.08088290783956979, 1.2565135586082907, 4.857675748475832] # Uses both sigma_n and specular spike
fit_params_s1_r1_gauss_n= [0.6280995506491662, 1.57787534395258, 0.14910541148290482, 1.5083045922962428]
fit_params_s9_r1_gauss_n= [0.8279, 1.5623, 0.1202, 1.09]
fit_params_s3_r1_gauss_n= [0.765861613638844, 1.572270174034155, 0.16606130754530377, 1.444391660582311]
fit_params_s2_lowp_r3_gauss_n= [0.717, 1.545, 0.209, 1.62]
fit_params_s2_hip_r3_gauss_n= [0.793, 1.577, 0.196, 1.70]
fit_params_s2_lowp_above_r3_gauss_n= [0.771, 1.545, 0.203, 1.54]
fit_params_s6_lowp_r3_gauss_n= [0.744, 1.572, 0.184, 1.89]
fit_params_s6_hip_r3_gauss_n= [0.733, 1.596, 0.192, 2.02]
fit_params_s9_lowp_r3_gauss_n= [0.742, 1.568, 0.12, 1.34]
fit_params_s9_hip_r3_gauss_n= [0.748, 1.596, 0.12, 1.71]
fit_params=fit_params_s2_lowp_r3_gauss_n

print("Fit parameters (rho_L, n, gamma): "+str(fit_params))

# Plot BRIDF model from fits
n_LXe_178 = 1.69
n_LXe = n_LXe_178
sigma_theta_i=-1
precision=-1
average_angle=-1

# For testing: try calculating just specular lobe on a grid of angles (theta_r, phi_r) for fixed theta_i
# Try this for M17 lowp in run3, where integral at theta_i=85 > 1, and there is no specular spike
# If BRIDF were constant vs theta_r, phi_r, would get 100% reflectivity if equal to 1/(2*pi)
# If BRIDF were constant in projected solid angle (i.e. Lambertian), would get 100% reflectivity if max value is 1/pi

# ! Try just calculating F_, P/4*cos(th_i) over grid, doing integral
# Integral of BRIDF*sin(th_r)/G is 1.056 (similar to 1.07 seen from quad integral); only 0.50 when keeping G
# Integral of P/4*cos(th_i) over grid is 1.73(!) for 85 deg, gamma=0.209, n_th_r=100, n_phi_r=500;
# Integral is 1.20 for 80 deg, same other params 
# Integral is 1.06 for 75 deg, same other params
theta_i = 75*np.pi/180
n_th_r=100
d_th_r=np.pi/2/n_th_r
theta_r_list = np.linspace(0,np.pi/2,n_th_r)
n_phi_r=500
d_phi=2*np.pi/n_phi_r
phi_r_list = np.linspace(-np.pi,np.pi,n_phi_r) # Skipping high angles (all very close to 0)
tt, pp = np.meshgrid(theta_r_list, phi_r_list)
t_flat=tt.flatten()
p_flat=pp.flatten()
BRIDF_array = []
G_array = []
t0=time.time()
# for theta_r, phi_r in zip(t_flat,p_flat):
	# BRIDF_array.append(BRIDF_specular(theta_r, phi_r, theta_i, n_LXe, 0.5, fit_params, precision=-1))
	# G_array.append(G_calc(theta_r-1e-8, phi_r, theta_i, n_LXe, 0.5, fit_params))
# sum_BRIDF=np.sum(BRIDF_array)*d_th_r*d_phi
# sum_BRIDF_over_G=np.sum([b/g for (b,g) in zip(BRIDF_array,G_array)])*d_th_r*d_phi
# BRIDF_array = np.array(BRIDF_array).reshape(np.shape(tt)) # set shape
# for ii in range(len(G_array)):
    # if t_flat[ii]>75*np.pi/180:
        # print("theta_r: ",t_flat[ii],"phi_r: ",p_flat[ii],"G: ",G_array[ii])
# G_array = np.array(G_array).reshape(np.shape(tt)) 
# # print(G_array[tt>75*np.pi/180.])
# sum_sin_BRIDF = np.sum(np.sin(tt)*BRIDF_array)*d_th_r*d_phi
# sum_sin_BRIDF_over_G = np.sum(np.sin(tt)*BRIDF_array/G_array)*d_th_r*d_phi
# print("sum_BRIDF: ",sum_BRIDF)
# print("sum_BRIDF_over_G: ",sum_BRIDF_over_G)
# print("sum_sin_BRIDF: ",sum_sin_BRIDF)
# print("sum_sin_BRIDF_over_G: ",sum_sin_BRIDF_over_G)

gamma=fit_params[2]
def P(alpha_):
	return np.power(gamma, 2) / \
		(np.pi * np.power(np.cos(alpha_), 4) *
		np.power(np.power(gamma, 2) + np.power(np.tan(alpha_), 2), 2))

F_array=[]
P_array=[]
for theta_r, phi_r in zip(t_flat,p_flat):    
	theta_prime = 0.5 * np.arccos(np.cos(theta_i) * np.cos(theta_r) - np.sin(theta_i) * np.sin(theta_r) * np.cos(phi_r))
	theta_i_prime = theta_prime
	F_array.append( F_wavelength_range(theta_i_prime, fit_params[1], 0.5, sigma_lambda=fit_params[3]) )
	alpha_specular = np.arccos((np.cos(theta_i) + np.cos(theta_r)) / (2 * np.cos(theta_i_prime)))
	P_array.append( P(alpha_specular) )
	
P_array=np.array(P_array).reshape(np.shape(tt))
F_array=np.array(F_array).reshape(np.shape(tt))
sum_P_cos = np.sum(np.sin(tt)*P_array/(4*np.cos(theta_i)))*d_th_r*d_phi
sum_F_P_cos = np.sum(np.sin(tt)*F_array*P_array/(4*np.cos(theta_i)))*d_th_r*d_phi
print("sum_P_cos: ",sum_P_cos)
print("sum_F_P_cos: ",sum_F_P_cos)


# fig=plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# pp_shift = np.copy(pp) # Convert to range [-pi, pi]
# pp_shift[pp_shift>np.pi]=pp_shift[pp_shift>np.pi]-2*np.pi
# ax.plot_surface(tt*180/np.pi,pp*180/np.pi,BRIDF_array)

# fig1=plt.figure()
# plt.contour(tt*180/np.pi,pp*180/np.pi,BRIDF_array)

# fig2=plt.figure()
# c=plt.pcolormesh(tt*180/np.pi,pp*180/np.pi,BRIDF_array)
# fig2.colorbar(c)

t1=time.time()
print("BRIDF calc time: {0}".format(t1-t0))

# Different incident angles to calculate for
# x = [85]#[0,10,20,30, 45.1, 55, 60, 65, 70, 75, 80, 85]#[0,30,45.1,60,70,75,80,85]#[5,60,70,80]#[5,30, 45, 60, 70, 75, 80, 85]#[0,10,20,30, 45, 55, 60, 65, 70, 75, 80, 85]#

# y_diffuse = [reflectance_diffuse(theta, n_LXe, 0.5, fit_params) for theta in x]
# y_specular = [reflectance_specular(theta, n_LXe, 0.5, fit_params) for theta in x]
# y_total = [y_diffuse[i] + y_specular[i] for i in range(len(y_specular))]

# print("Diffuse reflectances: ",y_diffuse)
# print("Specular reflectances: ",y_specular)
# print("Total reflectances: ",y_total)

t2=time.time()
print("Hemispherical reflectance calc time: {0}".format(t2-t1))

plt.show()

# Integral w/ 500 points in phi_r (1D): 1.07200  
# Integral w/ 700 points in phi_r (1D): 1.07262
# 2D integral, broken up: -pi to -10 deg: 0.0091; 10 deg to pi: 0.0037; -10 deg to +10 deg: 0.8863

# Phi_r:  -0.006295776860901547
# Interval right endpoints (deg):  [54.140625  42.890625  77.34375    0.        65.390625  30.9375
 # 78.75      47.8125    70.6640625 56.953125  36.5625    25.3125
 # 50.625     73.125     61.875     40.78125   61.171875  33.75
 # 29.53125   45.703125  67.5       22.5       75.9375    66.09375
 # 57.65625   54.84375   52.03125   49.21875   59.0625    46.40625
 # 43.59375   39.375     37.96875   35.15625   32.34375   28.125
 # 71.71875   71.015625  78.046875  60.46875   56.25      53.4375
 # 51.328125  48.515625  45.        64.6875    42.1875    77.6953125
 # 65.7421875 70.3125   ]
# Integral * d_phi:  0.09115155895209258
# Phi_r:  0.006295776860901103
# Interval right endpoints (deg):  [54.140625  42.890625  77.34375    0.        65.390625  30.9375
 # 78.75      47.8125    70.6640625 56.953125  36.5625    25.3125
 # 50.625     73.125     61.875     40.78125   61.171875  33.75
 # 29.53125   45.703125  67.5       22.5       75.9375    66.09375
 # 57.65625   54.84375   52.03125   49.21875   59.0625    46.40625
 # 43.59375   39.375     37.96875   35.15625   32.34375   28.125
 # 71.71875   71.015625  78.046875  60.46875   56.25      53.4375
 # 51.328125  48.515625  45.        64.6875    42.1875    77.6953125
 # 65.7421875 70.3125   ]
# Integral * d_phi:  0.09115155895209258
