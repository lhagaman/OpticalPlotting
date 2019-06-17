import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
from file_reader import Run, get_independent_variables_and_relative_intensities
from plotting import plot_runs, plot_TSTR_fit
from TSTR_fit_new import fit_parameters, fit_parameters_and_angle, fit_parameters_grid, fitter, BRIDF_plotter, reflectance_diffuse, reflectance_specular, BRIDF, chi_squared
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

# Setting default color map:
# matplotlib.rcParams['image.cmap'] = 'viridis' # For matplotlib's viridis colormap; many others available
# More matplotlib color maps, including discussion on perceptual uniformity and conversion to b&w: https://matplotlib.org/users/colormaps.html
# More info on color maps:
# Colorcet http://colorcet.pyviz.org/index.html - e.g. import colorcet as cc; cc.fire
# Colorcet ensures perceptual uniformity
# Seaborn https://seaborn.pydata.org/tutorial/color_palettes.html
# Seaborn has many features
# cubehelix seems like a good choice for this application
# Run interactive palette choosing script in ipython:
# 	import seaborn
# 	seaborn.choose_cubehelix_palette()
# Choosing color schemes (more geared toward UI development): https://material.io/design/color/#tools-for-picking-colors
# Note that perceptual uniformity isn't important for discrete color sets (only real maps where value->color)
# For plots, just care that b&w values are distinguishable and preferably monotonic
# gist_earth even stopping at 0.9 gets too light at the end; same for plasma, inferno (stopping at 1.0)
# tab20b isn't monotonic in grayscale
# viridis is good

import seaborn as sns
cmap=sns.cubehelix_palette(as_cmap=True,n_colors=6,gamma=0.7,hue=1.0,light=0.7,dark=0.15)

# vacuum data
path_vac = "Vacuum measurements after 3rd xenon run\\Jan 9-12\\LZ Skived\\Blue height\\178nm\\"
### After 3rd run LZ Skived 178nm Vacuum measurements after 3rd xenon run\\Jan 9-12\\LZ Skived\\Blue height\\178nm\\

mirrors = [[30,"Vacuum measurements after 3rd xenon run\\Jan 9-12\\Mirror alignment\\Blue height 2\\2019_01_10__16_03_43.txt"],
           [45,"Vacuum measurements after 3rd xenon run\\Jan 9-12\\Mirror alignment\\Blue height 2\\2019_01_10__16_01_50.txt"],
           [52,"Vacuum measurements after 3rd xenon run\\Jan 9-12\\Mirror alignment\\Blue height 2\\2019_01_10__16_00_07.txt"],
           [60,"Vacuum measurements after 3rd xenon run\\Jan 9-12\\Mirror alignment\\Blue height 2\\2019_01_10__15_58_07.txt"],
           [67,"Vacuum measurements after 3rd xenon run\\Jan 9-12\\Mirror alignment\\Blue height 2\\2019_01_10__15_56_24.txt"],
           [75,"Vacuum measurements after 3rd xenon run\\Jan 9-12\\Mirror alignment\\Blue height 2\\2019_01_10__15_53_05.txt"]]

skived30 = Run(path_vac + "2019_01_10__10_52_19.txt", mirrors)
skived45 = Run(path_vac + "2019_01_10__10_45_18.txt", mirrors)
skived52 = Run(path_vac + "2019_01_10__10_39_26.txt", mirrors)
skived60 = Run(path_vac + "2019_01_10__10_34_19.txt", mirrors)
skived67 = Run(path_vac + "2019_01_10__10_29_11.txt", mirrors)
skived75 = Run(path_vac + "2019_01_10__10_23_22.txt", mirrors)
angle_shifts_s9_vac=[28,45,52,60,67,75]

# Second LXe run
path = "2nd Xenon Run Measurements\\"

s9_lowp_30 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_52_04.txt")
s9_lowp_45 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_47_09.txt")
s9_lowp_52 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_42_21.txt")
s9_lowp_60 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_37_31.txt")
s9_lowp_67 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_32_47.txt")
s9_lowp_75 = Run(path + "Sample 9 lower pressure\\2018_12_05__13_27_55.txt")
angle_shifts_s9_lowp=[29, 47, 53, 59, 67, 75]

angle_shifts_none = [30, 45, 52, 60, 67, 75]

# For LXe data:
# Params to change in file_reader.py: const_err=100, frac_err=0.05, intensity_correction=0.753
# Used scaled bkg from 1st LXe run, no dark bkg file loaded in:
# In file_reader.py, used self.dark_bkg_intensities=60, bkg_scaling=1.5
# beam_bkg_filename = "First Xe Run Measurements\\first measurements with no bubbles in cell 11-01-2\\Initial power and background at 178 nm\\2018_11_01__14_56_35.txt"
# bkg = (self.beam_bkg_intensities-self.dark_bkg_intensities)*(self.incidentpower/self.beam_bkg_incidentpower)*bkg_scaling +self.dark_bkg_intensities
# Used sigma_theta_i=-1 for fitting and wavelength_F = True in TSTR_fit_new.py
# Make sure n=n_LXe for plotting below
# Angle range in file_reader stops at 80, same for plot_TSTR_fit
runs = [s9_lowp_30,s9_lowp_45,s9_lowp_52,s9_lowp_60,s9_lowp_67,s9_lowp_75]
angle_shifts=angle_shifts_s9_lowp

# For vacuum data:
# Params to change in file_reader.py: const_err=40, frac_err=0.05, intensity_correction=0.93133
# Just used dark bkg as full bkg to subtract:
# In file_reader.py, used self.dark_bkg_intensities from file
# dark_bkg_filename = "Vacuum measurements after 3rd xenon run\\Jan 9-12\\Background\\No beam\\2019_01_11__17_18_44.txt"
# bkg = self.dark_bkg_intensities
# self.beam_bkg_angles=self.dark_bkg_angles
# Used sigma_theta_i=2 for fitting and wavelength_F = False in TSTR_fit_new.py
# Angle range in file_reader is not cut off, stops at 85 in plot_TSTR_fit
runs = [skived30, skived45, skived52, skived60, skived67, skived75]
angle_shifts=angle_shifts_s9_vac
# Make sure n=1.0 for plotting below

for run, angle in zip(runs, angle_shifts): run.change_theta_i(angle)
labels=[r"$\theta_i=30^{\circ}$","45$^{\circ}$","52$^{\circ}$","60$^{\circ}$", "67$^{\circ}$", "75$^{\circ}$"]#,"30 degrees","45 degrees","52 degrees","60 degrees", "67 degrees", "75 degrees"]

# Plot BRIDF data
sample_name="M17 Skived"
fig, ax = plt.subplots()
plot_runs(runs, title="", log=True, labels=False, label=False, include_legend=False, errorbars=True, legend_loc=0, figure=False, colormap=True)
# plot_runs(runs, title=sample_name+" in 0.2 barg LXe, 178 nm, 75 deg", log=False, labels=False, include_legend=False, errorbars=True, legend_loc=0)
t0=time.time()

# Fit data
average_angle=4
precision=0.25
sigma_theta_i=2

# Fit parameters using power correction, TR dist, 2.00 solid angle factor, Gaussian distribution in n
fit_params_s9_lowp_gauss_n= [0.7617591236933063, 1.581667834504223, 0.10965539347694214, 1.2535811658618297]
fit_params_s9_vac= [0.73367974544643089, 1.7043657376479024, 0.11826240150179418, 7.9564857778025644]

fit_params = fit_params_s9_vac


print("Fit parameters (rho_L, n, gamma): "+str(fit_params))
#print("Fit angle: "+str(fit_ang))
t1=time.time()
print("Fitting time: {0}".format(t1-t0))


# Plot BRIDF model from fits
n_LXe_178 = 1.69
n_LXe_220 = 1.5044552
n_LXe_300 = 1.42975267
n_LXe_400 = 1.404459446
n_LXe = n_LXe_178
n=1.0
# n=n_LXe

plt.xlim(0,85)
plt.ylim(1e-3,1e2)
plt.yscale("log")

#colors=["r","g","b","m","c","y"]
colors=[plt.cm.plasma(i) for i in np.linspace(0.2,0.90,len(runs))] 
# colors=[cmap(i) for i in np.linspace(0,1,len(runs))] 
label_x=[0.63,0.93,0.88,0.85,0.9,0.88] # Positions for s9 low_p
label_y=[0.27,0.31,0.45,0.63,0.75,0.88]
label_x=[0.25,0.45,0.55,0.65,0.75,0.85] # Positions for s9 in vacuum
label_y=[0.6,0.62,0.65,0.7,0.76,0.85]
angle_labels=[r"$\theta_i=30^{\circ}$",r"$45^{\circ}$",r"$52^{\circ}$",r"$60^{\circ}$",r"$67^{\circ}$",r"$75^{\circ}$"]
for ii in range(len(angle_shifts)):
	plot_TSTR_fit(angle_shifts[ii], n, fit_params, color=colors[ii], label=None, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
	plt.text(label_x[ii], label_y[ii], angle_labels[ii],transform=plt.gca().transAxes,fontsize=13,color=colors[ii])
	

# Set x-axis tick marks to every 10 deg
loc = matplotlib.ticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)

# Add grid lines
plt.grid(b=True,which='major',color="lightgray",linestyle='--')
# plt.grid(b=True,which='major',axis='x',color="lightgray",linestyle=':')
# plt.grid(b=True,which='both',axis='y',color="lightgray",linestyle=':')

#plt.tight_layout()

t2=time.time()
print("Plotting time: {0}".format(t2-t1))

run_data=get_independent_variables_and_relative_intensities(runs)
chi_sq=chi_squared(run_data[0], run_data[1], run_data[2], fit_params, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
print("Chi squared from fit: ",chi_sq)

# Now calculate hemispherical reflectance
# plt.figure()

# # Different incident angles to calculate for
# x = [0,10,20,30, 45.1, 55, 60, 65, 70, 75, 80, 85]#[0,30,45.1,60,70,75,80,85]#[5,60,70,80]#[5,30, 45, 60, 70, 75, 80, 85]#[0,10,20,30, 45, 55, 60, 65, 70, 75, 80, 85]#

# y_diffuse = [reflectance_diffuse(theta, n_LXe, 0.5, fit_params) for theta in x]
# y_specular = [reflectance_specular(theta, n_LXe, 0.5, fit_params) for theta in x]
# y_total = [y_diffuse[i] + y_specular[i] for i in range(len(y_specular))]

# print("Diffuse reflectances: ",y_diffuse)
# print("Specular reflectances: ",y_specular)
# print("Total reflectances: ",y_total)


# plt.plot(x, y_diffuse, label="diffuse")
# plt.plot(x, y_specular, label="specular")
# plt.plot(x, y_total, label="total")
# plt.plot(x, y_specular_vacuum_after, label="specular, vacuum",linestyle='-',color='y')
# plt.plot(x, y_specular_LXe_lowp, label="specular, 0.2 barg LXe",linestyle='-.',color='y')
# # #plt.plot(x, y_specular_LXe_lowp2, label="spec_lowp2")
# # plt.plot(x, y_specular_LXe_hip, label="specular, 1.45 barg LXe",linestyle='--',color='y')
# # #plt.plot(x, y_specular_LXe_hip2, label="spec_hip2")
# plt.plot(x, y_diffuse_vacuum_after, label="diffuse, vacuum",linestyle='-',color='c')
# plt.plot(x, y_diffuse_LXe_lowp, label="diffuse, 0.2 barg LXe",linestyle='-.',color='c')
# # #plt.plot(x, y_diffuse_LXe_lowp2, label="diff_lowp2")
# # # plt.plot(x, y_diffuse_LXe_hip, label="diffuse, 1.45 barg LXe",linestyle='--',color='c')
# # #plt.plot(x, y_diffuse_LXe_hip2, label="diff_hip2")
# plt.plot(x, y_total_vacuum_after, label="total, vacuum",linestyle='-',color='b')
# plt.plot(x, y_total_LXe_lowp, label="total, 0.2 barg LXe",linestyle='-.',color='b')
# # #plt.plot(x, y_total_LXe_lowp2, label="total_lowp2")
# # # plt.plot(x, y_total_LXe_hip, label="total, 1.45 barg LXe",linestyle='--',color='b')
# # #plt.plot(x, y_total_LXe_hip2, label="total_hip2")
# # # # Line styles: '-', '--', '-.', ':'

# plt.xlabel("incident angle (degrees)")
# plt.ylabel("reflectance (fraction)")
# plt.legend()
# plt.ylim(0,1.1)

# plt.title("Fitted "+sample_name+" Reflectance, 178 nm")
t3=time.time()
print("Reflectance calc time: {0}".format(t3-t2))
plt.show()

