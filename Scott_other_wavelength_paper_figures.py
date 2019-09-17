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

# Second LXe run
path = "2nd Xenon Run Measurements\\"

# Second run, data at different wavelengths
s9_400nm_30 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__18_12_04.txt")
s9_400nm_45 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__18_06_45.txt")
s9_400nm_52 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__18_01_31.txt")
s9_400nm_60 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_55_29.txt") 
s9_400nm_67 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_50_00.txt")
s9_400nm_75 = Run(path + "400 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_44_20.txt")
angle_shifts_s9_400nm=[28, 45, 50, 58, 65, 74]

s9_300nm_30 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_13_09.txt")
s9_300nm_45 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_07_12.txt")
s9_300nm_52 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__17_01_52.txt")
s9_300nm_60 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_56_28.txt")
s9_300nm_67 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_51_05.txt")
s9_300nm_75 = Run(path + "300 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__16_45_01.txt")
angle_shifts_s9_300nm=[30, 45, 52, 60, 65, 77]

s9_220nm_30 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_50_30.txt")
s9_220nm_45 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_43_49.txt")
s9_220nm_52 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_37_13.txt")
s9_220nm_60 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_30_34.txt") 
s9_220nm_67 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_22_53.txt")
s9_220nm_75 = Run(path + "220 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__15_16_18.txt")
angle_shifts_s9_220nm=[30, 45, 52, 60, 65, 74]

s9_165nm_30 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_26_35.txt")
s9_165nm_45 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_16_28.txt")
s9_165nm_52 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_10_06.txt")
s9_165nm_60 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__14_03_03.txt") 
s9_165nm_67 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__13_56_29.txt")
s9_165nm_75 = Run(path + "165 nm measurements\\Power and sample 9 reflectivity measurements\\2018_12_06__13_49_07.txt")
angle_shifts_s9_165nm=[28, 44, 50, 58, 66, 74]

angle_shifts_none = [30, 45, 52, 60, 67, 75]

# For LXe data:
# Params to change in file_reader.py: const_err=100, frac_err=0.05
# intensity_correction=0.642 (165 nm), 0.856 (220 nm), 0.901 (300 nm), 0.909 (400 nm)
# Used scaled bkg for corresponding wavelength, no dark bkg file loaded in:
# In file_reader.py, used self.dark_bkg_intensities=150 (165 nm), 110 (220 nm), 120 (300 nm), 140 (400 nm)
# bkg_scaling=0.5 (0.8 for 165 nm)
# bkg = (self.beam_bkg_intensities-self.dark_bkg_intensities)*(self.incidentpower/self.beam_bkg_incidentpower)*bkg_scaling +self.dark_bkg_intensities
# Used sigma_theta_i=-1 for fitting and wavelength_F = True in TSTR_fit_new.py
# Angle range in file_reader stops at 80, same for plot_TSTR_fit in plotting.py
# Make sure n=n_LXe for plotting below
# Also explicitly set n=n_LXe for the given wavelength in TSTR_fit_new.py, both in F_wavelength_range() and in BRIDF_pair()
runs = [s9_165nm_30,s9_165nm_45,s9_165nm_52,s9_165nm_60,s9_165nm_67,s9_165nm_75]
angle_shifts=angle_shifts_s9_165nm


for run, angle in zip(runs, angle_shifts): run.change_theta_i(angle)
labels=[r"$\theta_i=30^{\circ}$","45$^{\circ}$","52$^{\circ}$","60$^{\circ}$", "67$^{\circ}$", "75$^{\circ}$"]

# Plot BRIDF data
fig, ax = plt.subplots()
plot_runs(runs, title="", log=True, labels=False, label=False, include_legend=False, errorbars=True, legend_loc=0, figure=False, colormap=True)
t0=time.time()

# Fit data
average_angle=4
precision=0.25
sigma_theta_i=-1
# fit_params = fit_parameters(get_independent_variables_and_relative_intensities(runs),p0=[.8, 1.44, .1, 0.8, 4.0],average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, use_errs=True,use_spike=True, use_nu=False,bounds=([0.1,1.1,0.03,0.01,1.0],[1.6,2.6,0.6,10.,50.]))

# Fit parameters using power correction, TR dist, 2.00 solid angle factor, Gaussian distribution in n
fit_params_s9_165nm_gauss_n= [0.217183610197179, 1.9181849064913858, 0.1536796098999006, 5.850296346659101]
fit_params_s9_220nm_gauss_n= [0.743200885061454, 1.4435971310198534, 0.08326263237026973, 0.541568091551811]
fit_params_s9_300nm_gauss_n= [0.9447069486729521, 1.4374429468203447, 0.1070442097507485, 1.5680079162288003]
fit_params_s9_400nm_gauss_n= [0.6021432485069854, 1.5440956423913457, 0.13516753727577357, 2.6357777432157024]

fit_params = fit_params_s9_165nm_gauss_n


print("Fit parameters (rho_L, n, gamma): "+str(fit_params))
#print("Fit angle: "+str(fit_ang))
t1=time.time()
print("Fitting time: {0}".format(t1-t0))


# Plot BRIDF model from fits
n_LXe_165 = 1.90128173
n_LXe_178 = 1.69
n_LXe_220 = 1.5044552
n_LXe_300 = 1.42975267
n_LXe_400 = 1.404459446
n_LXe = n_LXe_165
n=n_LXe

plt.xlim(0,86)
plt.ylim(1e-3,1e2)
plt.yscale("log")

#colors=["r","g","b","m","c","y"]
colors=[plt.cm.plasma(i) for i in np.linspace(0.2,0.90,len(runs))] 
# colors=[cmap(i) for i in np.linspace(0,1,len(runs))] 
label_x=[0.52,0.94,0.88,0.87,0.9,0.88] # Positions for 165 nm
label_y=[0.21,0.29,0.45,0.56,0.65,0.76]
# label_x=[0.61,0.94,0.94,0.88,0.85,0.86] # Positions for 220 nm
# label_y=[0.27,0.15,0.23,0.45,0.63,0.90]
# label_x=[0.61,0.94,0.94,0.88,0.85,0.86] # Positions for 300 nm
# label_y=[0.32,0.15,0.23,0.48,0.60,0.83]
# label_x=[0.60,0.94,0.94,0.94,0.85,0.86] # Positions for 400 nm; uses angle range of 0-86
# label_y=[0.28,0.21,0.29,0.36,0.55,0.70]
# label_x=[0.63,0.93,0.88,0.85,0.9,0.88] # Positions for s9 low_p (178 nm)
# label_y=[0.27,0.31,0.45,0.63,0.75,0.88]

angle_labels=[r"$\theta_i=30^{\circ}$",r"$45^{\circ}$",r"$52^{\circ}$",r"$60^{\circ}$",r"$67^{\circ}$",r"$75^{\circ}$"]
for ii in range(len(angle_shifts)):
	plot_TSTR_fit(angle_shifts[ii], n, fit_params, color=colors[ii], label=None, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i, include_fit_text=False)
	plt.text(label_x[ii], label_y[ii], angle_labels[ii],transform=plt.gca().transAxes,fontsize=13,color=colors[ii])

plt.text(0.103, 0.71, r"$\lambda$=165 nm",transform=plt.gca().transAxes,fontsize=13)	

# Set x-axis tick marks to every 10 deg
loc = matplotlib.ticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)

# Add grid lines
plt.grid(b=True,which='major',color="lightgray",linestyle='--')

#plt.tight_layout()

t2=time.time()
print("Plotting time: {0}".format(t2-t1))

run_data=get_independent_variables_and_relative_intensities(runs)
chi_sq=chi_squared(run_data[0], run_data[1], run_data[2], fit_params, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
print("Chi squared from fit: ",chi_sq)

plt.show()

