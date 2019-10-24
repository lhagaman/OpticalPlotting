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

# Setting matplotlib style using a style sheet: https://hfstevance.com/blog/2019/7/22/matplotlib-style
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
labels=[r"$\theta_i=30^{\circ}$","45$^{\circ}$","52$^{\circ}$","60$^{\circ}$", "67$^{\circ}$", "75$^{\circ}$"]

# Plot BRIDF data
sample_name="M17 Skived"
fig, ax = plt.subplots()
plot_runs(runs, title="", log=True, labels=False, label=False, include_legend=False, errorbars=True, legend_loc=0, figure=False, colormap=True)


# Fit data
average_angle=4
precision=0.25
sigma_theta_i=2

# Fit parameters using power correction, TR dist, 2.00 solid angle factor, Gaussian distribution in n
fit_params_s9_lowp_gauss_n= [0.7617591236933063, 1.581667834504223, 0.10965539347694214, 1.2535811658618297]
fit_params_s9_vac= [0.73367974544643089, 1.7043657376479024, 0.11826240150179418, 7.9564857778025644]
fit_params_s8_lowp_gauss_n= [0.9144536241966592, 1.5793399801859325, 0.08088290783956979, 1.2565135586082907, 4.857675748475832]

fit_params = fit_params_s9_vac


print("Fit parameters (rho_L, n, gamma): "+str(fit_params))

t1=time.time()


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

# run_data=get_independent_variables_and_relative_intensities(runs)
# chi_sq=chi_squared(run_data[0], run_data[1], run_data[2], fit_params, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i)
# print("Chi squared from fit: ",chi_sq)

# Now calculate hemispherical reflectance
fig2, ax2 = plt.subplots()

# Different incident angles to calculate for
x = [0,10,20,30, 45.01, 55, 60, 65, 70, 75, 80, 85]

# y_diffuse = [reflectance_diffuse(theta, n_LXe, 0.5, fit_params) for theta in x]
# y_specular = [reflectance_specular(theta, n_LXe, 0.5, fit_params) for theta in x]
# y_total = [y_diffuse[i] + y_specular[i] for i in range(len(y_specular))]

y_diffuse_vacuum_after = np.array([0.5854756333526625, 0.5854637654540487, 0.5852734323557967, 0.5843378330184013, 0.5781164319396017, 0.5648764205016575, 0.5521003835823036, 0.5320546931846964, 0.5005023765349177, 0.45035995118392885, 0.3692660409718563, 0.23436291271672846])
y_specular_vacuum_after = np.array([0.067067395165867863, 0.067178644142209448, 0.067764278912973513, 0.069721910622691077, 0.080284263006858245, 0.10021448509888969, 0.11806390704929674, 0.14446694450462855, 0.1837961174175215, 0.24523829211959774, 0.35678371257419206, 0.63661553122534764])
P_int_vacuum_after = np.array([0.98626722, 0.98560187, 0.98545861, 0.98549136, 0.98585454, 0.98671111, 0.98774621, 0.98989637, 0.99494025, 1.00883217, 1.05659101, 1.29664186]) # Integral of P(alpha)*sin(theta_r)/(4*cos(theta_i)) at each theta_i; should be 1, difference is presumably from numerical integration error
# y_specular_vacuum_after /= P_int_vacuum_after
y_total_vacuum_after = y_specular_vacuum_after + y_diffuse_vacuum_after
y_diffuse_vacuum_s8 = [0.7745583597188982, 0.7745451376434893, 0.7743314829696339, 0.7732652555639425, 0.765934975894844, 0.7497282283183715, 0.7336706247502036, 0.707982537880124, 0.6668053847700792, 0.600358570492193, 0.49181387633877344, 0.3109399449098984]
y_specular_vacuum_s8 = [0.044412473893000834, 0.044467836424858027, 0.044853619665046557, 0.046379979302917984, 0.055749040781957193, 0.075256179942695511, 0.093976436302740121, 0.12332498538510532, 0.16975365245641871, 0.24504066458477217, 0.37424248593626502, 0.62761217174077899]
P_int_vacuum_s8 = np.array([0.99350062, 0.99270412, 0.9925851, 0.99257328, 0.99264948, 0.99285361,
0.99311103, 0.99366995, 0.99507372, 0.9993997, 1.01738778, 1.13901055])
y_specular_vacuum_s8 /= P_int_vacuum_s8
y_total_vacuum_s8 = y_specular_vacuum_s8 + y_diffuse_vacuum_s8

y_diffuse_LXe_s9_lowp_gauss_n = np.array([0.6555220900082763, 0.6552533863910289, 0.6549497225806359, 0.6545160708355516, 0.6526151469567986, 0.6455809912310351, 0.6253474225536518, 0.4828032742121666, 0.27398385756675303, 0.10555036651013987, 0.04161843185672369, 0.019423169473866696])
y_specular_LXe_s9_lowp_gauss_n = np.array([0.0013792301879329701, 0.0013907277886691955, 0.0014670816824138941, 0.0017944406223630114, 0.006597003741463414, 0.036212891671076934, 0.09897912422754776, 0.24429904591916568, 0.4497687829214006, 0.633264713105453, 0.7660458024612156, 0.9554088934644756])
P_int_LXe_s9_lowp_gauss_n = np.array([0.98811858, 0.9874244,  0.98728108, 0.98729702, 0.98756767, 0.98821943, 0.98901473, 0.99068422, 0.99466309, 1.00589779, 1.04604892, 1.25891371])
# y_specular_LXe_s9_lowp_gauss_n /= P_int_LXe_s9_lowp_gauss_n
y_total_LXe_s9_lowp_gauss_n = y_specular_LXe_s9_lowp_gauss_n + y_diffuse_LXe_s9_lowp_gauss_n
y_diffuse_LXe_s8_lowp_gauss_n = np.array([0.7884466683816694, 0.788256353418893, 0.7880321186735665, 0.7876662132784379, 0.7856335665490475, 0.7770829665973333, 0.7507743533210781, 0.5745027731544261, 0.3115078692200302, 0.08448520812234325, 0.02924765179180736, 0.013162071720879362])
y_specular_LXe_s8_lowp_gauss_n = np.array([0.0014238945753465631, 0.001430458451793799, 0.0014888550616363077, 0.001759214904748927, 0.005393303821050943, 0.027585126391612244, 0.08267691197508208, 0.23013986297620168, 0.5896515389981184, 0.7880207287155164, 0.9326905430583597, 1.1146922646663053])
y_specular_lobe_only_LXe_s8_lowp_gauss_n = np.array([0.0014149943798950461, 0.001420865957344394, 0.001476681561927991, 0.0017398773265264382, 0.005305921143541413, 0.02704503980857446, 0.08081099020737664, 0.2207991308429237, 0.39978292584086694, 0.5035881396097269, 0.5024997833416238, 0.45985880641307364])
y_specular_spike_only_LXe_s8_lowp_gauss_n = y_specular_LXe_s8_lowp_gauss_n-y_specular_lobe_only_LXe_s8_lowp_gauss_n
P_int_LXe_s8_lowp_gauss_n = np.array([0.99350062, 0.99270412, 0.9925851, 0.99257328, 0.99264948, 0.99285361,
0.99311103, 0.99366995, 0.99507372, 0.9993997, 1.01738778, 1.13901055])
y_specular_lobe_only_LXe_s8_lowp_gauss_n /= P_int_LXe_s8_lowp_gauss_n
y_total_LXe_s8_lowp_gauss_n = y_specular_lobe_only_LXe_s8_lowp_gauss_n + y_specular_spike_only_LXe_s8_lowp_gauss_n + y_diffuse_LXe_s8_lowp_gauss_n

# y_specular_LXe=y_specular_LXe_s9_lowp_gauss_n
# y_diffuse_LXe=y_diffuse_LXe_s9_lowp_gauss_n
# y_total_LXe=y_total_LXe_s9_lowp_gauss_n
# P_int_LXe=P_int_LXe_s9_lowp_gauss_n

y_specular_LXe=y_specular_lobe_only_LXe_s8_lowp_gauss_n+y_specular_spike_only_LXe_s8_lowp_gauss_n
y_diffuse_LXe=y_diffuse_LXe_s8_lowp_gauss_n
y_total_LXe=y_total_LXe_s8_lowp_gauss_n
P_int_LXe=P_int_LXe_s8_lowp_gauss_n

plt.plot(x, y_specular_vacuum_s8, label="Specular, vacuum",linestyle='-',color='y')
plt.plot(x, y_specular_LXe, label="Specular, LXe",linestyle='--',color='y')
plt.plot(x, y_diffuse_vacuum_s8, label="Diffuse, vacuum",linestyle='-',color='c')
plt.plot(x, y_diffuse_LXe, label="Diffuse, LXe",linestyle='--',color='c')
plt.plot(x, y_total_vacuum_s8, label="Total, vacuum",linestyle='-',color='b')
plt.plot(x, y_total_LXe, label="Total, LXe",linestyle='--',color='b')

# R_labsphere=np.array([0.91,1.06,0.74,0.68,0.89,0.89])
# R_extrap=np.array([0.87,0.90,0.78,0.64,0.82,0.76])
R_labsphere=np.array([0.91,1.06,0.74,0.68,0.89,0.89,0.87,0.89]) # Including Spectralon data (last two points)
R_extrap=np.array([0.87,0.90,0.78,0.64,0.82,0.76,0.70,0.65])
# frac_err=np.mean((R_labsphere-R_extrap)/(R_extrap)) # avg fractional bias
# frac_err=np.sqrt(np.sum(((R_labsphere-R_extrap)/(R_extrap))**2)/len(R_extrap)) # fractional RMS of difference
# frac_err=0.17 # calculated from max difference between extrapolated reflectance and Labsphere value (worst for LUX at 500 nm, 0.76 vs 0.89; ignoring M17 turn data which is suspect)
# frac_err=0.08 # avg fractional bias between extrapolated reflectance and Labsphere value
# frac_err=0.14 # avg fractional bias including Spectralon data
frac_err=0.11 # fractional RMS of difference
# frac_err=0.18 # fractional RMS of difference including Spectralon data
num_int_err_vac=np.abs(1-P_int_vacuum_s8)*(y_specular_vacuum_s8/y_total_vacuum_s8)
num_int_err_LXe=np.abs(1-P_int_LXe)*(y_specular_LXe/y_total_LXe)
err_vac=np.sqrt(frac_err**2+num_int_err_vac**2)
err_LXe=np.sqrt(frac_err**2+num_int_err_LXe**2)
print(err_vac,err_LXe)
#plt.fill_between(x,y_total_LXe*(1-err_LXe),y_total_LXe*(1+err_LXe),alpha=0.1,facecolor='b')
#plt.fill_between(x,y_total_vacuum_s8*(1-err_vac),y_total_vacuum_s8*(1+err_vac),alpha=0.1,facecolor='b')
# # Line styles: '-', '--', '-.', ':'

plt.xlabel(r"Incident angle $\theta_i$ (degrees)")
plt.ylabel(r"Hemispherical reflectance $R(\theta_i)$")
plt.legend(loc=(0.05,0.08))
plt.ylim(0,1.1)

# Set x-axis tick marks to every 10 deg
loc = matplotlib.ticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
ax2.xaxis.set_major_locator(loc)

# Add grid lines
plt.grid(b=True,which='major',color="lightgray",linestyle='--')

# plt.title("Fitted "+sample_name+" Reflectance, 178 nm")
t3=time.time()
print("Reflectance calc time: {0}".format(t3-t2))
plt.show()

