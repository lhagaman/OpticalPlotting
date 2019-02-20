import numpy as np
import matplotlib.pyplot as plt
from gaussian_fit import gaussian_get_params
from TSTR_fit_new import BRIDF_plotter

# takes a single run, list of runs, or list of lists of runs. For a list of lists, it gives each inner list the same color and label.
def plot_runs(runs, title=False, rot=False, voltage=False, labels=False, label=False, show=False, xlabel="", ylabel="", figure=True, smooth=False, legend_loc=0, include_legend=True, linestyle="-", log=False, errorbars=True, color="", colormap=False):
	if type(runs) != type([]): # if runs is a single run
		runs = [runs] # make it a list

	if type(runs[0]) != type([]):
		nested = False
	else:
		nested = True

	if figure:
		plt.figure()

	if color:		
		color_list = [color]*len(runs)
	else:
		if len(runs) > 12 or colormap:	# If using a long list, color in order according to 'cool' colormap
			color_list = [plt.cm.cool(i) for i in np.linspace(0,1,len(runs))] 
		else:
			color_list = ["r", "g", "b", "m", "c", "y", "k", "lightpink", "darksalmon", "slategray", "plum", "lightcoral", "indigo", "darkorange"]

	minx = 1000
	maxx = -1000

	miny = 1000
	maxy = -1000

	if nested:
		flattened_runs = sum(runs)
	else:
		flattened_runs = runs

	for i in range(len(flattened_runs)):
		run = flattened_runs[i]
		# rot means that angles are relative to the rotation stage rather than to the sample normal
		if rot:
			x = run.rot_angles
		else:
			x = run.angles
		minx = np.min([minx, np.min(x)])
		maxx = np.max([maxx, np.max(x)])
		# voltage means use absolute intensities, not normalized ones
		if voltage:
			y = run.intensities
			yerr = run.intensity_std
		else:
			y = run.relative_intensities
			yerr = run.relative_std
		maxy = np.max([maxy, np.max(y)])
		miny = np.min([miny, np.min(y)])
		if label: #use only one label, only label first run, designed for all runs being the same color
			if i != 0:
				label=False
		if labels:
			label=labels[i]
		# smooth means a line will also be plotted, not just points
		if smooth: 
			plt.plot(x, y, c=color_list[i], linestyle=linestyle)
		if errorbars: 
			if label:
				plt.errorbar(x, y, yerr=yerr, fmt="o", c=color_list[i], ms=2, capsize=2, label=label)
			else:
				plt.errorbar(x, y, yerr=yerr, fmt="o", c=color_list[i], ms=2, capsize=2)
		else: 
			if label:
				plt.scatter(x, y, marker="x", c=color_list[i], s=5, label=label)
			else:
				plt.scatter(x, y, marker="x", c=color_list[i], s=5)

	if xlabel:
		plt.xlabel(xlabel)
	else:
		if rot:
			plt.xlabel("rotation stage angle \n(0 is looking directly into beam, 180 is blocking beam)")
		else:
			plt.xlabel("angle relative to sample normal (degrees)")

	if ylabel:
		plt.ylabel(ylabel)
	else:
		if voltage:
			plt.ylabel("rate (Hz)")
		else:
			plt.ylabel("intensity (reflected rate/str)/(incident rate)")
	if log:
		plt.ylim(0.5 * np.max([miny, 0.001]), 5. * maxy)
		plt.yscale("log")
	else:
		plt.ylim(0, 1.1 * maxy)
	plt.xlim(minx, maxx)
	#plt.ylim(1e-3,1e2)
	if title:
		plt.title(title)
	if include_legend:
		plt.legend(loc=legend_loc)
	if show:
		plt.show()

def plot_gaussian_fit(run, mu=1000):
	params = gaussian_get_params(run, mu)

	A, B, sigma, mu = params

	label = "gaussian fit\nA: " + str(A) + "\nB: " + str(B) + "\nsigma: " + str(sigma) + "\nmu: " + str(mu)
	plot_fit_by_params(min(x), max(x), params, label)


#works with or without spike
def plot_TSTR_fit(theta_i, n, fit_params, label="", color="", average_angle=0, precision=-1, sigma_theta_i=2.0, fit_text="Fit: ", fit_text_offset=0, phi_r=0, include_fit_text=True):
	min_angle = 0
	max_angle = 85
	d_theta = 1
	n_angles = (max_angle-min_angle)/d_theta+1
	x = np.linspace(min_angle, max_angle, n_angles)
	if label:
		plt.plot(x, BRIDF_plotter(x, phi_r, theta_i, n, 0.5, fit_params, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i), label=label, color=color)
	else:
		plt.plot(x, BRIDF_plotter(x, phi_r, theta_i, n, 0.5, fit_params, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i), color=color)
	if include_fit_text:
		if len(fit_params) == 3:
			plt.text(0.05,0.05 + fit_text_offset, fit_text + r"$\rho_L$={0:.4f}, n={1:.4f}, $\gamma$={2:.4f}".format(*fit_params),transform=plt.gca().transAxes,fontsize=13)
		else:
			plt.text(0.05,0.05 + fit_text_offset, fit_text + r"$\rho_L$={0:.4f}, n={1:.4f}, $\gamma$={2:.4f}, K={3:.4f}".format(*fit_params),transform=plt.gca().transAxes,fontsize=13)
	plt.legend()

"""fit_params=["rho_L_1","n_1","gamma_1","K_1",
             ,"gamma_2","frac_1"]
superposes two TSTR fits, intended to be and one layer of LXe and PTFE (labeled as 1) and one layer of LXe and gas (labeled as 2)
"""
def plot_double_TSTR_fit(theta_i, n, fit_params, labels=[], colors="", average_angle=0, precision=-1, sigma_theta_i=2.):
	fit_params_1 = fit_params[0:4]
	gamma_2 = fit_params[4]
	frac_1 = fit_params[5]
	fit_params_2 = [fit_params[0], 1.01, gamma_2] # don't include specular spike; rho_2 = rho_1; n_2 = 1+eps to avoid log errors

	min_angle = 0
	max_angle = 85
	d_theta = 1
	n_angles = (max_angle-min_angle)/d_theta+1
	x = np.linspace(min_angle, max_angle, n_angles)
	y_1 = frac_1 * np.array(BRIDF_plotter(x, 0., theta_i, n, 0.5, fit_params_1, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
	y_2 = (1 - frac_1) * np.array(BRIDF_plotter(x, 0., theta_i, n, 0.5, fit_params_2, average_angle=average_angle, precision=precision, sigma_theta_i=sigma_theta_i))
	y = y_1 + y_2

	if labels:
		plt.plot(x, y, label=labels[0], color=colors[0])
	else:
		plt.plot(x, y, color=colors[0])

	if labels:
		plt.plot(x, y_1, label=labels[1], color=colors[1], linestyle='--')
	else:
		plt.plot(x, y_1, color=colors[1], linestyle='--')

	if labels:
		plt.plot(x, y_2, label=labels[2], color=colors[2], linestyle='-.')
	else:
		plt.plot(x, y_2, color=colors[2], linestyle='-.')

	rho_L_1 = fit_params_1[0]
	n_1 = fit_params_1[1]
	gamma_1 = fit_params_1[2]
	K_1 =  fit_params_1[3]

	string = "Fit: rho_L_1=" + str(np.around(rho_L_1, 3)) + ", n_1=" + str(np.around(n_1, 3)) + ", gamma_1=" + str(np.around(gamma_1, 3)) + ", K_1=" + str(np.around(K_1, 3))
	plt.text(0.05, 0.15, string, transform=plt.gca().transAxes,fontsize=10)
	string = "      gamma_2=" + str(np.around(gamma_2, 3)) + ", frac_1=" + str(np.around(frac_1, 3)) + ", sigma_theta_i=" + str(np.around(sigma_theta_i, 3))
	plt.text(0.05, 0.1, string, transform=plt.gca().transAxes,fontsize=10)
	
	"""
	plt.text(0.05,0.1,r"Fit: $\rho_L_1$={0:.3f}, n_1={1:.3f}, $\gamma_1$={2:.3f}".format(*fit_params_1),transform=plt.gca().transAxes,fontsize=13)
	plt.text(0.05,0.05,r"Fit: $\rho_L_2$={0:.3f}, n_2={1:.3f}, $\gamma_2$={2:.3f}".format(*fit_params_2),transform=plt.gca().transAxes,fontsize=13)
	plt.text(0.05,0.,r"Fit: frac\_1={0:.3f}".format(*[frac_1]),transform=plt.gca().transAxes,fontsize=13)
	"""
	
	plt.legend()


