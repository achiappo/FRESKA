#################################################################################################################
#################################################################################################################
#											PARAMETERS INPUT
#################################################################################################################
#################################################################################################################
# dark matter distribution section

# Select dark matter profile to adopt (choices: Zhao, Burkert, Einasto)
#*************************
DM_prof = 'Zhao'
#*************************

# dark matter profile parameters: replace the values with None to leave free in the fit
if DM_prof=='Zhao':
	aDM, bDM, cDM = 1, 1, 3			# enter shape parameters for Zhao profile
	DM_setting = {'DM_prof':DM_prof,'aDM':aDM,'bDM':bDM,'cDM':cDM}
elif DM_prof=='Einasto':
	alpha_Ei = None					# enter logarithmic slope for Einasto profile
	DM_setting = {'DM_prof':DM_prof,'alpha_Ei':alpha_Ei}
elif DM_prof=='Burkert':
	DM_setting = {'DM_prof':DM_prof}

#################################################################################################################
# Stellar distribution section

# Choose between providing either a generalised stellar density profile (Hernquist 1990) (star_density) 
# and its shape parameters or a surface brightness profile (surf_bright)
#*************************
stellar = 'star_density'
#*************************

# stellar density parameters: replace the values with None to leave free in the fit
if stellar=='star_density':
	rST = None						# enter scale radiues for Hernquist profile
	aST, bST, cST = 2, 5, 0  		# enter shape parameters
	ST_setting = {'stellar':stellar,'rST':rST,'aST':aST,'bST':bST,'cST':cST}
elif stellar=='surf_bright':
	# Select surface brightness profile to adopt (choices: Plummer, exponential, King, Sersic)
	#*************************
	I_prof = 'Plummer'
	#*************************

	# Surface brightness parameters: replace the values with None to leave free in the fit
	if I_prof=='Plummer':
		rST = None					# enter half-light radius to use in Plummer profile
		ST_setting = {'stellar':stellar,'I_prof':I_prof,'rST':rST}
	elif I_prof=='exponential':
		r_c = None					# enter exponential scale radius to use in exponential profile
		ST_setting = {'stellar':stellar,'I_prof':I_prof,'r_c':r_c}
	elif I_prof=='King':
		r_c = None					# enter core radius to use in King profile
		r_lim = None				# enter maximum radius
		ST_setting = {'stellar':stellar,'I_prof':I_prof,'r_c':r_c,'r_lim':r_lim}
	elif I_prof=='Sersic':
		r_c = None					# enter scale radius to use in Sersic profile
		n = None					# enter sharpness parameter of logarithmic decrease
		ST_setting = {'stellar':stellar,'I_prof':I_prof,'r_c':r_c,'n':n}


#################################################################################################################
# Velocity anisotropy section:

# Isotropic (IS), Radial (RD), Constant-beta (CA), Osipkov-Merritt (OM)
#*************************
anisotropy  = 'IS'
#*************************

# stellar anisotropy parameters: replace the values with None to leave free in the fit
if anisotropy=='IS':
	ST_setting = {'anisotropy':anisotropy}
elif anisotropy=='RD':
	ST_setting = {'anisotropy':anisotropy}
elif anisotropy=='CA':
	beta = None					# enter beta for costant anisotropy profile
	ST_setting = {'anisotropy':anisotropy,'beta':beta}
elif anisotropy=='OM':
	r_a = None					# enter scale radius for Osipkov-Merritt profile (Osipkov 1979; Merritt 1985)
	ST_setting = {'anisotropy':anisotropy,'r_a':r_a}

#################################################################################################################
# Save all the input parameters into a yaml file to be read by functions.pyx
import yaml
yaml.dump([DM_setting,ST_setting
	{'stellar':stellar,'stellar_params':ST_params},
	{'anisotropy':anisotropy,'anisotropy_params':ASTY_params}],open('params.yaml','w'))
