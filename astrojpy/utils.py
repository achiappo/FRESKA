import numpy as np

# extract data from data files
def load_data(gal, directory='.'):
    # Read the parameter from the input file
    with open(directory+'/data/params/params_%s.dat'%gal,'r') as datafile:
        data = datafile.readlines()

    parameters = [line[:-1] for line in data]
    D  = float(parameters[1])
    rh = float(parameters[2])
    rt = float(parameters[3])
        
    x,v,dv = np.loadtxt(directory+'/data/velocities/velocities_%s.dat'%gal,
                        dtype=float,
                        usecols=(0,1,2),
                        unpack=True)
    return x, v, dv, D, rh, rt

def load_gaia(homedir, MockSize, dataSize, dset, mod, D, with_velocity_errors):
    #homedir = '/home/andrea/Desktop/work/DWARF/jvalue/project1/test/Isotrop_Core_nonPlum'
    #MockSize = 100
    #dataSize = 1000
    #dset=1
    # enter model choice - cf. casedir (options 1,2,3,4)
    #mod = 4

    with_velocity_errors=True
    
    data = ['/data/gs100_bs050_rcrs100_rarcinf_core_0400mpc3_df_%i_%i.dat'%(dataSize,dset),			# Isotrop_Core_nonPlum
        	'/data/gs010_bs050_rcrs100_rarcinf_core_0400mpc3_df_%i_%i.dat'%(dataSize,dset),			# Isotrop_Core_Plum
        	'/data/gs100_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_%i_%i.dat'%(dataSize,dset),			# Isotrop_Cusp_nonPlum
        	'/data/gs010_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_%i_%i.dat'%(dataSize,dset)]			# Isotrop_Cusp_Plum
    err  = ['/data/gs100_bs050_rcrs100_rarcinf_core_0400mpc3_df_%i_%i_err.dat'%(dataSize,dset),		# Isotrop_Core_nonPlum
        	'/data/gs010_bs050_rcrs100_rarcinf_core_0400mpc3_df_%i_%i_err.dat'%(dataSize,dset),  	# Isotrop_Core_Plum
        	'/data/gs100_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_%i_%i_err.dat'%(dataSize,dset),  	# Isotrop_Cusp_nonPlum
        	'/data/gs010_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_%i_%i_err.dat'%(dataSize,dset)]  	# Isotrop_Cusp_Plum
    print 'Using ', homedir+data[mod-1]
    x,y,z,vx,vy,vz = np.loadtxt(homedir+data[mod-1],unpack=True)
    R = np.sqrt(x**2+y**2) # assumed direction of observation along z-axis for simplicity (as suggested on the Gaia wiki)
    d = np.sqrt(x**2+y**2+(D-z)**2)
    if not with_velocity_errors:
        v = (x*vx+y*vy+(D-z)*vz)/d
        dv = np.zeros_like(v)
    else:
        # Errors (from mock data) preparation
        Evx,Evy,Evz = np.loadtxt(homedir+err[mod-1],unpack=True,usecols=(3,4,5))
        Ex,Ey,Ez = Evx-vx, Evy-vy,Evz-vz
        v = (x*Evx+y*Evy+(D-z)*Evz)/d
        dv = (x*Ex+y*Ey+(D-z)*Ez)/d
    
    if MockSize<dataSize:
        idx=np.random.randint(low=dataSize, size=MockSize)
        R, v, dv = R[idx], v[idx], dv[idx]

    rh = 1. if mod == 1 or mod == 2 else 0.25
    r0_true = 1.
    rho0_true = 40.e7 if mod == 1 or mod == 2 else 6.4e7

    return R, v, dv, rh, r0_true, rho0_true

def envelope(Jmc, Smc, Lmc):
    """ tool to envelope the result of an MCMC scan
        to the lowermost L values and ordered in J
    """
    # determine minimum likelihood value and corresponding J
    Lmin = min(Lmc)
    Jmin = Jmc[ np.where(Lmc==Lmin)[0][0] ]
    # rearrange arrays for increasing J
    inds = np.argsort(Jmc)
    Jmc, Smc, Lmc = Jmc[inds], Smc[inds], Lmc[inds]
    # split arrays into "left wing" and "right wing" values
    Jlow, Jhig = Jmc[ Jmc<=Jmin ], Jmc[ Jmc>Jmin ]
    Slow, Shig = Smc[ Jmc<=Jmin ], Smc[ Jmc>Jmin ]
    Llow, Lhig = Lmc[ Jmc<=Jmin ], Lmc[ Jmc>Jmin ]

    # build left wing of the likelihood curve
    Jenv, Senv, Lenv = [], [], []
    # append first element
    Jenv.append(Jlow[0])
    Senv.append(Slow[0])
    Lenv.append(Llow[0])
    # fill left lowermost L values
    for J,S,L in zip(Jlow, Slow, Llow):
        if L<Lenv[-1]:
            Jenv.append(J)
            Senv.append(S)
            Lenv.append(L)
    
    # build right wing of the likelihood curve
    JenvR, SenvR, LenvR = [], [], []
    # append last element
    JenvR.append(Jhig[-1])
    SenvR.append(Shig[-1])
    LenvR.append(Lhig[-1])
    # fill right lowermost L values
    for J,S,L in zip(reversed(Jhig), reversed(Shig), reversed(Lhig)):
        if L<LenvR[-1]:
            JenvR.append(J)
            SenvR.append(S)
            LenvR.append(L)
    
    # combine segments into individual arrays
    Jenv.extend([j for j in reversed(JenvR)])
    Senv.extend([s for s in reversed(SenvR)])
    Lenv.extend([l for l in reversed(LenvR)])

    # convert into numpy arrays for convenience and return
    return np.asarray(Jenv), np.asarray(Senv), np.asarray(Lenv)

