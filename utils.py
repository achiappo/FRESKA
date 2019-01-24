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

def load_gaia(homedir, MockSize, dataSize, dset, mod, D, with_velocity_errors=True):

    data = ['/data/gs100_bs050_rcrs100_rarcinf_core_0400mpc3_df_%i_%i.dat'%(dataSize,dset),      # Isotrop_Core_nonPlum
            '/data/gs010_bs050_rcrs100_rarcinf_core_0400mpc3_df_%i_%i.dat'%(dataSize,dset),      # Isotrop_Core_Plum
            '/data/gs100_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_%i_%i.dat'%(dataSize,dset),      # Isotrop_Cusp_nonPlum
            '/data/gs010_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_%i_%i.dat'%(dataSize,dset),      # Isotrop_Cusp_Plum
            '/data/gs100_bs050_rcrs025_rarc100_core_0400mpc3_df_%i_%i.dat'%(dataSize,dset),      # OsipkMerr_Core_nonPlum
            '/data/gs010_bs050_rcrs025_rarc100_core_0400mpc3_df_%i_%i.dat'%(dataSize,dset),      # OsipkMerr_Core_Plum
            '/data/gs100_bs050_rcrs010_rarc100_cusp_0064mpc3_df_%i_%i.dat'%(dataSize,dset),      # OsipkMerr_Cusp_nonPlum
            '/data/gs010_bs050_rcrs010_rarc100_cusp_0064mpc3_df_%i_%i.dat'%(dataSize,dset)]      # OsipkMerr_Cusp_Plum

    err  = ['/data/gs100_bs050_rcrs100_rarcinf_core_0400mpc3_df_%i_%i_err.dat'%(dataSize,dset),  # Isotrop_Core_nonPlum
            '/data/gs010_bs050_rcrs100_rarcinf_core_0400mpc3_df_%i_%i_err.dat'%(dataSize,dset),  # Isotrop_Core_Plum
            '/data/gs100_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_%i_%i_err.dat'%(dataSize,dset),  # Isotrop_Cusp_nonPlum
            '/data/gs010_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_%i_%i_err.dat'%(dataSize,dset),  # Isotrop_Cusp_Plum
            '/data/gs100_bs050_rcrs025_rarc100_core_0400mpc3_df_%i_%i_err.dat'%(dataSize,dset),  # OsipkMerr_Core_nonPlum
            '/data/gs010_bs050_rcrs025_rarc100_core_0400mpc3_df_%i_%i_err.dat'%(dataSize,dset),  # OsipkMerr_Core_Plum
            '/data/gs100_bs050_rcrs010_rarc100_cusp_0064mpc3_df_%i_%i_err.dat'%(dataSize,dset),  # OsipkMerr_Cusp_nonPlum
            '/data/gs010_bs050_rcrs010_rarc100_cusp_0064mpc3_df_%i_%i_err.dat'%(dataSize,dset)]  # OsipkMerr_Cusp_Plum

    print 'Using ', homedir+data[mod]
    x,y,z,vx,vy,vz = np.loadtxt(homedir+data[mod],unpack=True)
    R = np.sqrt(x**2+y**2) # assumed direction of observation along z-axis for simplicity (as suggested on the Gaia wiki)
    d = np.sqrt(x**2+y**2+(D-z)**2)
    if not with_velocity_errors:
        v = (x*vx+y*vy+(D-z)*vz)/d
        dv = np.zeros_like(v)
    else:
        # Errors (from mock data) preparation
        Evx,Evy,Evz = np.loadtxt(homedir+err[mod],unpack=True,usecols=(3,4,5))
        Ex,Ey,Ez = Evx-vx, Evy-vy,Evz-vz
        v = (x*Evx+y*Evy+(D-z)*Evz)/d
        dv = (x*Ex+y*Ey+(D-z)*Ez)/d
    
    if MockSize<dataSize:
        idx=np.random.randint(low=dataSize, size=MockSize)
        R, v, dv = R[idx], v[idx], dv[idx]

    if mod < 2:
        rh = 1.
    elif 2 <= mod <= 5:
        rh = 0.25
    else:
        rh = 0.1

    cst = 1. if mod%2 == 0 else 0.1
    r0_true = 1.
    rho0_true = 40.e7 if mod in [0,1,4,5] else 6.4e7

    return R, v, dv, rh, cst, r0_true, rho0_true

def envelope(samples, lnprobs, param=0):
    """ tool to envelope the result of a MCMC scan
        to the lowermost -lnLike values along an 
        ordered direction of the samples 
    """
    # verify that the parameter chosen corresponds 
    # to the dimensionality of the samples array
    samples_dim = samples.shape[-1]
    if samples_dim != len(samples):
        unidim = False
        assert param in range(samples_dim), \
            "wrong choice of 'param' index! \n \
            dimensionality of samples: %i"%(samples_dim)
    else:
        unidim = True
    
    # separate parameter of interest 
    # and others into distinct arrays
    if not unidim:
        Pmc = samples[:, param]
        Smc = samples[:, [i for i in range(samples_dim) if i!=param]]
    else:
        Pmc = samples
        Smc = np.zeros_like(Pmc)

    # rearrange arrays for increasing P (parameter of interest array)
    Lmc = np.absolute(lnprobs)
    sortind = np.argsort(Pmc)
    Pmc, Smc, Lmc = Pmc[sortind], Smc[sortind], Lmc[sortind]
    # determine minimum -lnlikelihood value and corresponding index
    Lmin = min(Lmc)
    indLmin = np.where(Lmc==Lmin)[0]
    
    # case 1: minimum is the left-most entry
    if min(indLmin)==0:
        # build only right wing of the envelope
        PenvR, SenvR, LenvR = [], [], []
        # append last element
        PenvR.append(Pmc[-1])
        SenvR.append(Smc[-1])
        LenvR.append(Lmc[-1])
        # fill right lowermost L values
        for P,S,L in zip(reversed(Pmc), reversed(Smc), reversed(Lmc)):
            if L<LenvR[-1]:
                PenvR.append(P)
                SenvR.append(S)
                LenvR.append(L)
        Penv = [p for p in reversed(PenvR)]
        Senv = [s for s in reversed(SenvR)]
        Lenv = [l for l in reversed(LenvR)]


    # case 2: the minimum is the right-most entry
    if max(indLmin)==len(Pmc)-1:
        # build only left wing of the envelope
        Penv, Senv, Lenv = [], [], []
        # append first element
        Penv.append(Pmc[0])
        Senv.append(Smc[0])
        Lenv.append(Lmc[0])
        # fill left lowermost L values
        for P,S,L in zip(Pmc, Smc, Lmc):
            if L<Lenv[-1]:
                Penv.append(P)
                Senv.append(S)
                Lenv.append(L)

    # case 3: (general case) the minimum is in the middle
    if 0<min(indLmin) and max(indLmin)<len(Pmc)-1:
        Pmin = Pmc[indLmin[0]]
        # split arrays into "left wing" and "right wing" values
        Plow, Phig = Pmc[ Pmc<=Pmin ], Pmc[ Pmc>Pmin ]
        Slow, Shig = Smc[ Pmc<=Pmin ], Smc[ Pmc>Pmin ]
        Llow, Lhig = Lmc[ Pmc<=Pmin ], Lmc[ Pmc>Pmin ]

        # build left wing of the envelope
        Penv, Senv, Lenv = [], [], []
        # append first element
        Penv.append(Plow[0])
        Senv.append(Slow[0])
        Lenv.append(Llow[0])
        # fill left lowermost L values
        for P,S,L in zip(Plow, Slow, Llow):
            if L<Lenv[-1]:
                Penv.append(P)
                Senv.append(S)
                Lenv.append(L)

        # build right wing of the envelope
        PenvR, SenvR, LenvR = [], [], []
        # append last element
        PenvR.append(Phig[-1])
        SenvR.append(Shig[-1])
        LenvR.append(Lhig[-1])
        # fill right lowermost L values
        for P,S,L in zip(reversed(Phig), reversed(Shig), reversed(Lhig)):
            if L<LenvR[-1]:
                PenvR.append(P)
                SenvR.append(S)
                LenvR.append(L)

        # combine segments into individual arrays
        Penv.extend([p for p in reversed(PenvR)])
        Senv.extend([s for s in reversed(SenvR)])
        Lenv.extend([l for l in reversed(LenvR)])

    # convert into numpy arrays for convenience and return
    return np.asarray(Penv), np.asarray(Senv), np.asarray(Lenv)

