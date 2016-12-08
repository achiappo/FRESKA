import numpy as np

# extract data from data files
def load_data(gal):
    # Read the parameter from the input file
    data = open('../data/params/params_%s.dat'%gal,'r').readlines()
    parameters = []
    for line in data:
        parameters.append(line[:-1])
        D  = float(parameters[1])
        rh = float(parameters[2])
        rt = float(parameters[3])
        
    x,v,dv = np.loadtxt('../data/velocities/velocities_%s.dat'%gal,dtype=float,usecols=(0,1,2),unpack=True)
    return x,v,dv,D,rh,rt

def load_gaia(homedir, MockSize, dataSize, dset, mod, D, with_velocity_errors):
    # homedir = '/home/cohen/WORK/JFACTOR/ASTROJPY/'
    # MockSize = 100
    # dataSize = 1000
    # dset=1
    # # enter model choice - cf. casedir (options 1,2,3,4)
    # mod = 4

    with_velocity_errors=True
    
    data = ['/data/gs100_bs050_rcrs100_rarcinf_core_0400mpc3_df_%i_%i.dat'%(dataSize,dset),      # Isotrop_Core_nonPlum
        '/data/gs010_bs050_rcrs100_rarcinf_core_0400mpc3_df_%i_%i.dat'%(dataSize,dset),      # Isotrop_Core_Plum
        '/data/gs100_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_%i_%i.dat'%(dataSize,dset),      # Isotrop_Cusp_nonPlum
        '/data/gs010_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_%i_%i.dat'%(dataSize,dset)]      # Isotrop_Cusp_Plum
    err  = ['/data/gs100_bs050_rcrs100_rarcinf_core_0400mpc3_df_%i_%i_err.dat'%(dataSize,dset),  # Isotrop_Core_nonPlum
        '/data/gs010_bs050_rcrs100_rarcinf_core_0400mpc3_df_%i_%i_err.dat'%(dataSize,dset),  # Isotrop_Core_Plum
        '/data/gs100_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_%i_%i_err.dat'%(dataSize,dset),  # Isotrop_Cusp_nonPlum
        '/data/gs010_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_%i_%i_err.dat'%(dataSize,dset)]  # Isotrop_Cusp_Plum
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
    R,v,dv=R[idx],v[idx],dv[idx]

    rh= 1. if mod == 1 or mod == 2 else 0.25
    r0_true=1.
    rho0_true=40.e7 if mod == 1 or mod == 2 else 6.4e7

    return R,v,dv, rh, r0_true, rho0_true
