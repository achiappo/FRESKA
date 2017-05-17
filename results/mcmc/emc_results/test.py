import numpy as np
samples = np.load( 'samples_J1504_N5000.npy')
lnprobs = np.load( 'lnprobs_J1500_N5000.npy')

for i,(sam, likes) in enumerate( zip(samples, lnprobs) ):
    print 'walker ', i
    for s,l in zip(sam, likes):
	print s,l

