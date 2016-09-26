import yaml
import argparse
import numpy as np
import pylab as plt

from test_func import Like

# parser
parser = argparse.ArgumentParser()
parser.add_argument('config', type=file)
parser.add_argument('-b', '--base_config', type=file, default='test.yaml')
args = parser.parse_args()

config = yaml.load(args.base_config)
config.update(yaml.load(args.config))

err = 0.01
Min_set = {}
profiles = []

for profile,dic in config.items():
    profiles.append(dic['name'])
    for param,vals in dic.items():
        if 'name' not in param:
            if vals['free']:
                Min_set.update({param:vals['val'],'error_%s'%param:err})
            else:
                Min_set.update({param:vals['val'],'fix_%s'%param:True})

# data extractor

# fitting class definition
class Fitter(object):

	def __init__(self, data, profiles):
		self.Like = Like(data, profiles)

	def __call__(self, params):
		return self.Like(params)

# fitting
settings = {'print_level':0,'errodef':0.5,'r0':1.,'error_r0':err}

Fit = Fitter(data, profiles)
Min = Minuit(Fit,Min_set.update(**settings))
Min.migrad()