import yaml
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config', type=file)
parser.add_argument('-b', '--base_config', type=file, default='../astroJ/default.yaml')
args = parser.parse_args()

config = yaml.load(args.base_config)
config.update(yaml.load(args.config))

# constructing 'setting' dictionary for Minuit
err = 0.01
Minuit_setting = {}
fitting_params = []
passed_params = { 'DM_params':[] , 'ST_params':[] , 'ANI_params':[] }

for var in sorted(config):
    if var.split('_')[0] in config['DM_prof']:
        passed_params['DM_params'].append(config[var]['val'])
        if config[var]['free']:
            Minuit_setting.update({var.split('_')[1]:config[var]['val'],'error_%s'%var.split('_')[1]:err})
            fitting_params.append(var.split('_')[1])
    
    if var.split('_')[0] in config['stellar']:
        passed_params['ST_params'].append(config[var]['val'])
        if config[var]['free']:
            Minuit_setting.update({var.split('_')[1]:config[var]['val'],'error_%s'%var.split('_')[1]:err})
            fitting_params.append(var.split('_')[1])
    else:
        if var.split('_')[0] in config['I_prof'] and config['stellar']=='surf_bright':
            passed_params['ST_params'].append(config[var]['val'])
            if config[var]['free']:
                Minuit_setting.update({var.split('_')[1]:config[var]['val'],'error_%s'%var.split('_')[1]:err})
                fitting_params.append(var.split('_')[1])
    
    if var.split('_')[0] in config['anisotropy']:
        passed_params['ANI_params'].append(config[var]['val'])
        if config[var]['free']:
            Minuit_setting.update({var.split('_')[1]:config[var]['val'],'error_%s'%var.split('_')[1]:err})
            fitting_params.append(var.split('_')[1])

print 'config yaml file', config
print 'Minuit_setting', Minuit_setting
print 'fitting_params', fitting_params
print 'passed_params', passed_params