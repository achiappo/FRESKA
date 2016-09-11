import yaml
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config', type=file)
parser.add_argument('-b', '--base_config', type=file, default='config.yaml')
args = parser.parse_args()

config = yaml.load(args.base_config)
config.update(yaml.load(args.config))

# constructing 'setting' dictionary for Minuit
err = 0.01
Minuit_setting = {}
fitting_params = {}
for profile_key,profile_dic in config.items():
    if profile_dic['use']:
        for variable_key,variable_dic in profile_dic.items():
            if variable_key not in ['use']:
                if variable_dic['free']:
                    Minuit_setting.update({variable_key:variable_dic['val'],
                    					   'error_%s'%variable_key:err})
                    fitting_params.update({variable_key:variable_dic['val']})
                else:
                    Minuit_setting.update({variable_key:variable_dic['val'],
                    					   'fixed_%s'%variable_key:True})

print 'Minuit_setting', Minuit_setting
