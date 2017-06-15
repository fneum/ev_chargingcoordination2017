import configparser
config = configparser.ConfigParser()
config.read('../parameters/evalParams.ini')
print (config.getboolean('uncertainty', 'unc_rv',fallback='Request not in parameter set!'))