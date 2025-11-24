__author__ = 'Alex Ma'

import slab
from pnaxpulseexperiment import PnaxPulseExperiment
import os

'''
General Experiments:
=======================
vacuum_rabi
cavity_ringdown
pulse_probe
Rabi
T1
Ramsey
Spin_Echo
raman
raman_rabi
raman_t1
raman_ramsey
raman_spin_echo

(To do:)
Histogram

General To do:
Save cfg into datafile
=======================

Pulse box connection
pulse0 - internal ADC
pulse1 - RF1 (Port 1) - readout
pulse2 - RF2 (Port 3) - drive1 (regular)
pulse3 -              - drive2 (drive 1+2 = ramsey)
pulse4 -              - drive3 (drive 1+2+3 = spin echo)

'''

expt_name ="vacuum_rabi"
# expt_name="pulse_probe"
# expt_name='t1'
# expt_name='rabi'
# expt_name='ramsey'
# config_file = os.getcwd()+'\\config_pnax_vacuum_rabi.json'
# config_file = os.getcwd()+'\\pulse_probe.json'
# config_file=os.getcwd()+'\\config_pnax_T1.json'
config_file=os.getcwd()+'\\config_pnax_rabi.json'
print(config_file)
# config_file = '..\\config_pnax_0321_Q3_LSS.json'
# config_file = '..\\config_pnax_0323_Q3_dcbias_5.1GHz.json'

# PnaxPulseExperiment(expt_name, config_file)

# temp='7'
filename='resonator_spectroscopy_pulsed_test'

fname=PnaxPulseExperiment(expt_name, config_file, filename=filename, path=os.getcwd()+'\\Data\\Time Domain')
# PnaxPulseExperiment(expt_name, config_file, filename=f'pulse_probe_-40dBm', path=os.getcwd()+'\\Data\\Time Domain')
# PnaxPulseExperiment('raman', config_file)
# PnaxPulseExperiment('raman_ramsey', config_file)
# PnaxPulseExperiment('rabi', config_file)  