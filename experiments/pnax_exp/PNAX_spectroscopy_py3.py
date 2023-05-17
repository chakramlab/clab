 # -*- coding: utf-8 -*-
"""
Created on Sun Aug 04 2015

@author: Nate E
"""

from slab import *
from slab.datamanagement import SlabFile
from numpy import *
import os
import time
from slab.instruments import *
from slab.instruments import InstrumentManager
from PNAX_spectroscopy_utils import *
from pnaxpulseexperiment import *
#from slab.instruments.labbrick import labbrick
im = InstrumentManager(config_path="C:\_Lib\python\slab\instruments\instruments.cfg")
nwa = im['PNAX']
# nwa=N5242A(address='192.168.14.181')
try:
    LSG=im['LB1']
except:
    print('Labbrick not connected')

print(nwa.get_id())
print('Device Connected')
print(os.getcwd())
expt_path = os.getcwd() + '\Data'



config={'instrument':nwa,
        'prefix':'Nb3-2 Storage 0dBm',
        'read_freq': 6.655911E9,
        'probe_freq_center':3.7467573e9,
        'span': 100E3,
        'read_power' : 0,
        'probe_power': 0,
        'sweep_pts':2000,
        'ifbw':100,
        'avgs':2,
        'avgs_state': True,
        'EF_probe':False,
        'ge_freq':6.432E9,
        'ge_power':5.0,
        'LSG_ext_ref': True,
        'measurement':'S21',
        'elec_delay':94.94E-9,
        'phase_offset':0.0,
        "search_range":20E6,
        "search_window":100E3,
        'expt_path':expt_path
        }




# freq_list=[1, 2, 3, 5]
# wide_sweep_CW(config)
single_tone_CW(config)
# two_tone_CW(config)