import numpy as np
import json

opx_ip = "192.168.88.252"
opx_port = 9510
octave_ip = "192.168.88.254"
octave_port = 80
con = "con1"
octave = "octave1"
######################
# AUXILIARY FUNCTIONS:
######################

with open("experiments.json", "r") as f:
    experiment_config = json.load(f)

def gauss(amplitude, mu, sigma, length):
    
    length = int(length)
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    
    return np.ndarray.tolist(np.nan_to_num(np.array([float(x) for x in gauss_wave]), nan=amplitude))


def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1-g**2)*(2*c**2-1))
    return [float(N * x) for x in [(1-g)*c, (1+g)*s, (1-g)*s, (1+g)*c]]


################
# CONFIGURATION:
################

long_readout_len = 20000
readout_len = 6000

qubit_IF = 270e6
qubit_multi_IF=379e6
rr_IF =  200e6  # rr -> readout resonator
rr_LO= 7.0e9
qubit_LO = 6.3e9

IF = rr_IF
LO = rr_LO
pi_pulse_time = int(experiment_config["__Shared Values__"]["Tpi (ns)"])
gauss_len = 16

qubit_pulse = "gaussian"
# rr_freq = 7.26e9

pi_pulse_length = pi_pulse_time
pi_pulse_length_over_2 = (pi_pulse_time//8)*4
pi_pulse_length_over_4 = (pi_pulse_time//16)*4

if(pi_pulse_time<16):
    pi_pulse_length = 16
if((pi_pulse_time/2) < 16):
    pi_pulse_length_over_2 = 16
if((pi_pulse_time/4) < 16):
    pi_pulse_length_over_4 = 16


config = {

    'version': 1,

    'controllers': {
        con : {
            "analog_outputs": {
                1: {"offset": 0.0},
                2: {"offset": 0.0},
                3: {"offset": 0.0},
                4: {"offset": 0.0},
                5: {"offset": 0.0},
                6: {"offset": 0.0},
                7: {"offset": 0.0},
                8: {"offset": 0.0},
                9: {"offset": 0.0},
                10: {"offset": 0.0},
            },
            "digital_outputs": {
                1: {},
                2: {},
                3: {},
                4: {},
                5: {},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
            },
        }
    },

    'elements': {

        'qubit': {
            'mixInputs': {
                'I': ('con1', 1),
                'Q': ('con1', 2),
                'lo_frequency': qubit_LO,
                'mixer': f"octave_{octave}_1"
            },
            'intermediate_frequency': qubit_IF,
            'operations': {
                'CW': 'CW',
                'saturation': f'{qubit_pulse}_pulse',
                'saturation_multi': 'saturation_pulse_multi',
                'gaussian': 'gaussian_pulse',
                'test':'test_pulse',
                'pi': f'{qubit_pulse}_pi_pulse',
                'pi/2' : f'{qubit_pulse}_pi/2_pulse',
                # 'pi/4' : f'{qubit_pulse}_pi/4_pulse',
                'marker': 'marker_pulse'
            },
            'digitalInputs': {
                "Switch": {
                    "port": ("con1", 1),
                    "delay": 58,
                    "buffer": 57,
                },
            },

        },
        'qubit_multi': {
            'mixInputs': {
                'I': ('con1', 5),
                'Q': ('con1', 6),
                'lo_frequency': qubit_LO,
                'mixer': f"octave_{octave}_3"
            },
            'intermediate_frequency': qubit_multi_IF,
            'operations': {
                'CW': 'CW',
                'saturation': 'square_pulse',
                'saturation_multi': 'saturation_pulse_multi',
                'gaussian': 'gaussian_pulse',
                'test': 'test_pulse',
                'pi': 'gaussian_pi_pulse',
                'marker': 'marker_pulse',
            },
            'digitalInputs': {
                "Switch": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },

        },

        'resonator': {
            'mixInputs': {
                'I': ('con1', 3),
                'Q': ('con1', 4),
                'lo_frequency': rr_LO,
                'mixer': f"octave_{octave}_2"
            },
            'intermediate_frequency': rr_IF,
            'operations': {
                'CW': 'CW',
                'saturation': 'square_pulse',
                'long_readout': 'long_readout_pulse',
                'readout': 'readout_pulse',
                'test':'test_readout_pulse',
    
            },
            "outputs": {
                'out1': ('con1', 1),
                'out2': ('con1', 2)
            },
            'time_of_flight': 24,
            'smearing': 0,
            'digitalInputs': {
                "Switch": {
                    "port": ("con1", 3),
                    "delay": 150,
                    "buffer": 15,
                },
            },
        },
		'Vsource': {
            'singleInput': {
                'port': ('con1', 7)
            },
            'operations': {
                'CW': 'port_pulse',
            },
            'hold_offset':{'duration': 200}

        },
		
    },

    "pulses": {

        "CW": {
            'operation': 'control',
            'length': 10000,
            'waveforms': {
                'I': 'const_wf',
                'Q': 'zero_wf'
            },
			"digital_marker":"ON",

        },
         "port_pulse": {
            'operation': 'control',
            'length': 10000,
            'waveforms': {
                'single': 'const_wf'
            }
        },
         "test_pulse": {
            'operation': 'control',
            'length': 10000,
            'waveforms': {
                'I': 'const_wf',
                'Q': 'zero_wf'
            }
        },

		"marker_pulse": {
            'operation': 'control',
            'length': 20,
            'waveforms': {
                'I': 'zero_wf',
                'Q': 'zero_wf'			
            },
			'digital_marker': 'ON',

        },
		
        "square_pulse": {
            'operation': 'control',
            'length': 50000,  # several T1s
            'waveforms': {
                'I': 'saturation_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
        },
        "saturation_pulse_multi": {
            'operation': 'control',
            'length': 50000,  # several T1s
            'waveforms': {
                'I': 'saturation_wf_multi',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
        },

        
        "gaussian_pulse": {
            'operation': 'control',
            'length': gauss_len,
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
        },

        'gaussian_pi_pulse': {
            'operation': 'control',
            'length': pi_pulse_length,
            'waveforms': {
                'I': 'pi_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
        },
        'square_pi_pulse': {
            'operation': 'control',
            'length': pi_pulse_length,  
            'waveforms': {
                'I': 'saturation_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
        },
        'gaussian_pi/2_pulse': {
            'operation': 'control',
            'length': pi_pulse_length_over_2,
            'waveforms': {
                'I': 'pi/2_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
        },
        'square_pi/2_pulse': {
            'operation': 'control',
            'length': 50000,  # several T1s
            'waveforms': {
                'I': 'saturation_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
        },
        # 'gaussian_pi/4_pulse': {
        #     'operation': 'control',
        #     'length': pi_pulse_length_over_4,
        #     'waveforms': {
        #         'I': 'pi/4_wf',
        #         'Q': 'zero_wf'
        #     },
        #     'digital_marker': 'ON',
        # },
        'square_pi/4_pulse': {
            'operation': 'control',
            'length': 50000,  # several T1s
            'waveforms': {
                'I': 'saturation_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
        },

        'long_readout_pulse': {
            'operation': 'measurement',
            'length': long_readout_len,
            'waveforms': {
                'I': 'long_readout_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'long_integW_cos': 'long_integW_cos',
                'long_integW_sin': 'long_integW_sin',
            },
			"digital_marker":"ON",
        },

        'readout_pulse': {
            'operation': 'measurement',
            'length': readout_len,
            'waveforms': {
                'I': 'readout_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'integW1': 'integW1',
                'integW2': 'integW2',
                'optW1': 'optW1',
                'optW2': 'optW2'
            },
			"digital_marker":"ON",
        },
        
        'test_readout_pulse': {
            'operation': 'measurement',
            'length': readout_len,
            'waveforms': {
                'I': 'readout_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'integW1': 'integW1',
                'integW2': 'integW2',
                'optW1': 'optW1',
                'optW2': 'optW2'
            },
			"digital_marker":"ON",
        },
		'long_empty_readout_pulse':{
            'operation': 'measurement',
            'length': long_readout_len,
            'waveforms': {
                'I': 'zero_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'long_integW_cos': 'long_integW_cos',
                'long_integW_sin': 'long_integW_sin',
            },
			"digital_marker":"ON",
        },

    },

    'waveforms': {

        'const_wf': {
            'type': 'constant',
            'sample': 0.4
        },

        'zero_wf': {
            'type': 'constant',
            'sample': 0.0
        },

        'saturation_wf': {
            'type': 'constant',
            'sample': 0.4
        },
        'saturation_wf_multi': {
            'type': 'constant',
            'sample': 0.15
        },
        'gauss_wf': {
            'type': 'arbitrary',
            'samples': gauss(0.4, 0.0, gauss_len//4, gauss_len)
        },
        'gauss_short_wf' : {
            'type' : 'arbitrary',
            'samples' : gauss(0.4, 0.0, gauss_len//4, gauss_len)
        },
        'pi_wf': {
            'type': 'arbitrary',
            'samples': gauss(0.4, 0.0, pi_pulse_time//4, pi_pulse_time) + [0]*int(16-pi_pulse_time)
        },
        'pi/2_wf' : {
            'type': 'arbitrary',
            'samples':gauss(0.4, 0.0, pi_pulse_time//8, pi_pulse_time/2) + [0]*int((16-pi_pulse_time/2))
        },
        # 'pi/4_wf' : {
        #     'type': 'arbitrary',
        #     'samples': gauss(0.4, 0.0, pi_pulse_length_over_4//4, pi_pulse_length_over_4) + [0]*int(16-pi_pulse_time/4)
        # },

        'long_readout_wf': {
            'type': 'constant',
            'sample': 0.4
        },

        'readout_wf': {
            'type': 'constant',
            'sample': 0.6
        },
    },

    'digital_waveforms': {
        'ON': {
            'samples': [(1, 0)]
        }
    },

    'integration_weights': {

        'long_integW_cos': {
            'cosine': [1.0] * int(long_readout_len),
            'sine': [0.0] * int(long_readout_len)
        },

        'long_integW_sin': {
            'cosine': [0.0] * int(long_readout_len),
            'sine': [1.0] * int(long_readout_len)
        },

        'integW1': {
            'cosine': [1.0] * int(readout_len),
            'sine': [0.0] * int(readout_len),
        },

        'integW2': {
            'cosine': [0.0] * int(readout_len),
            'sine': [1.0] * int(readout_len),
        },

        'optW1': {
            'cosine': [1.0] * int(readout_len),
            'sine': [0.0] * int(readout_len)
        },

        'optW2': {
            'cosine': [0.0] * int(readout_len),
            'sine': [1.0] * int(readout_len)
        },
    },

    "mixers": { #DO NOT CHANGE KEYS OF DICTIONARY, these are specific to the octave
        f"octave_{octave}_1": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": (1, 0, 0, 1),
            },
        ],
        f"octave_{octave}_2": [
            {
                "intermediate_frequency": rr_IF,
                "lo_frequency": rr_LO,
                "correction": (1, 0, 0, 1),
            },
        ],
        f"octave_{octave}_3": [
            {
                "intermediate_frequency": qubit_multi_IF,
                "lo_frequency": qubit_LO,
                "correction": (1, 0, 0, 1),
            },
        ],
        f"octave_{octave}_4": [
            {
                "intermediate_frequency": IF,
                "lo_frequency": LO,
                "correction": (1, 0, 0, 1),
            },
        ],
        f"octave_{octave}_5": [
            {
                "intermediate_frequency": IF,
                "lo_frequency": LO,
                "correction": (1, 0, 0, 1),
            },
        ],
    },
}
