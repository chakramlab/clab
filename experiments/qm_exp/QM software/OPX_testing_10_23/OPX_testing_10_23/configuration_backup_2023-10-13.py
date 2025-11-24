import numpy as np
from set_octave import OctaveUnit, octave_declaration

######################
# AUXILIARY FUNCTIONS:
######################


def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    return [float(x) for x in gauss_wave]


def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1-g**2)*(2*c**2-1))
    return [float(N * x) for x in [(1-g)*c, (1+g)*s, (1-g)*s, (1+g)*c]]


################
# CONFIGURATION:
################

long_readout_len = 10000
readout_len = 5000

qubit_IF = 370e6
qubit_multi_IF=379e6
rr_IF =  30e6  # rr -> readout resonator
rr_LO= 7.235e9
qubit_LO = 6.2e9

gauss_len=200

# rr_freq = 7.26e9

config = {

    'version': 1,

    'controllers': {
        'con1': {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': 0.0},  #Voltage source I
                2: {'offset': 0.0},  #Voltage source Q
                3: {'offset': -0.005},  # RR I
                4: {'offset':  0.0005},  # RR Q
                5: {'offset': -0.019},
                6: {'offset': 0.001},
            },
            'digital_outputs': {
				1: {},
                3: {},
			},
            'analog_inputs': {
                1: {'offset': 0.17885652719067774,"gain_db":2},  # I
                2: {'offset': 0.0}   # Q
            }
        }
    },

    'elements': {

        'qubit': {
            'mixInputs': {
                'I': ('con1', 5),
                'Q': ('con1', 6),
                'lo_frequency': qubit_LO,
                'mixer': 'mixer_qubit'
            },
            'intermediate_frequency': qubit_IF,
            'operations': {
                'CW': 'CW',
                'saturation': 'saturation_pulse',
                'saturation_multi': 'saturation_pulse_multi',
                'gaussian': 'gaussian_pulse',
                'test':'test_pulse',
                'pi': 'pi_pulse',
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
        'qubit_multi': {
            'mixInputs': {
                'I': ('con1', 5),
                'Q': ('con1', 6),
                'lo_frequency': qubit_LO,
                'mixer': 'mixer_qubit_multi'
            },
            'intermediate_frequency': qubit_multi_IF,
            'operations': {
                'CW': 'CW',
                'saturation': 'saturation_pulse',
                'saturation_multi': 'saturation_pulse_multi',
                'gaussian': 'gaussian_pulse',
                'test': 'test_pulse',
                'pi': 'pi_pulse',
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
                'mixer': 'mixer_RR'
            },
            'intermediate_frequency': rr_IF,
            'operations': {
                'CW': 'CW',
                'saturation': 'saturation_pulse',
                'long_readout': 'long_readout_pulse',
                'readout': 'readout_pulse',
                'test':'test_readout_pulse',
            },
            "outputs": {
                'out1': ('con1', 1),
                'out2': ('con1', 2)
            },
            'time_of_flight': 200,
            'smearing': 0,
            'digitalInputs': {
                "Switch": {
                    "port": ("con1", 3),
                    "delay": 0,
                    "buffer": 0,
                },
            },
        },
		'Vsource': {
            'singleInput': {
                'port': ('con1', 1)
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
		
        "saturation_pulse": {
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
            }
        },

        'pi_pulse': {
            'operation': 'control',
            'length': 6000,
            'waveforms': {
                'I': 'pi_wf',
                'Q': 'zero_wf'
            }
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

        'pi_wf': {
            'type': 'arbitrary',
            'samples': gauss(0.3, 0.0, 6.0, 6000)
        },

        'long_readout_wf': {
            'type': 'constant',
            'sample': 0.4
        },

        'readout_wf': {
            'type': 'constant',
            'sample': 0.2
        },
    },

    'digital_waveforms': {
        'ON': {
            'samples': [(1, 0)]
        }
    },

    'integration_weights': {

        'long_integW_cos': {
            'cosine': [1.0] * int(long_readout_len / 4),
            'sine': [0.0] * int(long_readout_len / 4)
        },

        'long_integW_sin': {
            'cosine': [0.0] * int(long_readout_len / 4),
            'sine': [1.0] * int(long_readout_len / 4)
        },

        'integW1': {
            'cosine': [1.0] * int(readout_len / 4),
            'sine': [0.0] * int(readout_len / 4),
        },

        'integW2': {
            'cosine': [0.0] * int(readout_len / 4),
            'sine': [1.0] * int(readout_len / 4),
        },

        'optW1': {
            'cosine': [1.0] * int(readout_len / 4),
            'sine': [0.0] * int(readout_len / 4)
        },

        'optW2': {
            'cosine': [0.0] * int(readout_len / 4),
            'sine': [1.0] * int(readout_len / 4)
        },
    },

    'mixers': {
        'mixer_qubit': [
            {'intermediate_frequency': qubit_IF, 'lo_frequency': qubit_LO,
             'correction': IQ_imbalance(-.06,-0.13)},
        ],
        'mixer_qubit_multi': [
            {'intermediate_frequency': qubit_multi_IF, 'lo_frequency': qubit_LO,
             'correction': IQ_imbalance(0, 0)},
        ],
        'mixer_RR': [
            {'intermediate_frequency': rr_IF, 'lo_frequency': rr_LO,
              'correction': IQ_imbalance(0.00,-0.065)}

        ],
    }
}


############################
# Set octave configuration #
############################
# Custom port mapping example
port_mapping = {
    ("con1", 1): ("octave1", "I1"),
    ("con1", 2): ("octave1", "Q1"),
    ("con1", 3): ("octave1", "I2"),
    ("con1", 4): ("octave1", "Q2"),
    ("con1", 5): ("octave1", "I3"),
    ("con1", 6): ("octave1", "Q3"),
    ("con1", 7): ("octave1", "I4"),
    ("con1", 8): ("octave1", "Q4"),
    ("con1", 9): ("octave1", "I5"),
    ("con1", 10): ("octave1", "Q5"),
}
# The Octave port is 11xxx, where xxx are the last three digits of the Octave internal IP that can be accessed from
# the OPX admin panel if you QOP version is >= QOP220. Otherwise, it is 50 for Octave1, then 51, 52 and so on.
octave_1 = OctaveUnit("octave1", qop_ip, port=11050, con="con1", clock="Internal", port_mapping="default")
# octave_2 = OctaveUnit("octave2", qop_ip, port=11051, con="con1", clock="Internal", port_mapping=port_mapping)

# Add the octaves
octaves = [octave_1]
# Configure the Octaves
octave_config = octave_declaration(octaves)

