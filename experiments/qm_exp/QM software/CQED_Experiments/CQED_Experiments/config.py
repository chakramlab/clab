### Example config has been commented out          ###
### Anything not commented out is used universally ###


opx_ip = "192.168.88.252"
opx_port = 9510
octave_ip = "192.168.88.254"
octave_port = 80
con = "con1"
octave = "octave1"
# The elements used to test the ports of the Octave
elements = ["qe1", "qe2", "qe3", "qe4", "qe5"]
IF = 50e6  # The IF frequency
LO = 6e9  # The LO frequency

config = {
    "version": 1,
    "controllers": {
        con: {
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
    "elements": {
        # "qe1": {
        #     "mixInputs": {
        #         "I": (con, 1),
        #         "Q": (con, 2),
        #         "lo_frequency": LO,
        #         "mixer": f"octave_{octave}_1",  # a fixed name, do not change.
        #     },
        #     "intermediate_frequency": IF,
        #     "operations": {
        #         "cw": "const",
        #         "cw_wo_trig": "const_wo_trig",
        #         "readout": "readout_pulse",
        #     },
        #     "digitalInputs": {
        #         "switch": {
        #             "port": (con, 1),
        #             "delay": 136,
        #             "buffer": 0,
        #         },
        #     },
        #     "outputs": {
        #         "out1": (con, 1),
        #         "out2": (con, 2),
        #     },
        #     "time_of_flight": 224,#24,
        #     "smearing": 0,#50,
        # },
        # "qe2": {
        #     "mixInputs": {
        #         "I": (con, 3),
        #         "Q": (con, 4),
        #         "lo_frequency": LO,
        #         "mixer": f"octave_{octave}_2",  # a fixed name, do not change.
        #     },
        #     "intermediate_frequency": IF,
        #     "operations": {
        #         "cw": "const",
        #         "cw_wo_trig": "const_wo_trig",
        #         "readout": "readout_pulse",
        #     },
        #     "digitalInputs": {
        #         "switch": {
        #             "port": (con, 2),
        #             "delay": 136,
        #             "buffer": 0,
        #         },
        #     },
        #     "outputs": {
        #         "out1": (con, 1),
        #         "out2": (con, 2),
        #     },
        #     "time_of_flight": 24,
        #     "smearing": 0,
        # },
        # "qe3": {
        #     "mixInputs": {
        #         "I": (con, 5),
        #         "Q": (con, 6),
        #         "lo_frequency": LO,
        #         "mixer": f"octave_{octave}_3",  # a fixed name, do not change.
        #     },
        #     "intermediate_frequency": IF,
        #     "operations": {
        #         "cw": "const",
        #         "cw_wo_trig": "const_wo_trig",
        #     },
        #     "digitalInputs": {
        #         "switch": {
        #             "port": (con, 3),
        #             "delay": 136,
        #             "buffer": 0,
        #         },
        #     },
        # },
        # "qe4": {
        #     "mixInputs": {
        #         "I": (con, 7),
        #         "Q": (con, 8),
        #         "lo_frequency": LO,
        #         "mixer": f"octave_{octave}_4",  # a fixed name, do not change.
        #     },
        #     "intermediate_frequency": IF,
        #     "operations": {
        #         "cw": "const",
        #         "cw_wo_trig": "const_wo_trig",
        #     },
        #     "digitalInputs": {
        #         "switch": {
        #             "port": (con, 4),
        #             "delay": 136,
        #             "buffer": 0,
        #         },
        #     },
        # },
        # "qe5": {
        #     "mixInputs": {
        #         "I": (con, 9),
        #         "Q": (con, 10),
        #         "lo_frequency": LO,
        #         "mixer": f"octave_{octave}_5",  # a fixed name, do not change.
        #     },
        #     "intermediate_frequency": IF,
        #     "operations": {
        #         "cw": "const",
        #         "cw_wo_trig": "const_wo_trig",
        #     },
        #     "digitalInputs": {
        #         "switch": {
        #             "port": (con, 5),
        #             "delay": 136,
        #             "buffer": 0,
        #         },
        #     },
        # },
    },
    "pulses": {
        # "const": {
        #     "operation": "control",
        #     "length": 1000,
        #     "waveforms": {
        #         "I": "const_wf",
        #         "Q": "zero_wf",
        #     },
        #     "digital_marker": "ON",
        # },
        # "const_wo_trig": {
        #     "operation": "control",
        #     "length": 1000,
        #     "waveforms": {
        #         "I": "const_wf",
        #         "Q": "zero_wf",
        #     },
        # },
        # "readout_pulse": {
        #     "operation": "measurement",
        #     "length": 1000, #1000, in nanoseconds 
        #     "waveforms": {
        #         "I": "readout_wf",
        #         "Q": "zero_wf",
        #     },
        #     "integration_weights": {
        #         "cos": "cosine_weights",
        #         "sin": "sine_weights",
        #         "minus_sin": "minus_sine_weights",
        #     },
        #     "digital_marker": "ON",
        # },
    },
    "waveforms": {
        # "zero_wf": {
        #     "type": "constant",
        #     "sample": 0.0,
        # },
        # "const_wf": {
        #     "type": "constant",
        #     "sample": 0.1,#0.0325,
        # },
        # "readout_wf": {
        #     "type": "constant",
        #     "sample": 0.01, #should correspond to -30 dBm (0.02 Vpp) in part 3
        # },
    },
    "digital_waveforms": {
        # "ON": {"samples": [(1, 0)]},
        # "OFF": {"samples": [(0, 0)]},
    },
    "integration_weights": {
        # "cosine_weights": {
        #     "cosine": [(1.0, 1000)],
        #     "sine": [(0.0, 1000)],
        # },
        # "sine_weights": {
        #     "cosine": [(0.0, 1000)],
        #     "sine": [(1.0, 1000)],
        # },
        # "minus_sine_weights": {
        #     "cosine": [(0.0, 1000)],
        #     "sine": [(-1.0, 1000)],
        # },
    },
    "mixers": { #DO NOT CHANGE KEYS OF DICTIONARY, these are specific to the octave
        f"octave_{octave}_1": [
            {
                "intermediate_frequency": IF,
                "lo_frequency": LO,
                "correction": (1, 0, 0, 1),
            },
        ],
        f"octave_{octave}_2": [
            {
                "intermediate_frequency": IF,
                "lo_frequency": LO,
                "correction": (1, 0, 0, 1),
            },
        ],
        f"octave_{octave}_3": [
            {
                "intermediate_frequency": IF,
                "lo_frequency": LO,
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