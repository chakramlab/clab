from laboneq.simple import pulse_library
import numpy as np

# a collection of qubit control and readout parameters as a python dictionary
def single_qubit_parameters():
    return {
        # qb drive settings
        "qb_freq": 207.5e6, # 4.874720467G qubit 0 drive frequency in [Hz] - relative to local oscillator
        "qb_pi_len": 55e-9,
        "qb_pi_amp": 1.,
        "qb_ef_freq": -40.67e6,
        "qb_ef_len": 39e-9,
        "qb_ef_amp": 1,
        "qb_drive_dBm_range": 1,
        "qb_resolved_pi_len": 40e-6,
        "qb_resolved_pi_amp": 0.01,
        "qb_resolved_pi_muted_amp": 0.1, # this is roughly X10 of "qb_resolved_pi_amp"
        
        # measurement settings
        "reset_delay": 200e-6,  # delay time after each measurement for qubit reset in [s]
        "cavity_reset_delay": 5e-6,
        'ro_len': 2e-6, # 2us sq
        'ro_freq': (7.6924054100525145 - 7.6)*1e9, # 7.783671267760919
        'ro_amp': 0.9, #
        "ro_delay": 0e-9,
        "ro_int_delay": 0e-9, # time of flight
        "ro_drive_dBm_range": -10,
        "ro_acq_dBm_range": -5,

        # sideband pulse settings
        "sb_f0g1_alice_freq": 100e6, #
        "sb_f0g1_alice_flat_len": 3.e-6, #
        "sb_f0g1_alice_amp": 1,
        "sb_f1g2_alice_freq":  100e6, 
        "sb_f1g2_alice_flat_len": 2e-6,
        "sb_f1g2_alice_amp": 0.6,
        "sb_f2g3_alice_freq": 100e6 - 2*0.6e6, 
        "sb_f2g3_alice_flat_len": 2.5e-6,
        "sb_f2g3_alice_amp": 1,
        "sb_f3g4_alice_freq": 100e6 - 3*0.6e6, 
        "sb_f3g4_alice_flat_len": 2.5e-6,
        "sb_f3g4_alice_amp": 1,
        "sb_f4g5_alice_freq": 100e6 - 4*0.6e6,
        "sb_f4g5_alice_flat_len": 2.5e-6,
        "sb_f4g5_alice_amp": 1,
        "sb_f5g6_alice_freq": 100e6 - 5*0.6e6,
        "sb_f5g6_alice_flat_len": 2.5e-6,
        "sb_f5g6_alice_amp": 1,
        "sb_f6g7_alice_freq": 100e6 - 6*0.6e6,
        "sb_f6g7_alice_flat_len": 2.5e-6,
        "sb_f6g7_alice_amp": 1,
        "sb_f7g8_alice_freq": 100e6 - 7*0.6e6,
        "sb_f7g8_alice_flat_len": 2.5e-6,
        "sb_f7g8_alice_amp": 1,
        "sb_alice_ramp_len": 160e-9, #
        "sb_alice_dBm_range": 10,

        "sb_bob_ramp_len": 8e-9, #
        "sb_bob_dBm_range": 0,
        
        # cavity drive settings
        "cav_alice_freq": 100e6, #
        "cav_alice_len": 1.5e-6, # 20ns - 2us gaus/arb
        "cav_alice_amp": 0.9, #
        "cav_alice_dBm_range": 10,
        "cav_bob_freq": 100e6, # 
        "cav_bob_len": 200e-9, # 20ns - 2us gaus/arb
        "cav_bob_amp": 0.9,
        "cav_bob_dBm_range": 10,

        # other
        "chi": 100e3,
        "parity_time": 2e-6,
        # "readout_depletion_time": 1.5e-6,
    }

def shfqc_lo_settings(serial_num): # Need to be in multiples of 200 MHz
    return {
        serial_num: {
            # SHFQA LO Frequency
            "QA0_LO": 7.6e9,#
            # SHFSG LO Frequencies, one center frequency per two channels on SHFQC
            "SG0_LO": 4.0e9,#
            "SG2_LO": 1e9,#
            "SG4_LO": 1e9
        }
    }

def create_lo_settings(serial_num):
    return {"q0": shfqc_lo_settings(serial_num)}

def create_qubit_parameters():
    return {"q0": single_qubit_parameters()}

qubit_parameters = create_qubit_parameters()

########################################################################################
## Pulse definitions
########################################################################################

# def gaussian_second_half(x, **pulse_params):
#     sigma=1 / 3
#     return np.exp(-((x+1.)**2 / (2 * sigma**2)))
# def gaussian_first_half(x, **pulse_params):
#     sigma=1 / 3
#     return np.exp(-((x-1.)**2 / (2 * sigma**2)))

def gaussian_square_custom(
    x, sigma=1 / 3, ramp=10e-9, zero_boundaries=False, length=100e-9, **_
):
    """Create a gaussian square waveform with a Gaussian shaped ramp up/down portion
    of length ``ramp`` and a flat top.

    Arguments:
        **_ (Any):
            All pulses accept the following keyword arguments:
            - uid ([str][]): Unique identifier of the pulse
            - length ([float][]): Length of the pulse in seconds
            - amplitude ([float][]): Amplitude of the pulse
        ramp (float):
            Gaussian rise/fall length in seconds
        sigma (float):
            Std. deviation of the Gaussian rise/fall portion of the pulse
        zero_boundaries (bool):
            Whether to zero the pulse at the boundaries

    Returns:
        pulse (Pulse): Gaussian square pulse.
    """

    risefall_in_samples = round(len(x) * (1 - (length - ramp) / length) / 2)
    flat_in_samples = len(x) - 2 * risefall_in_samples
    gauss_x = np.linspace(-1.0, 1.0, 2 * risefall_in_samples)
    gauss_part = np.exp(-(gauss_x**2) / (2 * sigma**2))
    gauss_sq = np.concatenate(
        (
            gauss_part[:risefall_in_samples],
            np.ones(flat_in_samples),
            gauss_part[risefall_in_samples:],
        )
    )
    if zero_boundaries:
        t_left = gauss_x[0] - (gauss_x[1] - gauss_x[0])
        delta = np.exp(-(t_left**2) / (2 * sigma**2))
        gauss_sq -= delta
        gauss_sq /= 1 - delta
    return gauss_sq
custom_gaussian_square = pulse_library.register_pulse_functional(sampler = gaussian_square_custom, name = "gaussian_square_custom")

# # Basic pulse definitions
readout_pulse = pulse_library.const(uid = "ro_pulse", length = qubit_parameters['q0']['ro_len'], amplitude=qubit_parameters['q0']['ro_amp'])
acquire_kernel = pulse_library.const(uid="acquire_kernel", length=qubit_parameters["q0"]["ro_len"], amplitude=1.)
ge_X180 = pulse_library.gaussian(uid="ge_X180_pulse", length=qubit_parameters["q0"]["qb_pi_len"], amplitude=qubit_parameters["q0"]["qb_pi_amp"])
ge_X90 = pulse_library.gaussian(uid="ge_X90_pulse", length=qubit_parameters["q0"]["qb_pi_len"], amplitude=0.5*qubit_parameters["q0"]["qb_pi_amp"])
ef_X180 = pulse_library.gaussian(uid="ef_X180_pulse", length=qubit_parameters["q0"]["qb_ef_len"], amplitude=qubit_parameters["q0"]["qb_ef_amp"])
# resolved_X180 = pulse_library.gaussian_square(uid="resolved_X180_pulse", length = qubit_parameters["q0"]["qb_resolved_pi_flat_len"] + qubit_parameters["q0"]["qb_resolved_pi_ramp_len"], width = qubit_parameters["q0"]["qb_resolved_pi_flat_len"], amplitude = qubit_parameters["q0"]["qb_resolved_pi_amp"], can_compress = True)
resolved_X180 = pulse_library.gaussian(uid="resolved_X180_pulse", length = qubit_parameters["q0"]["qb_resolved_pi_len"], amplitude = qubit_parameters["q0"]["qb_resolved_pi_amp"])
resolved_X180_muted = pulse_library.gaussian(uid="resolved_X180_pulse_muted", length = qubit_parameters["q0"]["qb_resolved_pi_len"], amplitude = qubit_parameters["q0"]["qb_resolved_pi_muted_amp"])

# Definitions for the sideband pulses. I am assuming that the ramp times are constant
sb_pulses = {"alice":{}}#, "bob":{}}
# cooling_tone = {"alice":{}, "bob":{}}
for name in sb_pulses.keys():
    for st in range(8):
        # sb_pulses[name][f"f{st}g{st+1}"] = pulse_library.gaussian_square(uid=f"sb_f{st}g{st+1}_{name}_pulse", length = qubit_parameters["q0"][f"sb_f{st}g{st+1}_{name}_flat_len"] + qubit_parameters["q0"][f"sb_{name}_ramp_len"], width = qubit_parameters["q0"][f"sb_f{st}g{st+1}_{name}_flat_len"], amplitude = qubit_parameters["q0"][f"sb_f{st}g{st+1}_{name}_amp"], can_compress = True)
        sb_pulses[name][f"f{st}g{st+1}"] = custom_gaussian_square(uid=f"sb_f{st}g{st+1}_{name}_pulse", length = qubit_parameters["q0"][f"sb_f{st}g{st+1}_{name}_flat_len"] + qubit_parameters["q0"][f"sb_{name}_ramp_len"], ramp = qubit_parameters["q0"][f"sb_{name}_ramp_len"], amplitude = qubit_parameters["q0"][f"sb_f{st}g{st+1}_{name}_amp"], can_compress = True)
        # cooling_tone[name][f"f{st}g{st+1}"] = custom_gaussian_square(uid=f"sb_f{st}g{st+1}_{name}_cooling_pulse", length = qubit_parameters["q0"][f"sb_f{st}g{st+1}_{name}_flat_len"] + qubit_parameters["q0"][f"sb_{name}_ramp_len"], ramp = qubit_parameters["q0"][f"sb_{name}_ramp_len"], amplitude = qubit_parameters["q0"][f"sb_f{st}g{st+1}_{name}_amp"], can_compress = True)

# Cavity pulse definitions
cav_alice = pulse_library.gaussian(uid = "cav_alice_pulse", length = qubit_parameters["q0"]["cav_alice_len"], amplitude = qubit_parameters["q0"]["cav_alice_amp"])
cav_bob = pulse_library.gaussian(uid = "cav_bob_pulse", length = qubit_parameters["q0"]["cav_bob_len"], amplitude = qubit_parameters["q0"]["cav_bob_amp"])