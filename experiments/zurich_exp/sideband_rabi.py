# # LabOne Q (to be deleted after testing):
# from laboneq.simple import *

# # General imports
# import matplotlib.pyplot as plt
# import numpy as np
# # from contextlib import nullcontext
# import sys
# import os
# current_directory = os.getcwd()
# parent_directory = os.path.dirname(current_directory)
# sys.path.append(parent_directory)

# # Custom helper file
# from helper_files.experiment_chunks import *
# from helper_files.laboneq_helper import *
# from settings_and_parameters.calib_settings import *

from laboneq.simple import (
    AcquisitionType,
    Experiment,
    ExperimentSignal,
    LinearSweepParameter,
    pulse_library,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import (
    default_signal_map_and_calibration,
    parsing_single_alice_bob_cav_and_sb_transitions,
)
from .load_qubit_params import load_qubit_params


def sideband_rabi(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="cavity_rabi",
    average_exponent = 5,  # 2^n averages, n=average_exponent, maximum: n = 17
    max_fock_state = 1, ## only works for 1 now
    swp_param=LinearSweepParameter(
        uid="swp_param", start=1e-9, stop=10e-6, count=6
    ),
    sb_length=None,
    sb_range=None,
    sb_amplitude=None,
    swp_amp=False,
    acquisition_type = AcquisitionType.INTEGRATION,
    alice_or_bob="alice",
    kernels = None,
    rotate_ro = False,
    thresholds = None,
    sb_freq = None,
    prepare_f=True,
    chunk_count = 1,
    ):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X180 = qubit_params_module.ge_X180
    ef_X180 = qubit_params_module.ef_X180
    sb_f0g1_alice = qubit_params_module.sb_pulses["alice"]["f0g1"]
    sb_f0g1_bob   = qubit_params_module.sb_pulses["bob"]["f0g1"]

    # we're using channel 1 but we need to call the lo sg0
    lo = lo_settings["q0"][serial_num]["SG0_LO"]


    transitions = [f"f{i}g{i+1}" for i in range(max_fock_state)]
    sb_drive_lines = {}
    for transition in transitions:
        if alice_or_bob == "alice":
            sb_drive_lines[transition] = f"sb_drive_alice_{transition}"
        else:
            sb_drive_lines[transition] = f"sb_drive_bob_{transition}"
    if sb_length is None:
        sb_length = sb_f0g1_alice.length if alice_or_bob=="alice" else sb_f0g1_bob.length
    if sb_amplitude is not None:
        if alice_or_bob=="alice":
            sb_f0g1_alice.amplitude = 1
        else:
            sb_f0g1_bob.amplitude = 1
    if sb_range is None:
        sb_range = qubit_parameters["q0"][f"sb_{alice_or_bob}_dBm_range"]
    if sb_freq is None:
        sb_freq = qubit_parameters["q0"][f"sb_{alice_or_bob}_freqs"][0]

    if swp_amp:
        sb_amplitude = swp_param
    else:
        sb_length = swp_param



    # Create Experiment
    exp = Experiment(uid = exp_id,
        signals = [
            ExperimentSignal("qb_drive"),
            ExperimentSignal("qb_ef_drive"),
            *[ExperimentSignal(sb_drive_lines[_]) for _ in transitions],
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )
    with exp.acquire_loop_rt(
        uid="shots", 
        count=pow(2, average_exponent),
        acquisition_type=acquisition_type,
    ):
        # with exp.sweep(uid="spect_sweep", parameter = freq_swp, reset_oscillator_phase = True):
        with exp.sweep(
            uid="time_or_amp_sweep", parameter=swp_param, reset_oscillator_phase=True, chunk_count=chunk_count):

            if prepare_f:
                with exp.section(uid = "ge_excitation",  play_after=None): #alignment=SectionAlignment.RIGHT):
                    exp.play(signal = "qb_drive", pulse = ge_X180)

                with exp.section(uid = "ef_excitation", play_after= "ge_excitation", on_system_grid=True):
                    exp.play(signal = "qb_ef_drive", pulse = ef_X180)

            with exp.section(uid = "sb_transition_f0g1", play_after="ef_excitation" if prepare_f else None):
                exp.play(signal = sb_drive_lines["f0g1"], pulse = sb_f0g1_alice if alice_or_bob=="alice" else sb_f0g1_bob, 
                         length=sb_length, amplitude=sb_amplitude)

            with exp.section(uid = "fe_transition", play_after = "sb_transition_f0g1"):
                exp.play(signal = "qb_ef_drive", pulse = ef_X180)

            with exp.section(uid = "readout", play_after = "fe_transition"):
                exp.measure(measure_signal = "measure", measure_pulse = readout_pulse,
                            acquire_signal = "acquire", integration_kernel = kernels, handle = "ac_0",
                            reset_delay = qubit_parameters["q0"]["cavity_reset_delay"], 
                            acquire_delay=qubit_parameters['q0']['acquire_delay'])

    # setup calibration and signal map for the experiment
    sig_freq_map = create_default_map_and_calibration(exp, 
                                                      serial_num, 
                                                      qubit_parameters,
                                                      lo_settings,
                                                      rotate_ro=rotate_ro, 
                                                      thresholds=thresholds,
                                                      )
    # update the frequency to incorporate the frequency sweep
    ch = "SG1"
    sig_freq_map[serial_num][ch] = {}
    sig_freq_map[serial_num][ch][sb_drive_lines["f0g1"]] = {}
    sig_freq_map[serial_num][ch][sb_drive_lines["f0g1"]]["frequency"] = sb_freq - lo
    sig_freq_map[serial_num][ch][sb_drive_lines["f0g1"]]["range"] = sb_range

    exp_calibration, map_q0 = default_signal_map_and_calibration(
        sig_freq_map,
        {
            "device_setup": device_setup,
            "lo_settings": lo_settings,
            "qubit_parameters": qubit_parameters,
        },
    )
    exp.set_calibration(exp_calibration)
    exp.set_signal_map(map_q0)

    return exp