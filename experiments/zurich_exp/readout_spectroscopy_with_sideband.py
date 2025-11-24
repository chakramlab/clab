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
    pulse_library
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import (
    default_signal_map_and_calibration,
    parsing_single_alice_bob_cav_and_sb_transitions,
)
from .load_qubit_params import load_qubit_params


def readout_spectroscopy_with_sideband(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="cavity_spectroscopy",
    average_exponent = 5,  # 2^n averages, n=average_exponent, maximum: n = 17
    max_fock_state = 1, ## only works for 1 now
    freq_swp=LinearSweepParameter(
        uid="freq_swp_param",
        start=-700e6,
        stop=700e6,
        count=101,
    ),
    amp_swp=LinearSweepParameter(
        uid="amp_swp_param",
        start=1,
        stop=1,
        count=1,
    ),
    sb_length=1e-6,
    sb_range=0,
    sb_amplitude=1,
    alice_or_bob="a",
    kernels = None,
    rotate_ro = False,
    thresholds = None,
    sb_freq = 3e9,
    sb = True
    ):


    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X180 = qubit_params_module.ge_X180
    ef_X180 = qubit_params_module.ef_X180
    cav_alice = qubit_params_module.cav_alice
    cav_bob = qubit_params_module.cav_bob
    sb_f0g1_alice = qubit_params_module.sb_pulses["alice"]["f0g1"]
    sb_f0g1_bob   = qubit_params_module.sb_pulses["bob"]["f0g1"]
    readout_pulse_length = qubit_params_module.readout_pulse.length


    lo = lo_settings["q0"][serial_num]['QA0_LO']
    freq_swp.start -= lo
    freq_swp.stop -= lo


    cav_name, dummy, dummy, dummy, cav_line, cav_pulse = (
        parsing_single_alice_bob_cav_and_sb_transitions(
            alice_or_bob=alice_or_bob,
            # sb_pulses=sb_pulses,
            cav_alice=cav_alice,
            cav_bob=cav_bob,
        )
    )


    transitions = [f"f{i}g{i+1}" for i in range(max_fock_state)]
    sb_drive_lines = {}
    sb_drive_pulses = {}
    for transition in transitions:
        if alice_or_bob == "alice":
            sb_drive_lines[transition] = f"sb_drive_alice_{transition}"
        else:
            sb_drive_lines[transition] = f"sb_drive_bob_{transition}"
        

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

    # the amplitude modified by 'sweep' will multiply to the default length. So here, we set the default to 1.
    readout_pulse_swp = pulse_library.const(
        uid="ro_pulse", length=readout_pulse_length, amplitude=1.0
    )

    with exp.sweep(uid="amp_sweep", parameter=amp_swp):
        with exp.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            acquisition_type=AcquisitionType.SPECTROSCOPY,
        ):
            with exp.sweep(uid="spect_sweep", parameter=freq_swp):
                if sb:
                    with exp.section(uid = "ge_excitation",  play_after=None): #alignment=SectionAlignment.RIGHT):
                        exp.play(signal = "qb_drive", pulse = ge_X180)
                    with exp.section(uid = "ef_excitation", play_after= "ge_excitation", on_system_grid=True):
                        exp.play(signal = "qb_ef_drive", pulse = ef_X180)
                    with exp.section(uid = "sb_transition_f0g1", play_after = "ef_excitation"):
                        exp.play(signal = sb_drive_lines["f0g1"], pulse = sb_f0g1_alice if alice_or_bob=="alice" else sb_f0g1_bob, 
                                 amplitude=sb_amplitude)
                    with exp.section(uid = "fe_transition", play_after = "sb_transition_f0g1"):
                        exp.play(signal = "qb_ef_drive", pulse = ef_X180)
                    # with exp.section(uid = "eg_transition",  play_after = "fe_transition", on_system_grid=True):
                    #     exp.play(signal = "qb_drive", pulse = ge_X180)
                    last_pulse = 'fe_transition'
                    # last_pulse = 'sb_transition_f0g1'
                else:
                    last_pulse = None

                with exp.section(uid = "readout", play_after = last_pulse):
                    exp.measure(measure_signal = "measure", 
                                measure_pulse = readout_pulse,
                                acquire_signal = "acquire", 
                                integration_kernel = kernels, 
                                handle = "ac_0",
                                reset_delay = qubit_parameters["q0"]["cavity_reset_delay"],
                                acquire_delay=0.5e-6)

    # setup calibration and signal map for the experiment
    sig_freq_map = create_default_map_and_calibration(exp, 
                                                      serial_num, 
                                                      qubit_parameters,
                                                      lo_settings,
                                                      rotate_ro=rotate_ro, 
                                                      thresholds=thresholds,
                                                      )
    if sb:
        sb_lo = lo_settings["q0"][serial_num]["SG0_LO"]
        ch = "SG1"
        sig_freq_map[serial_num][ch] = {}
        sig_freq_map[serial_num][ch][sb_drive_lines["f0g1"]] = {}
        sig_freq_map[serial_num][ch][sb_drive_lines["f0g1"]]["frequency"] = sb_freq - sb_lo
        sig_freq_map[serial_num][ch][sb_drive_lines["f0g1"]]["amplitude"] = sb_amplitude #amp_swp
        sig_freq_map[serial_num][ch][sb_drive_lines["f0g1"]]["range"] = sb_range
        sig_freq_map[serial_num][ch][sb_drive_lines["f0g1"]]["length"] = sb_length
    else:
        pass

    sig_freq_map[serial_num]["QA0"]["measure/acquire"]["frequency"] = freq_swp


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