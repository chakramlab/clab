import numpy as np
from laboneq.simple import (
    AcquisitionType,
    Experiment,
    ExperimentSignal,
    LinearSweepParameter,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import (
    default_signal_map_and_calibration,
)
from .load_qubit_params import load_qubit_params


def sideband_buffer_ramsey(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="sideband_buffer_ramsey",
    average_exponent = 5,  # 2^n averages, n=average_exponent, maximum: n = 17
    time_swp=LinearSweepParameter(uid="time_swp", start=8e-9, stop=10e-6, count=10),
    sw_detuning_freq=0,
    acquisition_type = AcquisitionType.INTEGRATION,
    alice_or_bob="alice",
    max_fock_state = 1, ## only works for 1 now
    rotate_ro = False,
    thresholds = None,
    chunk_count = 1,
    ):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X90 = qubit_params_module.ge_X90
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

    pulse = sb_f0g1_alice if alice_or_bob=="alice" else sb_f0g1_bob

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

    phase_swp = 2 * np.pi * time_swp * sw_detuning_freq
    sweep_param = [time_swp, phase_swp]

    with exp.acquire_loop_rt(
        uid="shots", 
        count=pow(2, average_exponent),
        acquisition_type=acquisition_type,
    ):
        # with exp.sweep(uid="spect_sweep", parameter = freq_swp, reset_oscillator_phase = True):
        with exp.sweep(
            uid="time_or_amp_sweep", parameter=sweep_param, reset_oscillator_phase=True, chunk_count=chunk_count):

            with exp.section(uid='transmon_ge_0', play_after=None, on_system_grid=True):
                exp.play(signal="qb_drive", pulse=ge_X90)

            with exp.section(uid='transmon_ef_0', play_after='transmon_ge_0', on_system_grid=True):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)

            with exp.section(uid='transmon_buffer', play_after='transmon_ef_0', on_system_grid=True):
                exp.play(signal=sb_drive_lines['f0g1'], pulse=pulse)
                exp.delay(signal=sb_drive_lines['f0g1'], time=time_swp)
                exp.play(signal=sb_drive_lines['f0g1'], pulse=pulse)

            with exp.section(uid='transmon_ef_1', play_after='transmon_buffer', on_system_grid=True):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)
            
            with exp.section(uid='transmon_ge_1', play_after='transmon_ef_1', on_system_grid=True):
                exp.play(signal="qb_drive", pulse=ge_X90, phase=phase_swp)

            with exp.section(uid = "readout", play_after = "transmon_ge_1", on_system_grid=True):
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

    return exp,lo