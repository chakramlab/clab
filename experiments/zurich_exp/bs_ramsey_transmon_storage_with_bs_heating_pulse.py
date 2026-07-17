import numpy as np
from laboneq.simple import (
    AcquisitionType,
    Experiment,
    ExperimentSignal,
    LinearSweepParameter,
    pulse_library,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def bs_ramsey_transmon_storage_with_bs_heating_pulse(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="bs_ramsey_transmon_storage_with_bs_heating_pulse",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    time_swp=LinearSweepParameter(uid="time_swp", start=8e-9, stop=10e-6, count=10),
    bs_freq=None,
    bs_range=None,
    bs_length=None,
    bs_amplitude=None,
    bs_freq_off_resonant=None,
    bs_off_resonant_amplitude=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    alice_or_bob="alice",
    rotate_ro=False,
    thresholds=None,
    swp_amp=False,
    storage_mode=1,
    echo=False,
    sw_detuning_freq=0,
    chunk_count = 1
    ):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X90 = qubit_params_module.ge_X90

    # bs pulse
    bs = qubit_params_module.sb_pulses[alice_or_bob][f'bs{storage_mode}']
    if bs_length is None:
        bs_length = bs.length
    # if bs_ramp is None:
    #     bs_ramp = qubit_params_module.sb_pulses['alice'][f'bs{storage_mode}'].pulse_parameters['ramp']
    if bs_amplitude is None:
        bs_amplitude = bs.amplitude
    if bs_range is None:
        bs_range = qubit_parameters["q0"][f"bs_{alice_or_bob}_dBm_ranges"][storage_mode]
    if bs_freq is None:
        bs_freq = qubit_parameters["q0"][f"bs_{alice_or_bob}_freqs"][storage_mode]
    if bs_freq_off_resonant is None:
        bs_freq_off_resonant = bs_freq + 100e6  # 100 MHz detuning for off-resonant pulse
        print(f"Off-resonant BS frequency not provided. Using {bs_freq_off_resonant/1e9} GHz")
    if bs_off_resonant_amplitude is None:
        bs_off_resonant_amplitude = bs_amplitude



    lo = lo_settings["q0"][serial_num]["SG4_LO"]
    lo_range = 0.5e9
    if bs_freq < lo - lo_range or bs_freq > lo + lo_range:
        old_lo = lo
        new_lo = bs_freq
        step = 200e6
        new_lo = round(new_lo / step) * step
        lo_settings["q0"][serial_num]["SG4_LO"] = new_lo
        lo = new_lo
        print(f"Warning: LO frequency changed to {new_lo/1e9} GHz")
        lo_change=True
    if bs_freq_off_resonant < lo - lo_range or bs_freq_off_resonant > lo + lo_range:
        print(f"WARNING: Frequency for off-resonant BS pulse {bs_freq_off_resonant/1e9} GHz is out of range of LO {lo/1e9} GHz +/- {lo_range/1e9} GHz")

    # if swp_amp:
    #     bs_amplitude = swp_param
    # else:
    #     bs_length = swp_param

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("bs"),
            ExperimentSignal("bs_off_resonant"),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    # set up phase sweep
    phase_swp = 2 * np.pi * time_swp * sw_detuning_freq
    swp_param = time_swp if echo else [time_swp, phase_swp]

    with exp.acquire_loop_rt(
        uid="shots", 
        count=pow(2, average_exponent), 
        acquisition_type=acquisition_type
        ):

        with exp.sweep(
            uid="time_or_amp_sweep", parameter=swp_param, reset_oscillator_phase=True, chunk_count=chunk_count
            ):

            with exp.section(uid = "ge_excitation",  play_after=None,  on_system_grid=True):
                exp.play(signal = "qb_drive", pulse = ge_X90)

            with exp.section(uid="bs", play_after="ge_excitation", on_system_grid=True):
                exp.play(signal="bs", pulse=bs,
                         length=bs_length, amplitude=bs_amplitude
                         )
            with exp.section(uid="bs_off_resonant", play_after="bs", on_system_grid=True):
                exp.play(signal="bs_off_resonant", pulse=bs, length=swp_param if echo else swp_param[0], amplitude=bs_off_resonant_amplitude)
            with exp.section(uid="bs2", play_after="bs_off_resonant", on_system_grid=True):
                exp.play(signal="bs", pulse=bs,
                         length=bs_length, amplitude=bs_amplitude, phase = phase_swp
                         )

            with exp.section(uid = "ge_excitation_2", play_after= "bs2", on_system_grid=True):
                exp.play(signal = "qb_drive", pulse = ge_X90)

            with exp.section(uid="readout", play_after="ge_excitation_2", on_system_grid=True):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_0",
                    reset_delay=qubit_parameters["q0"]["cavity_reset_delay"],
                    acquire_delay=qubit_parameters["q0"]["acquire_delay"],
                )

    # setup calibration and signal map for the experiment
    sig_freq_map = create_default_map_and_calibration(
        exp,
        serial_num,
        qubit_parameters,
        lo_settings,
        rotate_ro=rotate_ro,
        thresholds=thresholds,
    )

    ch = "SG4"
    sig_freq_map[serial_num][ch]["bs"] = {}
    sig_freq_map[serial_num][ch]["bs"]["frequency"] = bs_freq - lo
    sig_freq_map[serial_num][ch]["bs"]["range"] = bs_range

    sig_freq_map[serial_num][ch]["bs_off_resonant"] = {}
    sig_freq_map[serial_num][ch]["bs_off_resonant"]["frequency"] = bs_freq_off_resonant - lo
    sig_freq_map[serial_num][ch]["bs_off_resonant"]["range"] = bs_range

    print(sig_freq_map[serial_num][ch]["bs"])

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
