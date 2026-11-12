import numpy as np
from laboneq.simple import (
    AcquisitionType,
    AveragingMode,
    Experiment,
    ExperimentSignal,
    LinearSweepParameter,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def snail_ramsey(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="snail_ramsey",
    average_exponent=5,
    time_swp=LinearSweepParameter(uid="time_swp", start=0, stop=20e-6, count=101),
    acquisition_type=AcquisitionType.INTEGRATION,
    sw_detuning_freq=0,
    n_echoes=0,
    rotate_ro=False,
    thresholds=None,
    chunk_count=1,
    averaging_mode=AveragingMode.CYCLIC,
):
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    readout_pulse = qubit_params_module.readout_pulse
    kernels = qubit_params_module.acquire_kernel
    snail_X90 = qubit_params_module.snail_01_X90

    if n_echoes:
        snail_X180 = qubit_params_module.snail_01_X180

    bs = qubit_params_module.bs_transmon_snail_eg_ge
    snail_freq = qubit_parameters["q0"]["snail_01_freq"]
    bs_freq = qubit_parameters["q0"]["bs_transmon_snail_eg_ge_freq"]

    for freq, lo_name in zip([snail_freq, bs_freq], ["SG2_LO", "SG4_LO"]):
        lo_settings["q0"][serial_num][lo_name] = max(1e9, round(freq / 200e6) * 200e6)
    lo_snail = lo_settings["q0"][serial_num]["SG2_LO"]
    lo_bs = lo_settings["q0"][serial_num]["SG4_LO"]

    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("snail_drive"),
            ExperimentSignal("bs"),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    phase_swp = 2.0 * np.pi * time_swp * sw_detuning_freq
    with exp.acquire_loop_rt(
        uid="shots",
        count=pow(2, average_exponent),
        acquisition_type=acquisition_type,
        averaging_mode=AveragingMode(averaging_mode),
    ):
        with exp.sweep(
            uid="time_sweep",
            parameter=time_swp,
            reset_oscillator_phase=True,
            chunk_count=chunk_count,
        ):

            with exp.section(uid="ge_excitation"):
                exp.play(signal="qb_drive", pulse=ge_pulse)

            with exp.section(
                uid="ef_excitation_0", play_after="ge_excitation", on_system_grid=True
            ):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)

            with exp.section(uid="sb_transition_f0g1_0", play_after="ef_excitation_0"):
                exp.play(signal=sb_signal, pulse=sb)

            # if storage_cross_kerr:
            # swap to storage

            with exp.section(uid="snail_0"):
                exp.play(signal="snail_drive", pulse=snail_X90, phase=0)
                exp.delay(signal="snail_drive", time=time_swp)

            with exp.section(uid="sb_transition_f0g1_1", play_after="snail_0"):
                exp.play(signal=sb_signal, pulse=sb)

            with exp.section(
                uid="ef_excitation_1",
                play_after="sb_transition_f0g1_1",
                on_system_grid=True,
            ):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)

            with exp.section(uid="snail_1", play_after="ef_excitation_1"):
                exp.play(signal="snail_drive", pulse=snail_X90, phase=phase_swp)

            with exp.section(uid="map", play_after="snail_1"):
                exp.play(signal="bs", pulse=bs)

            with exp.section(uid="readout", play_after="map"):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_0",
                    reset_delay=5 * qubit_parameters["q0"]["snail_01_t1"],
                    acquire_delay=qubit_parameters["q0"]["acquire_delay"],
                )

    sig_freq_map = create_default_map_and_calibration(
        exp,
        serial_num,
        qubit_parameters,
        lo_settings,
        rotate_ro=rotate_ro,
        thresholds=thresholds,
    )
    sig_freq_map[serial_num]["SG3"]["snail_drive"] = {
        "frequency": snail_freq - lo_snail,
        "range": qubit_parameters["q0"]["snail_01_dBm_range"],
    }
    sig_freq_map[serial_num]["SG4"]["bs"] = {
        "frequency": bs_freq - lo_bs,
        "range": qubit_parameters["q0"]["bs_transmon_snail_eg_ge_dBm_range"],
    }
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

    return exp, lo_snail
