import numpy as np
from laboneq.simple import (
    AcquisitionType,
    AveragingMode,
    Experiment,
    ExperimentSignal,
    SweepParameter,
    pulse_library,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def snail_12_spectroscopy(
    device_setup,
    serial_num,
    qubit_params_file_path,
    freq_swp,
    length=None,
    amplitude=None,
    exp_id="snail_12_spectroscopy",
    average_exponent=5,
    acquisition_type=AcquisitionType.INTEGRATION,
    rotate_ro=False,
    thresholds=None,
    chunk_count=1,
    averaging_mode=AveragingMode.CYCLIC,
    by2=False,
):
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    readout_pulse = qubit_params_module.readout_pulse
    kernels = qubit_params_module.acquire_kernel
    bs = qubit_params_module.bs_transmon_snail_eg_ge
    bs_freq = qubit_parameters["q0"]["bs_transmon_snail_eg_ge_freq"]

    snail_01_freq = qubit_parameters["q0"]["snail_01_by2_freq"]
    snail_01_pulse = (
        qubit_params_module.snail_01_X180
        if not by2
        else qubit_params_module.snail_01_by2_X180
    )
    snail_12_freq = (min(freq_swp.values) + max(freq_swp.values)) / 2
    snail_12_pulse = (
        qubit_params_module.snail_12_X180
        if not by2
        else qubit_params_module.snail_12_by2_X180
    )

    snail_01_range = (
        qubit_parameters["q0"]["snail_01_dBm_range"]
        if not by2
        else qubit_parameters["q0"]["snail_01_by2_dBm_range"]
    )
    snail_12_range = (
        qubit_parameters["q0"]["snail_12_dBm_range"]
        if not by2
        else qubit_parameters["q0"]["snail_12_by2_dBm_range"]
    )

    if length is not None:
        snail_12_pulse.length = length

    if amplitude is not None:
        snail_12_pulse.amplitude = amplitude

    for freq, lo_name in zip(
        [snail_01_freq, snail_12_freq, bs_freq], ["SG2_LO", "SG2_LO", "SG4_LO"]
    ):
        lo_settings["q0"][serial_num][lo_name] = max(1e9, round(freq / 200e6) * 200e6)
    lo_snail_01 = lo_settings["q0"][serial_num]["SG2_LO"]
    lo_snail_12 = lo_settings["q0"][serial_num]["SG2_LO"]
    lo_bs = lo_settings["q0"][serial_num]["SG4_LO"]

    freq_swp = SweepParameter(
        uid=freq_swp.uid, values=np.asarray(freq_swp.values) - lo_snail_12
    )

    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("snail_01_drive"),
            ExperimentSignal("snail_12_drive"),
            ExperimentSignal("bs"),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    with exp.acquire_loop_rt(
        uid="shots",
        count=pow(2, average_exponent),
        acquisition_type=acquisition_type,
        averaging_mode=AveragingMode(averaging_mode),
    ):
        with exp.sweep(
            uid="spectroscopy_sweep",
            parameter=freq_swp,
            reset_oscillator_phase=True,
            chunk_count=chunk_count,
        ):
            with exp.section(
                uid="snail_01_excitation", play_after=None, on_system_grid=True
            ):
                exp.play(signal="snail_01_drive", pulse=snail_01_pulse)

            with exp.section(
                uid="snail_12_excitation",
                play_after="snail_01_excitation",
                on_system_grid=True,
            ):
                exp.play(signal="snail_12_drive", pulse=snail_12_pulse)

            with exp.section(uid="map", play_after="snail_12_excitation"):
                exp.play(signal="bs", pulse=bs)

            with exp.section(uid="readout", play_after="map"):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_0",
                    reset_delay=10 * qubit_parameters["q0"]["snail_01_t1"],
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
    sig_freq_map[serial_num]["SG3"]["snail_01_drive"] = {
        "frequency": snail_01_freq - lo_snail_01,
        "range": snail_01_range,
    }
    sig_freq_map[serial_num]["SG3"]["snail_12_drive"] = {
        "frequency": freq_swp,
        "range": snail_12_range,
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

    return exp, lo_snail_12
