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


def snail_buffer_cross_kerr(
    device_setup,
    serial_num,
    qubit_params_file_path,
    freq_swp,
    length=None,
    amplitude=None,
    exp_id="snail_buffer_cross_kerr_spectroscopy",
    average_exponent=5,
    acquisition_type=AcquisitionType.INTEGRATION,
    rotate_ro=False,
    thresholds=None,
    chunk_count=1,
    averaging_mode=AveragingMode.CYCLIC,
    alice_or_bob="alice",
    bs_drive_freq=None,
    prepare_superposition=True,
    transmon_buffer_chi=0.0,
):
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    readout_pulse = qubit_params_module.readout_pulse
    kernels = qubit_params_module.acquire_kernel
    bs = qubit_params_module.bs_transmon_snail_eg_ge
    bs_freq = (
        qubit_parameters["q0"]["bs_transmon_snail_eg_ge_freq"] + transmon_buffer_chi
    )

    if bs_drive_freq is not None:
        bs_freq = bs_drive_freq

    ge_X90 = qubit_params_module.ge_X90
    ge_X180 = qubit_params_module.ge_X180
    ge_pulse = ge_X90 if prepare_superposition else ge_X180
    ef_X180 = qubit_params_module.ef_X180

    sb = qubit_params_module.sb_pulses[alice_or_bob]["f0g1"]
    sb_signal = f"sb_drive_{alice_or_bob}_f0g1"

    snail_freq = (min(freq_swp.values) + max(freq_swp.values)) / 2
    snail_pulse = qubit_params_module.snail_01_X180

    if length is not None:
        snail_pulse.length = length
    if amplitude is not None:
        snail_pulse.amplitude = amplitude

    for freq, lo_name in zip([snail_freq, bs_freq], ["SG2_LO", "SG4_LO"]):
        lo_settings["q0"][serial_num][lo_name] = max(1e9, round(freq / 200e6) * 200e6)

    lo_snail = lo_settings["q0"][serial_num]["SG2_LO"]
    lo_bs = lo_settings["q0"][serial_num]["SG4_LO"]
    freq_swp = SweepParameter(
        uid=freq_swp.uid, values=np.asarray(freq_swp.values) - lo_snail
    )

    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("qb_ef_drive"),
            ExperimentSignal(sb_signal),
            ExperimentSignal("snail_drive"),
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

            with exp.section(uid="ge_excitation"):
                exp.play(signal="qb_drive", pulse=ge_pulse)

            with exp.section(
                uid="ef_excitation", play_after="ge_excitation", on_system_grid=True
            ):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)

            with exp.section(uid="sb_transition_f0g1", play_after="ef_excitation"):
                exp.play(signal=sb_signal, pulse=sb)

            with exp.section(uid="snail_excitation", play_after="sb_transition_f0g1"):
                exp.play(signal="snail_drive", pulse=snail_pulse)

            with exp.section(uid="map", play_after="snail_excitation"):
                exp.play(signal="bs", pulse=bs)

            with exp.section(uid="readout", play_after="map"):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_0",
                    reset_delay=qubit_parameters["q0"]["cavity_reset_delay"],
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
        "frequency": freq_swp,
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
