import numpy as np
from laboneq.simple import (
    AcquisitionType,
    AveragingMode,
    Experiment,
    ExperimentSignal,
    SweepParameter,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def snail_transmon_f0g1_rabi(
    device_setup,
    serial_num,
    qubit_params_file_path,
    swp_param,
    exp_id="snail_transmon_f0g1_rabi",
    average_exponent=5,
    acquisition_type=AcquisitionType.INTEGRATION,
    rotate_ro=False,
    thresholds=None,
    chunk_count=1,
    averaging_mode=AveragingMode.CYCLIC,
    prepare_e=True,
    sideband_on=True,
):
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    readout_pulse = qubit_params_module.readout_pulse
    kernels = qubit_params_module.acquire_kernel
    ge_X180 = qubit_params_module.ge_X180
    sideband_pulse = qubit_params_module.sb_snail_transmon_f0g1
    sb_freq = qubit_parameters["q0"]["sb_snail_transmon_f0g1_freq"]

    lo = max(1e9, round(sb_freq / 200e6) * 200e6)
    lo_settings["q0"][serial_num]["SG4_LO"] = lo

    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("snail_sideband"),
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
            uid="rabi_sweep",
            parameter=swp_param,
            reset_oscillator_phase=True,
            chunk_count=chunk_count,
        ):
            with exp.section(uid="ge_preparation"):
                exp.play(
                    signal="qb_drive", pulse=ge_X180, amplitude=None if prepare_e else 0
                )

            with exp.section(uid="sideband_rabi", play_after="ge_preparation"):
                exp.play(
                    signal="snail_sideband",
                    pulse=sideband_pulse,
                    amplitude=None if sideband_on else 0,
                    length=swp_param,
                )

            with exp.section(uid="readout", play_after="sideband_rabi"):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_0",
                    reset_delay=5 * 2 * qubit_parameters["q0"]["snail_01_t1"],
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
    sig_freq_map[serial_num]["SG4"]["snail_sideband"] = {
        "frequency": sb_freq - lo,
        "range": qubit_parameters["q0"]["sb_snail_transmon_f0g1_dBm_range"],
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
    return exp, lo
