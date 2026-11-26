from laboneq.simple import (
    AcquisitionType,
    AveragingMode,
    Experiment,
    ExperimentSignal,
    pulse_library,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def snail_rabi(
    device_setup,
    serial_num,
    qubit_params_file_path,
    swp_param,
    snail_drive_freq=None,
    amplitude=None,
    exp_id="snail_rabi",
    average_exponent=5,
    acquisition_type=AcquisitionType.INTEGRATION,
    rotate_ro=False,
    thresholds=None,
    chunk_count=1,
    single_shot=False,
    by2=False,
):
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    readout_pulse = qubit_params_module.readout_pulse
    kernels = qubit_params_module.acquire_kernel
    bs = qubit_params_module.bs_transmon_snail_eg_ge
    bs_freq = qubit_parameters["q0"]["bs_transmon_snail_eg_ge_freq"]
    snail_pulse = (
        qubit_params_module.snail_01_X180
        if not by2
        else qubit_params_module.snail_01_by2_X180
    )

    if amplitude is not None:
        snail_pulse.amplitude = amplitude

    snail_range = (
        qubit_parameters["q0"]["snail_01_dBm_range"]
        if not by2
        else qubit_parameters["q0"]["snail_01_by2_dBm_range"]
    )

    config_freq = (
        qubit_parameters["q0"]["snail_01_freq"]
        if not by2
        else qubit_parameters["q0"]["snail_01_by2_freq"]
    )
    snail_drive_freq = config_freq if snail_drive_freq is None else snail_drive_freq

    for freq, lo_name in zip([snail_drive_freq, bs_freq], ["SG2_LO", "SG4_LO"]):
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

    if single_shot:
        averaging_mode = AveragingMode.SINGLE_SHOT
    else:
        averaging_mode = AveragingMode.CYCLIC

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
            with exp.section(uid="snail_excitation"):
                exp.play(signal="snail_drive", pulse=snail_pulse, length=swp_param)

            with exp.section(uid="map", play_after="snail_excitation"):
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
        "frequency": snail_drive_freq - lo_snail,
        "range": snail_range,
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
