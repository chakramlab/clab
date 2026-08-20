from laboneq.simple import Experiment, ExperimentSignal, LinearSweepParameter

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def rabi_ge(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="rabi_ge",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    amplitude_rabi=False,
    time_rabi=False,
    swp_param=LinearSweepParameter(
        uid="sweep_param",
        start=40e-9,
        stop=400e-9,
        count=11,
    ),
    chunk_count=1,
    rotate_ro=False,
    thresholds=None,
    amplitude_factor=None,
    qubit_drive_freq=None,
    length=None,
    resolved=False,
    chi=False,
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    qubit_pulse = (
        qubit_params_module.ge_X180
        if not resolved
        else qubit_params_module.resolved_X180
    )

    # shift qb_freq by chi
    if chi:
        qubit_parameters["q0"]["qb_resolved_freq"] += qubit_parameters["q0"]["chi"]

    qb_drive_signal = "qb_drive_resolved" if resolved else "qb_drive"

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal(qb_drive_signal),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    if not (time_rabi ^ amplitude_rabi):
        raise ValueError(
            "Please select either time_rabi or amplitude_rabi to be True and the other to be False."
        )

    if amplitude_factor is None:
        amplitude_factor = 1
    if length is None:
        length = qubit_pulse.length
    if qubit_drive_freq is None:
        qubit_drive_freq = (
            qubit_parameters["q0"]["qb_freq"]
            if not resolved
            else qubit_parameters["q0"]["qb_resolved_freq"]
        )

    with exp.acquire_loop_rt(
        uid="shots",
        count=pow(2, average_exponent),
    ):
        with exp.sweep(
            uid="time_or_amp_sweep",
            parameter=swp_param,
            reset_oscillator_phase=True,
            chunk_count=chunk_count,
        ):
            with exp.section(uid="qubit_excitation"):
                if time_rabi:
                    exp.play(
                        signal=qb_drive_signal,
                        pulse=qubit_pulse,
                        length=swp_param,
                        amplitude=amplitude_factor,
                    )
                elif amplitude_rabi:
                    exp.play(
                        signal=qb_drive_signal,
                        pulse=qubit_pulse,
                        amplitude=swp_param,
                        length=length,
                    )
            with exp.section(uid="readout", play_after="qubit_excitation"):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_0",
                    reset_delay=qubit_parameters["q0"]["reset_delay"],
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
    if qubit_drive_freq is not None:
        if not resolved:
            sig_freq_map[serial_num]["SG0"][qb_drive_signal]["frequency"] = (
                qubit_drive_freq - lo_settings["q0"][serial_num]["SG0_LO"]
            )
        else:
            sig_freq_map[serial_num]["SG5"][qb_drive_signal]["frequency"] = (
                qubit_drive_freq - lo_settings["q0"][serial_num]["SG4_LO"]
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

    return exp
