import numpy as np
from laboneq.simple import (
    AcquisitionType,
    Experiment,
    ExperimentSignal,
    LinearSweepParameter,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def ramsey_ge(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="ramsey_ge",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    time_swp=LinearSweepParameter(uid="time_swp", start=8e-9, stop=10e-6, count=10),
    acquisiiton_type=AcquisitionType.INTEGRATION,  # check spelling
    sw_detuning_freq=0,
    echo=False,
    n_echoes=1,
    rotate_ro=False,
    thresholds=None,
    chunk_count=1,
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    ge_X90 = qubit_params_module.ge_X90
    ge_X180 = qubit_params_module.ge_X180
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    phase_swp = 2 * np.pi * time_swp * sw_detuning_freq
    sweep_param = time_swp if echo else [time_swp, phase_swp]
    with exp.acquire_loop_rt(
        uid="shots", count=pow(2, average_exponent), acquisition_type=acquisiiton_type
    ):
        with exp.sweep(
            uid="time_sweep", parameter=sweep_param, reset_oscillator_phase=True, chunk_count=chunk_count
        ):
            with exp.section(uid="qubit_excitation"):
                exp.play(signal="qb_drive", pulse=ge_X90)                
                if echo:
                    for _ in range(n_echoes):
                        exp.delay(signal="qb_drive", time=time_swp/(n_echoes+1))
                        exp.play(signal="qb_drive", pulse=ge_X180, phase=np.pi/2)
                    exp.delay(signal="qb_drive", time=time_swp/(n_echoes+1))                                
                else:
                    exp.delay(signal="qb_drive", time=time_swp)
                exp.play(signal="qb_drive", pulse=ge_X90, phase=phase_swp)
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
