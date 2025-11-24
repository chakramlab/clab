from laboneq.simple import (
    Experiment,
    ExperimentSignal,
    LinearSweepParameter,
    pulse_library,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def qubit_ge_spectroscopy(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="qubit_ge_spectroscopy",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    length=100e-6,  # length of the square pulse
    freq_swp=LinearSweepParameter(
        uid="freq_swp_param", start=390e6, stop=420e6, count=11
    ),
    amplitude_factor=None,
    resolved=False
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X180 = qubit_params_module.ge_X180
    ef_X180 = qubit_params_module.ef_X180
    resolved_X180 = qubit_params_module.resolved_X180

    lo = lo_settings["q0"][serial_num]['SG0_LO']
    freq_swp.start -= lo
    freq_swp.stop -= lo

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    qubit_pulse = ge_X180 if not resolved else resolved_X180

    if amplitude_factor is None:
        amplitude_factor = 1

    with exp.acquire_loop_rt(
        uid="shots",
        count=pow(2, average_exponent),
        reset_oscillator_phase=True,
    ):
        with exp.sweep(
            uid="spect_sweep", parameter=freq_swp, reset_oscillator_phase=True
        ):
            with exp.section(uid="qubit_excitation"):
                exp.play(signal="qb_drive", pulse=qubit_pulse, amplitude=amplitude_factor, length=length)
            with exp.section(uid="readout", play_after="qubit_excitation"):                
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_0",
                    reset_delay=qubit_parameters["q0"]["reset_delay"],
                    acquire_delay=qubit_parameters['q0']['acquire_delay']
                )

    # setup calibration and signal map for the experiment
    sig_freq_map = create_default_map_and_calibration(exp, serial_num, qubit_parameters, lo_settings)
    
    # update the frequency to incorporate the frequency sweep
    sig_freq_map[serial_num]["SG0"]["qb_drive"]["frequency"] = freq_swp

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
