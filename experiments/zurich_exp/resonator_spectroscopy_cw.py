from laboneq.simple import (
    AcquisitionType,
    Experiment,
    ExperimentSignal,
    LinearSweepParameter,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def resonator_spectroscopy_cw(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="resonator_spectroscopy_cw",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    integration_time=2e-6,
    freq_swp=LinearSweepParameter(
        uid="freq_swp_param", start=300e6, stop=400e6, count=101
    ),
    # kernels = acquire_kernel,
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]

    lo = lo_settings["q0"][serial_num]['QA0_LO']
    freq_swp.start -= lo
    freq_swp.stop -= lo

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )
    with exp.acquire_loop_rt(
        uid="shots",
        count=pow(2, average_exponent),
        acquisition_type=AcquisitionType.SPECTROSCOPY,
    ):
        with exp.sweep(uid="spect_sweep", parameter=freq_swp):
            with exp.section(uid="spectroscopy"):
                exp.measure(
                    acquire_signal="acquire",
                    integration_length=integration_time,
                    handle="ac_0",
                    # measure_pulse = readout_pulse,
                )
                # exp.measure(measure_signal = "measure", measure_pulse = readout_pulse,
                # acquire_signal = "acquire", integration_kernel = kernels, handle = "ac_0",
                # reset_delay = qubit_parameters["q0"]["reset_delay"]
                # )

            with exp.section(uid="delay", play_after="spectroscopy", length=1e-6):
                exp.reserve(signal="measure")
                exp.reserve(signal="acquire")

    # setup calibration and signal map for the experiment
    sig_freq_map = create_default_map_and_calibration(exp, serial_num, qubit_parameters, lo_settings)
    # update the frequency to incorporate the frequency sweep
    sig_freq_map[serial_num]["QA0"]["measure/acquire"]["frequency"] = freq_swp

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
