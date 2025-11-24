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


def resonator_spectroscopy(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="resonator_spectroscopy",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    freq_swp=LinearSweepParameter(
        uid="freq_swp_param",
        start=-700e6,
        stop=700e6,
        count=101,
    ),
    amp_swp=LinearSweepParameter(
        uid="amp_swp_param",
        start=1,
        stop=1,
        count=1,
    ),
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse_length = qubit_params_module.readout_pulse.length
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

    # the amplitude modified by 'sweep' will multiply to the default length. So here, we set the default to 1.
    readout_pulse_swp = pulse_library.const(
        uid="ro_pulse", length=readout_pulse_length, amplitude=1.0
    )

    with exp.sweep(uid="amp_sweep", parameter=amp_swp):
        with exp.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            acquisition_type=AcquisitionType.SPECTROSCOPY,
        ):
            with exp.sweep(uid="spect_sweep", parameter=freq_swp):
                with exp.section(uid="readout0"):
                    # exp.play(signal="measure", pulse=readout_pulse_swp, phase = np.pi/2, amplitude=amp_swp,)
                    # exp.acquire(signal="acquire",handle="ac_0",length=qubit_parameters['q0']['ro_len'],)
                    exp.measure(
                        measure_signal="measure",
                        measure_pulse=readout_pulse_swp,
                        measure_pulse_amplitude=amp_swp,
                        acquire_signal="acquire",
                        integration_length=qubit_parameters["q0"]["ro_len"],
                        handle="ac_0",
                        reset_delay=10e-6,
                    )
                # with exp.section(uid = "delay", play_after="readout0", length = 10e-6):
                #     pass

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
