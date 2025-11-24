from laboneq.simple import AcquisitionType, AveragingMode, Experiment, ExperimentSignal

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def iq_blobs(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="iq_blobs",
    average_exponent=12,
    acquire_delay=None,
    measure_f=False,
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X180 = qubit_params_module.ge_X180
    ef_X180 = qubit_params_module.ef_X180

    if acquire_delay is not None:
        assert acquire_delay <= readout_pulse.length, "Acquire delay exceeds readout pulse length"

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    with exp.acquire_loop_rt(
        uid="shots",
        count=pow(2, average_exponent),
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.SINGLE_SHOT,
    ):

        with exp.section(uid="readout_g"):
            exp.measure(
                measure_signal="measure",
                measure_pulse=readout_pulse,
                acquire_signal="acquire",
                integration_kernel=kernels,
                acquire_delay=acquire_delay,
                integration_length=readout_pulse.length-acquire_delay if acquire_delay is not None else readout_pulse.length,
                handle="ac_0",
                reset_delay=qubit_parameters["q0"]["reset_delay"],
            )

        with exp.section(uid="qubit_ge_excitation", play_after="readout_g"):
            exp.play(signal="qb_drive", pulse=ge_X180)

        with exp.section(uid="readout_e", play_after="qubit_ge_excitation"):
            exp.measure(
                measure_signal="measure",
                measure_pulse=readout_pulse,
                acquire_signal="acquire",
                integration_kernel=kernels,
                acquire_delay=acquire_delay,
                integration_length=readout_pulse.length-acquire_delay if acquire_delay is not None else readout_pulse.length,
                handle="ac_1",
                reset_delay=qubit_parameters["q0"]["reset_delay"],
            )

        if measure_f:

            with exp.section(uid="qubit_ef_excitation", play_after="readout_e"):
                exp.play(signal="qb_drive", pulse=ef_X180)

            with exp.section(uid="readout_f", play_after="qubit_ef_excitation"):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_2",
                    reset_delay=qubit_parameters["q0"]["reset_delay"],
                )

    # setup calibration and signal map for the experiment
    sig_freq_map = create_default_map_and_calibration(
        exp, serial_num, qubit_parameters, lo_settings
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