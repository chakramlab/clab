from laboneq.simple import Experiment, ExperimentSignal, AveragingMode

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def calibrate(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="calibration",
    average_exponent=12,
    resolved_X180=False,
    averaging_mode=AveragingMode.CYCLIC
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel

    if resolved_X180:
        ge_X180 = qubit_params_module.resolved_X180
        qb_drive_signal = "qb_drive_resolved"
    else:
        ge_X180 = qubit_params_module.ge_X180
        qb_drive_signal = "qb_drive"
    # ef_X180 = qubit_params_module.ef_X180

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal(qb_drive_signal),
            # ExperimentSignal("qb_ef_drive"),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    with exp.acquire_loop_rt(
        uid="shots",
        count=pow(2, average_exponent),
        averaging_mode=averaging_mode,
    ):

        with exp.section(uid="readout_g"):
            exp.measure(
                measure_signal="measure",
                measure_pulse=readout_pulse,
                acquire_signal="acquire",
                integration_kernel=kernels,
                handle="ac_0_g",
                reset_delay=qubit_parameters["q0"]["reset_delay"],
                acquire_delay=qubit_parameters['q0']['acquire_delay']
            )

        with exp.section(uid="qubit_excitation", play_after="readout_g"):
            exp.play(signal=qb_drive_signal, pulse=ge_X180)

        with exp.section(uid="readout_e", play_after="qubit_excitation"):
            exp.measure(
                measure_signal="measure",
                measure_pulse=readout_pulse,
                acquire_signal="acquire",
                integration_kernel=kernels,
                handle="ac_0_e",
                reset_delay=qubit_parameters["q0"]["reset_delay"],
                acquire_delay=qubit_parameters['q0']['acquire_delay']
            )

        # with exp.section(uid="qubit_excitation", play_after="readout_e"):
        #     exp.play(signal=qb_drive_signal, pulse=ge_X180)
            
        # with exp.section(uid="qubit_ef_excitation", play_after="qubit_excitation"):
        #     exp.play(signal='qb_ef_drive', pulse=ef_X180)

        # with exp.section(uid="readout_f", play_after="qubit_ef_excitation"):
        #     exp.measure(
        #         measure_signal="measure",
        #         measure_pulse=readout_pulse,
        #         acquire_signal="acquire",
        #         integration_kernel=kernels,
        #         handle="ac_0_f",
        #         reset_delay=qubit_parameters["q0"]["reset_delay"],
        #     )

    # setup calibration and signal map for the experiment
    sig_freq_map = create_default_map_and_calibration(
        exp,
        serial_num,
        qubit_parameters,
        lo_settings,
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
