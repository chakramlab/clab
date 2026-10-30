from laboneq.simple import (
    AcquisitionType,
    AveragingMode,
    Experiment,
    ExperimentSignal,
    LinearSweepParameter,
    SectionAlignment,
    pulse_library,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def no_prep_shots(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="no_prep_shots",
    shots=10000,
    acquisition_type=AcquisitionType.INTEGRATION,
    averaging_mode=AveragingMode.SINGLE_SHOT,
    rotate_ro=False,
    thresholds=None,
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel

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
        count=shots,
        acquisition_type=acquisition_type,
        averaging_mode=averaging_mode,
    ):

        with exp.section(uid="readout", play_after=None, on_system_grid=True):
            exp.measure(
                measure_signal="measure",
                measure_pulse=readout_pulse,
                acquire_signal="acquire",
                integration_kernel=kernels,
                handle="ac_0",
                reset_delay=10e-6,
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
