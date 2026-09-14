from laboneq.simple import AcquisitionType, Experiment, ExperimentSignal

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def ge_pulse_train(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="ge_pulse_train",
    average_exponent=5,  # 2^n averages
    swp_param=None,  # Expecting a SweepParameter tracking integer pulse counts
    qb_drive_freq=None,
    qb_drive_amplitude=None,
    qb_drive_length=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    rotate_ro=False,
    thresholds=None,
    chunk_count=1,
    pulse_delay=0,
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X180 = qubit_params_module.ge_X180

    qb_drive_signal = "qb_drive"

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal(qb_drive_signal),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    if qb_drive_amplitude is not None:
        ge_X180.amplitude = 1
    if qb_drive_length is None:
        qb_drive_length = qubit_parameters["q0"]["qb_pi_len"]
    if qb_drive_freq is None:
        qb_drive_freq = qubit_parameters["q0"]["qb_freq"]

    with exp.acquire_loop_rt(
        uid="shots", count=pow(2, average_exponent), acquisition_type=acquisition_type
    ):

        with exp.sweep(
            uid="pulse_count_sweep",
            parameter=swp_param,
            reset_oscillator_phase=True,
            chunk_count=chunk_count,
        ):

            with exp.section(uid="ge_excitation", play_after=None, on_system_grid=True):
                with exp.match(sweep_parameter=swp_param):
                    for n in swp_param.values:
                        n_int = int(n)
                        with exp.case(n_int):
                            for _ in range(n_int):
                                exp.play(
                                    signal=qb_drive_signal,
                                    pulse=ge_X180,
                                    length=qb_drive_length,
                                    amplitude=(
                                        qb_drive_amplitude
                                        if qb_drive_amplitude is not None
                                        else None
                                    ),
                                )
                                (
                                    exp.delay(signal=qb_drive_signal, time=pulse_delay)
                                    if _ < n_int - 1
                                    else None
                                )

            with exp.section(uid="readout", play_after="ge_excitation"):
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

    if qb_drive_freq is not None:
        sig_freq_map[serial_num]["SG0"][qb_drive_signal]["frequency"] = (
            qb_drive_freq - lo_settings["q0"][serial_num]["SG0_LO"]
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
