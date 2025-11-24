from laboneq.simple import (
    AcquisitionType,
    Experiment,
    ExperimentSignal,
    LinearSweepParameter,
    pulse_library,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import (
    default_signal_map_and_calibration,
    parsing_single_alice_bob_cav_and_sb_transitions,
)
from .load_qubit_params import load_qubit_params


def bs_rabi_test(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="bs_rabi",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    swp_param=LinearSweepParameter(
        uid="swp_param", start=1e-9, stop=10e-6, count=50
    ),
    swp_amp=False,
    bs_freq=1e9,
    bs_range=10,
    length=1e-6,
    amplitude=1,
    acquisition_type=AcquisitionType.INTEGRATION,
    alice_or_bob="a",
    rotate_ro=False,
    thresholds=None,
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    resolved_X180 = qubit_params_module.resolved_X180
    cav_alice = qubit_params_module.cav_alice
    cav_bob = qubit_params_module.cav_bob

    cav_name, dummy, dummy, dummy, cav_line, cav_pulse = (
        parsing_single_alice_bob_cav_and_sb_transitions(
            alice_or_bob=alice_or_bob,
            # sb_pulses=sb_pulses,
            cav_alice=cav_alice,
            cav_bob=cav_bob,
        )
    )

    # test pulse
    bs_pulse = pulse_library.const(
        uid="bs_pulse", length=length, amplitude=amplitude, can_compress=True
    )

    # shift qb freq by chi
    qubit_parameters["q0"]["qb_freq"] += qubit_parameters["q0"]["chi"]

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("bs"),
            ExperimentSignal(cav_line),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )
    with exp.acquire_loop_rt(
        uid="shots", count=pow(2, average_exponent), acquisition_type=acquisition_type
    ):
        with exp.sweep(
            uid="time_or_amp_sweep", parameter=swp_param, reset_oscillator_phase=True
        ):
            with exp.section(uid="cav_displacement", play_after=None):
                exp.play(signal=cav_line, pulse=cav_pulse)

            with exp.section(uid="bs", play_after="cav_displacement"):
                exp.play(signal="bs", pulse=bs_pulse, length=swp_param)

            with exp.section(uid="resolved_pi_ge", play_after="bs"):
                exp.play(signal="qb_drive", pulse=resolved_X180)

            with exp.section(uid="readout", play_after="resolved_pi_ge"):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_0",
                    reset_delay=qubit_parameters["q0"]["cavity_reset_delay"],
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

    ch = "SG4"
    sig_freq_map[serial_num][ch]["bs"] = {}
    sig_freq_map[serial_num][ch]["bs"]["frequency"] = bs_freq
    sig_freq_map[serial_num][ch]["bs"]["range"] = bs_range

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
