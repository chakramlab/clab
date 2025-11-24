from laboneq.simple import (
    AcquisitionType,
    Experiment,
    ExperimentSignal,
    LinearSweepParameter,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import (
    default_signal_map_and_calibration,
    parsing_single_alice_bob_cav_and_sb_transitions,
)
from .load_qubit_params import load_qubit_params


def cavity_spectroscopy(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="cavity_spectroscopy",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    freq_swp=LinearSweepParameter(
        uid="freq_swp_param", start=-700e6, stop=700e6, count=6
    ),
    acquisition_type=AcquisitionType.INTEGRATION,
    alice_or_bob="a",
    rotate_ro=False,
    thresholds=None,
    qubit_drive_freq=None,
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

    lo = lo_settings["q0"][serial_num]['SG2_LO']
    freq_swp.start -= lo
    freq_swp.stop -= lo

    cav_name, dummy, dummy, dummy, cav_line, cav_pulse = (
        parsing_single_alice_bob_cav_and_sb_transitions(
            alice_or_bob=alice_or_bob,
            cav_alice=cav_alice,
            cav_bob=cav_bob,
        )
    )

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive_resolved"),
            ExperimentSignal(cav_line),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )
    with exp.acquire_loop_rt(
        uid="shots", count=pow(2, average_exponent), acquisition_type=acquisition_type
    ):
        with exp.sweep(
            uid="spect_sweep", parameter=freq_swp, reset_oscillator_phase=True
        ):
            with exp.section(uid="cav_displacement"):
                exp.play(signal=cav_line, pulse=cav_pulse)
            with exp.section(uid="resolved_pi_ge", play_after="cav_displacement"):
                exp.play(signal="qb_drive_resolved", pulse=resolved_X180)
            with exp.section(uid="readout", play_after="resolved_pi_ge"):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_0",
                    reset_delay=qubit_parameters["q0"]["cavity_reset_delay"],
                    acquire_delay=qubit_parameters['q0']['acquire_delay']
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
        sig_freq_map[serial_num]["SG0"]["qb_drive_resolved"]["frequency"] = qubit_drive_freq - lo_settings["q0"][serial_num]["SG0_LO"]

    # update the frequency to incorporate the frequency sweep
    sig_freq_map[serial_num]["SG3"][cav_line]["frequency"] = freq_swp

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
