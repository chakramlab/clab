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


def pnrqs(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="pnrqs",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    freq_swp=LinearSweepParameter(uid="freq_swp_param", start=0e6, stop=50e6, count=11),
    acquisition_type=AcquisitionType.INTEGRATION,
    alice_or_bob="a",
    state=0,  # currently only accepts state 0 and 1
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
    sb_pulses = qubit_params_module.sb_pulses

    reset_delay = qubit_parameters["q0"]["cavity_reset_delay"]

    lo = lo_settings["q0"][serial_num]['SG0_LO']
    freq_swp.start -= lo
    freq_swp.stop -= lo

    cav_name, transitions, sb_lines, sb_pulses_parsed, cav_line, cav_pulse = (
        parsing_single_alice_bob_cav_and_sb_transitions(
            alice_or_bob=alice_or_bob,
            sb_pulses=sb_pulses,
            max_fock_state=state,
            cav_alice=cav_alice,
            cav_bob=cav_bob,
        )
    )
    print("cav_line:", cav_line)

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            # ExperimentSignal("qb_drive"),
            # ExperimentSignal("qb_ef_drive"),
            ExperimentSignal("qb_selective_pi_drive"),
            ExperimentSignal(cav_line),
            # *[ExperimentSignal(sb_lines[_]) for _ in transitions],
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    with exp.acquire_loop_rt(
        uid="shots",
        count=pow(2, average_exponent),
        acquisition_type=acquisition_type,
    ):
        with exp.sweep(
            uid="spect_sweep", parameter=freq_swp, reset_oscillator_phase=True
        ):
            # previous_section = prepare_cav_state(exp,
            #           state = str(state),
            #           ge_X180 = ge_X180,
            #           ef_X180 = ef_X180,
            #           sb_pulses = sb_pulses,
            #           alice_or_bob = alice_or_bob,
            #           function_index = 0,
            #           )
            with exp.section(uid="cav_displacement", play_after=None):
                # pass
                exp.play(signal=cav_line, pulse=cav_pulse)

            with exp.section(uid="resolved_pi_ge", play_after="cav_displacement"):
                # exp.play(signal = "qb_drive", pulse = resolved_X180)
                exp.play(signal="qb_selective_pi_drive", pulse=resolved_X180)
            with exp.section(uid="readout", play_after="resolved_pi_ge"):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_0",
                    reset_delay=reset_delay,
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
    # update the frequency to incorporate the frequency sweep
    # sig_freq_map[serial_num]["SG0"]["qb_drive"]["frequency"] = freq_swp
    # note that this will also sweep the freq for regular pi pulse... is that okay? if not, do something like below:
    # update the calibration to incorporate the additional logical line for selective pi pulse
    ch = "SG0"
    sig_freq_map[serial_num][ch]["qb_selective_pi_drive"] = {}
    sig_freq_map[serial_num][ch]["qb_selective_pi_drive"]["frequency"] = freq_swp
    sig_freq_map[serial_num][ch]["qb_selective_pi_drive"]["range"] = qubit_parameters[
        "q0"
    ]["qb_drive_resolved_dBm_range"]
    sig_freq_map[serial_num][ch]["qb_selective_pi_drive"]["automute"] = True

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
