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


def beamsplitter_spectroscopy(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="beamsplitter_spectroscopy",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    cavity_mode_index=0,
    alice_or_bob="a",  # "a" for Alice, "b" for Bob
    state=0,  # currently only accepts state 0 and 1
    bs_max=1,  # maximum number of sideband transitions to consider, starting at 0
    freq_swp=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    kernels=None,
    rotate_ro=False,
    thresholds=None,
    read_on_one_peak=True,
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

    lo = lo_settings["q0"][serial_num]["SG4_LO"]
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

    if read_on_one_peak:
        # shift qb freq by chi
        if alice_or_bob == "a":
            qubit_parameters["q0"]["qb_freq"] += qubit_parameters["q0"]["cav_alice_chi"]
        elif alice_or_bob == "b":
            qubit_parameters["q0"]["qb_freq"] += qubit_parameters["q0"]["cav_bob_chi"]
        else:
            raise ValueError("alice_or_bob must be 'a' or 'b'")

    if freq_swp is None:
        raise ValueError(
            "Frequency sweep parameter 'freq_swp' must be provided for this experiment."
        )

    transitions = [f"sb_bs{i}" for i in range(bs_max)]
    sb_drive_lines = {}
    sb_drive_pulses = sb_pulses["alice"]["sb_bs0"]
    # for transition in transitions:
    #     sb_drive_lines[transition] = "sb_drive_alice_bs0"
    #     sb_drive_pulses[transition] = sb_pulses["alice"][transition]
    # print(sb_drive_pulses)

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive_resolved"),
            ExperimentSignal(cav_line),
            # *[ExperimentSignal(sb_drive_lines[_]) for _ in transitions],
            ExperimentSignal("sb_drive_alice_bs0"),
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
            with exp.section(
                uid="cavity_ge_excitation"
            ):  # ,  alignment=SectionAlignment.RIGHT):
                exp.play(signal=cav_line, pulse=cav_pulse)
            with exp.section(
                uid="sb_transition_bs0", play_after="cavity_ge_excitation"
            ):
                exp.play(signal="sb_drive_alice_bs0", pulse=sb_drive_pulses)
            with exp.section(
                uid="resolved_ge_excitation", play_after="sb_transition_bs0"
            ):
                exp.play(signal="qb_drive_resolved", pulse=resolved_X180)
            with exp.section(uid="readout", play_after="resolved_ge_excitation"):
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
    print(sig_freq_map)
    # update the frequency to incorporate the frequency sweep
    sig_freq_map[serial_num]["SG4"]["sb_drive_alice_bs0"]["frequency"] = freq_swp

    # exp_calibration, map_q0 = default_signal_map_and_calibration(sig_freq_map, metadata)

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
