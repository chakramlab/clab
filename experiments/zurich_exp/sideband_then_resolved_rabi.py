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


def resolved_rabi_with_sideband(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="resolved_rabi_with_sideband",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    amplitude_rabi=False,
    time_rabi=True,
    swp_param=LinearSweepParameter(
        uid="sweep_param", start=40e-9, stop=400e-9, count=11
    ),
    rotate_ro=False,
    thresholds=None,
    amplitude=None,
    use_cavity_reset_delay=True,  # Whether to use the cavity reset delay from qubit parameters
    chi=True,
    qubit_drive_freq=None,
    prepare_fock_state=True,
    alice_or_bob="alice",
    max_fock_state=1,  # only works for 1 now
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    resolved_X180 = qubit_params_module.resolved_X180
    ge_X180 = qubit_params_module.ge_X180
    ge_X180_swp = resolved_X180
    ef_X180 = qubit_params_module.ef_X180
    sb_pulses = qubit_params_module.sb_pulses
    sb_f0g1_alice = qubit_params_module.sb_pulses["alice"]["f0g1"]
    sb_f0g1_bob   = qubit_params_module.sb_pulses["bob"]["f0g1"]

    reset_delay = qubit_parameters["q0"]["cavity_reset_delay"]

    transitions = [f"f{i}g{i+1}" for i in range(max_fock_state)]
    sb_drive_lines = {}
    sb_drive_pulses = {}
    for transition in transitions:
        if alice_or_bob == "alice":
            sb_drive_lines[transition] = f"sb_drive_alice_{transition}"
        else:
            sb_drive_lines[transition] = f"sb_drive_bob_{transition}"

    # shift qb_freq by chi
    if chi:
        qubit_parameters["q0"]["qb_resolved_freq"] += qubit_parameters["q0"][f"cav_{alice_or_bob}_chi"]

    # Create Experiment
    if use_cavity_reset_delay:
        reset_delay = qubit_parameters["q0"]["cavity_reset_delay"]
    else:
        reset_delay = qubit_parameters["q0"]["reset_delay"]

    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("qb_ef_drive"),
            ExperimentSignal("qb_drive_resolved"),
            *[ExperimentSignal(sb_drive_lines[_]) for _ in transitions],
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    if not (time_rabi ^ amplitude_rabi):
        raise ValueError(
            "Please select either time_rabi or amplitude_rabi to be True and the other to be False."
        )

    if amplitude is None:
        amplitude = resolved_X180.amplitude
    with exp.acquire_loop_rt(
        uid="shots",
        count=pow(2, average_exponent),
    ):
        with exp.sweep(
            uid="time_or_amp_sweep", parameter=swp_param, reset_oscillator_phase=True, chunk_count=swp_param.count
        ):
            if prepare_fock_state:
                with exp.section(uid = "ge_excitation",  play_after=None): #alignment=SectionAlignment.RIGHT):
                    exp.play(signal = "qb_drive", pulse = ge_X180)
                with exp.section(uid = "ef_excitation", play_after= "ge_excitation", on_system_grid=True):
                    exp.play(signal = "qb_ef_drive", pulse = ef_X180)
                with exp.section(uid = "sb_transition_f0g1", play_after = "ef_excitation"):
                    exp.play(signal = sb_drive_lines["f0g1"], pulse = sb_f0g1_alice if alice_or_bob=="alice" else sb_f0g1_bob)
                with exp.section(uid = "fe_transition", play_after = "sb_transition_f0g1"):
                    exp.play(signal = "qb_ef_drive", pulse = ef_X180)
                play_after = "fe_transition"
            else:
                play_after = None
            with exp.section(uid="qubit_rabi", play_after=play_after):
                if time_rabi:
                    exp.play(signal="qb_drive_resolved", pulse=ge_X180_swp, length=swp_param, )
                elif amplitude_rabi:
                    exp.play(signal="qb_drive_resolved", pulse=ge_X180_swp, amplitude=swp_param, )
            with exp.section(uid="readout", play_after="qubit_rabi"):
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
    
    if qubit_drive_freq is not None:
        sig_freq_map[serial_num]["SG0"]["qb_drive_resolved"]["frequency"] = qubit_drive_freq - lo_settings["q0"][serial_num]["SG3_LO"]

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
