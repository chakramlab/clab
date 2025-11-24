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


def bs_spectroscopy_with_sb(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="bs_with_sb_spectroscopy",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    freq_swp=LinearSweepParameter(
        uid="freq_swp_param", start=-700e6, stop=700e6, count=6
    ),
    bs_length=1e-6,
    bs_range=10,
    bs_amplitude=1,    
    acquisition_type=AcquisitionType.INTEGRATION,
    alice_or_bob="alice",
    max_fock_state=1,  # only works for 1 now
    rotate_ro=False,
    thresholds=None,
    read_on_one_peak=True
    ):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X180 = qubit_params_module.ge_X180
    ef_X180 = qubit_params_module.ef_X180
    resolved_X180 = qubit_params_module.resolved_X180
    sb_f0g1_alice = qubit_params_module.sb_pulses["alice"]["f0g1"]
    sb_f0g1_bob   = qubit_params_module.sb_pulses["bob"]["f0g1"]
    # bs pulse
    bs1_bob = pulse_library.const(
        uid="bs1_bob", length=bs_length, amplitude=bs_amplitude, can_compress=True
    )
    
    lo = lo_settings["q0"][serial_num]["SG4_LO"]
    freq_swp.start -= lo
    freq_swp.stop -= lo

    if read_on_one_peak:
        # shift resolved qb freq by chi
        if alice_or_bob == "alice":
            qubit_parameters["q0"]["qb_resolved_freq"] += qubit_parameters["q0"]["cav_alice_chi"]
        elif alice_or_bob == "bob":
            qubit_parameters["q0"]["qb_resolved_freq"] += qubit_parameters["q0"]["cav_bob_chi"]
        else:
            raise ValueError("alice_or_bob must be 'alice' or 'bob'")

    transitions = [f"f{i}g{i+1}" for i in range(max_fock_state)]
    sb_drive_lines = {}
    sb_drive_pulses = {}
    for transition in transitions:
        if alice_or_bob == "alice":
            sb_drive_lines[transition] = f"sb_drive_alice_{transition}"
        else:
            sb_drive_lines[transition] = f"sb_drive_bob_{transition}"

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("qb_ef_drive"),
            ExperimentSignal("qb_drive_resolved"),
            ExperimentSignal("bs"),
            *[ExperimentSignal(sb_drive_lines[_]) for _ in transitions],
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )
    with exp.acquire_loop_rt(
        uid="shots", 
        count=pow(2, average_exponent), 
        acquisition_type=acquisition_type
    ):
        with exp.sweep(
            uid="spect_sweep", parameter=freq_swp, reset_oscillator_phase=True
        ):
            with exp.section(uid = "ge_excitation",  play_after=None):
                exp.play(signal = "qb_drive", pulse = ge_X180)
            with exp.section(uid = "ef_excitation", play_after= "ge_excitation", on_system_grid=True):
                exp.play(signal = "qb_ef_drive", pulse = ef_X180)
            with exp.section(uid = "sb_transition_f0g1", play_after = "ef_excitation"):
                exp.play(signal = sb_drive_lines["f0g1"], 
                         pulse = sb_f0g1_alice if alice_or_bob=="alice" else sb_f0g1_bob, 
                )
            with exp.section(uid="bs", play_after="sb_transition_f0g1"):
                exp.play(signal="bs", pulse=bs1_bob)

            with exp.section(uid="resolved_pi_ge", play_after="bs"):
                exp.play(signal="qb_drive_resolved", pulse=resolved_X180)

            with exp.section(uid="readout", play_after="resolved_pi_ge"):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_0",
                    reset_delay=qubit_parameters["q0"]["cavity_reset_delay"],
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

    ch = "SG4"
    sig_freq_map[serial_num][ch]["bs"] = {}
    sig_freq_map[serial_num][ch]["bs"]["frequency"] = freq_swp
    sig_freq_map[serial_num][ch]["bs"]["range"] = bs_range
    sig_freq_map[serial_num][ch]["bs"]["length"] = bs_length
    sig_freq_map[serial_num][ch]["bs"]["amplitude"] = bs_amplitude

    print(sig_freq_map[serial_num][ch]["bs"])


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
