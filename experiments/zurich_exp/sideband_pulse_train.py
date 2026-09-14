from laboneq.simple import AcquisitionType, Experiment, ExperimentSignal

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def sideband_pulse_train(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="sideband_pulse_train",
    average_exponent=5,  # 2^n averages
    swp_param=None,  # Expecting a SweepParameter tracking integer pulse counts
    alice_or_bob="alice",
    max_fock_state=1,
    sb_freq=None,
    sb_length=None,
    sb_range=None,
    sb_amplitude=None,
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
    ef_X180 = qubit_params_module.ef_X180
    sb_f0g1_alice = qubit_params_module.sb_pulses["alice"]["f0g1"]
    sb_f0g1_bob = qubit_params_module.sb_pulses["bob"]["f0g1"]

    qb_drive_signal = "qb_drive"
    qb_ef_drive_signal = "qb_ef_drive"

    transitions = [f"f{i}g{i+1}" for i in range(max_fock_state)]
    sb_drive_lines = {}
    for transition in transitions:
        if alice_or_bob == "alice":
            sb_drive_lines[transition] = f"sb_drive_alice_{transition}"
        else:
            sb_drive_lines[transition] = f"sb_drive_bob_{transition}"
    if sb_length is None:
        sb_length = (
            sb_f0g1_alice.length if alice_or_bob == "alice" else sb_f0g1_bob.length
        )
    if sb_amplitude is not None:
        if alice_or_bob == "alice":
            sb_f0g1_alice.amplitude = 1
        else:
            sb_f0g1_bob.amplitude = 1
    if sb_range is None:
        sb_range = qubit_parameters["q0"][f"sb_{alice_or_bob}_dBm_range"]
    if sb_freq is None:
        sb_freq = qubit_parameters["q0"][f"sb_{alice_or_bob}_freqs"][0]

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal(qb_drive_signal),
            ExperimentSignal(qb_ef_drive_signal),
            *[ExperimentSignal(sb_drive_lines[_]) for _ in transitions],
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    with exp.acquire_loop_rt(
        uid="shots", count=pow(2, average_exponent), acquisition_type=acquisition_type
    ):

        with exp.sweep(
            uid="pulse_count_sweep",
            parameter=swp_param,
            reset_oscillator_phase=True,
            chunk_count=chunk_count,
        ):

            with exp.section(
                uid="ge_excitation",
                play_after=None,
            ):
                exp.play(
                    signal=qb_drive_signal,
                    pulse=ge_X180,
                )

            with exp.section(
                uid="ef_excitation", play_after="ge_excitation", on_system_grid=True
            ):
                exp.play(
                    signal=qb_ef_drive_signal,
                    pulse=ef_X180,
                )

            with exp.section(uid="sb_transition_f0g1", play_after="ef_excitation"):
                with exp.match(sweep_parameter=swp_param):
                    for n in swp_param.values:
                        n_int = int(n)
                        with exp.case(n_int):
                            for _ in range(n_int):
                                exp.play(
                                    signal=sb_drive_lines["f0g1"],
                                    pulse=(
                                        sb_f0g1_alice
                                        if alice_or_bob == "alice"
                                        else sb_f0g1_bob
                                    ),
                                    amplitude=sb_amplitude,
                                )

            with exp.section(
                uid="fe_excitation",
                play_after="sb_transition_f0g1",
                # on_system_grid=True,
            ):
                exp.play(
                    signal=qb_ef_drive_signal,
                    pulse=ef_X180,
                )
                # (
                #     exp.delay(signal=qb_drive_signal, time=pulse_delay)
                #     if _ < n_int - 1
                #     else None
                # )

            with exp.section(uid="readout", play_after="fe_excitation"):
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

    lo = lo_settings["q0"][serial_num]["SG0_LO"]
    ch = "SG1"
    sig_freq_map[serial_num][ch] = {}
    sig_freq_map[serial_num][ch][sb_drive_lines["f0g1"]] = {}
    sig_freq_map[serial_num][ch][sb_drive_lines["f0g1"]]["frequency"] = sb_freq - lo
    sig_freq_map[serial_num][ch][sb_drive_lines["f0g1"]]["range"] = sb_range

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
