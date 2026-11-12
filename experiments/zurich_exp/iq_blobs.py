from laboneq.simple import (
    AcquisitionType,
    AveragingMode,
    Experiment,
    ExperimentSignal,
    pulse_library,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def iq_blobs(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="iq_blobs",
    shots=1500,
    frequency=None,
    amplitude=None,
    acquire_delay=None,
    measure_f=False,
    prepare_buffer=False,
    alice_or_bob="alice",
    ss_params=False,
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = (
        qubit_params_module.readout_pulse
        if not ss_params
        else qubit_params_module.readout_pulse_ss
    )
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X180 = qubit_params_module.ge_X180
    ef_X180 = qubit_params_module.ef_X180

    sb_f0g1_alice = qubit_params_module.sb_pulses["alice"]["f0g1"]
    sb_f0g1_bob = qubit_params_module.sb_pulses["bob"]["f0g1"]

    transitions = [f"f{i}g{i+1}" for i in range(1)]
    sb_drive_lines = {}
    sb_drive_pulses = {}
    for transition in transitions:
        if alice_or_bob == "alice":
            sb_drive_lines[transition] = f"sb_drive_alice_{transition}"
        else:
            sb_drive_lines[transition] = f"sb_drive_bob_{transition}"

    reset_delay = qubit_parameters["q0"]["reset_delay"]
    cavity_reset_delay = qubit_parameters["q0"]["cavity_reset_delay"]

    if frequency is not None:
        lo = lo_settings["q0"][serial_num]["QA0_LO"]
        frequency -= lo

    if acquire_delay is not None:
        assert (
            acquire_delay <= readout_pulse.length
        ), "Acquire delay exceeds readout pulse length"
    else:
        acquire_delay = qubit_parameters["q0"]["acquire_delay"]

    if amplitude is not None:
        readout_pulse_length = (
            qubit_params_module.readout_pulse.length
            if not ss_params
            else qubit_params_module.readout_pulse_ss.length
        )
        readout_pulse = pulse_library.const(
            uid="ro_pulse", length=readout_pulse_length, amplitude=1.0
        )

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("qb_ef_drive"),
            *[ExperimentSignal(sb_drive_lines[_]) for _ in transitions],
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    with exp.acquire_loop_rt(
        uid="shots",
        count=shots,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.SINGLE_SHOT,
    ):
        # first readout
        if prepare_buffer:
            with exp.section(
                uid="ge_excitation_load_0", play_after=None, on_system_grid=True
            ):
                exp.play(signal="qb_drive", pulse=ge_X180)
            with exp.section(
                uid="ef_excitation_load_0",
                play_after="ge_excitation_load_0",
                on_system_grid=True,
            ):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)
            with exp.section(
                uid="sb_load_buffer_0",
                play_after="ef_excitation_load_0",
                on_system_grid=True,
            ):
                exp.play(
                    signal=sb_drive_lines["f0g1"],
                    pulse=(sb_f0g1_alice if alice_or_bob == "alice" else sb_f0g1_bob),
                )
            play_after = "sb_load_buffer_0"
        else:
            play_after = None

        with exp.section(uid="readout_g", play_after=play_after):
            exp.measure(
                measure_signal="measure",
                measure_pulse=readout_pulse,
                measure_pulse_amplitude=amplitude,
                acquire_signal="acquire",
                integration_kernel=kernels,
                acquire_delay=acquire_delay,
                integration_length=(
                    readout_pulse.length - acquire_delay
                    if acquire_delay is not None
                    else readout_pulse.length
                ),
                handle="ac_0",
                reset_delay=reset_delay if not prepare_buffer else cavity_reset_delay,
            )
        play_after = "readout_g"

        if prepare_buffer:
            with exp.section(
                uid="ge_excitation_load_1", play_after=play_after, on_system_grid=True
            ):
                exp.play(signal="qb_drive", pulse=ge_X180)
            with exp.section(
                uid="ef_excitation_load_1",
                play_after="ge_excitation_load_1",
                on_system_grid=True,
            ):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)
            with exp.section(
                uid="sb_load_buffer_1",
                play_after="ef_excitation_load_1",
                on_system_grid=True,
            ):
                exp.play(
                    signal=sb_drive_lines["f0g1"],
                    pulse=(sb_f0g1_alice if alice_or_bob == "alice" else sb_f0g1_bob),
                )
            play_after = "sb_load_buffer_1"

        with exp.section(uid="qubit_ge_excitation_0", play_after=play_after):
            exp.play(signal="qb_drive", pulse=ge_X180)

        with exp.section(uid="readout_e", play_after="qubit_ge_excitation_0"):
            exp.measure(
                measure_signal="measure",
                measure_pulse=readout_pulse,
                measure_pulse_amplitude=amplitude,
                acquire_signal="acquire",
                integration_kernel=kernels,
                acquire_delay=acquire_delay,
                integration_length=(
                    readout_pulse.length - acquire_delay
                    if acquire_delay is not None
                    else readout_pulse.length
                ),
                handle="ac_1",
                reset_delay=reset_delay if not prepare_buffer else cavity_reset_delay,
            )
        play_after = "readout_e"

        if measure_f:
            if prepare_buffer:
                with exp.section(
                    uid="ge_excitation_load_2",
                    play_after=play_after,
                    on_system_grid=True,
                ):
                    exp.play(signal="qb_drive", pulse=ge_X180)
                with exp.section(
                    uid="ef_excitation_load_2",
                    play_after="ge_excitation_load_2",
                    on_system_grid=True,
                ):
                    exp.play(signal="qb_ef_drive", pulse=ef_X180)
                with exp.section(
                    uid="sb_load_buffer_2",
                    play_after="ef_excitation_load_2",
                    on_system_grid=True,
                ):
                    exp.play(
                        signal=sb_drive_lines["f0g1"],
                        pulse=(
                            sb_f0g1_alice if alice_or_bob == "alice" else sb_f0g1_bob
                        ),
                    )
                play_after = "sb_load_buffer_2"

            with exp.section(uid="qubit_ge_excitation_1", play_after=play_after):
                exp.play(signal="qb_drive", pulse=ge_X180)

            with exp.section(
                uid="qubit_ef_excitation",
                play_after="qubit_ge_excitation_1",
                on_system_grid=True,
            ):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)

            with exp.section(uid="readout_f", play_after="qubit_ef_excitation"):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    measure_pulse_amplitude=amplitude,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    acquire_delay=acquire_delay,
                    integration_length=(
                        readout_pulse.length - acquire_delay
                        if acquire_delay is not None
                        else readout_pulse.length
                    ),
                    handle="ac_2",
                    reset_delay=(
                        reset_delay * 2 if not prepare_buffer else cavity_reset_delay
                    ),
                )

    # setup calibration and signal map for the experiment
    sig_freq_map = create_default_map_and_calibration(
        exp, serial_num, qubit_parameters, lo_settings, single_shot=ss_params
    )

    if frequency is not None:
        sig_freq_map[serial_num]["QA0"]["measure/acquire"]["frequency"] = frequency

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
