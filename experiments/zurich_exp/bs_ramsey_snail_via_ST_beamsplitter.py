import numpy as np
from laboneq.simple import (
    AcquisitionType,
    AveragingMode,
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


def bs_ramsey_snail_via_ST_beamsplitter(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="bs_ramsey_snail_via_ST_beamsplitter",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    time_swp=LinearSweepParameter(uid="time_swp", start=8e-9, stop=10e-6, count=10),
    bs_freq=None,
    bs_range=None,
    bs_length=None,
    bs_amplitude=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    averaging_mode=AveragingMode.CYCLIC,
    rotate_ro=False,
    thresholds=None,
    swp_amp=False,
    storage_mode=1,
    n_echoes=0,  # Number of CPMG echoes
    echo_axis="Y",  # "X" or "Y" for the pi pulse phase
    sw_detuning_freq=0,
    reset_delay=None,
    chunk_count=None,
    channel=None,
    lo_name=None,
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X90 = qubit_params_module.ge_X90
    ge_X180 = qubit_params_module.ge_X180

    # bs pulse
    bs = qubit_params_module.bs_transmon_snail_eg_ge
    if bs_length is None:
        bs_length = bs.length
    if bs_amplitude is not None:
        bs.amplitude = 1
    if bs_range is None:
        bs_range = qubit_parameters["q0"]["bs_transmon_snail_eg_ge_dBm_range"]
    if bs_freq is None:
        bs_freq = qubit_parameters["q0"]["bs_transmon_snail_eg_ge_freq"]

    ch = "SG4" if channel is None else channel
    lo_name = "SG4_LO" if lo_name is None else lo_name
    lo = lo_settings["q0"][serial_num][lo_name]

    lo_range = 500e6  # on either side of the lo
    lo_min = 1e9
    step = 200e6

    new_lo = bs_freq
    new_lo = round(new_lo / step) * step
    print(f"Trying new LO frequency: {new_lo/1e9} GHz")

    if new_lo < lo_min:
        print("Warning: LO frequency is too low. Adjusting to 1 GHz.")
        new_lo = lo_min

    lo = new_lo
    lo_settings["q0"][serial_num][lo_name] = lo
    print(f"Warning: LO ({lo_name}) frequency changed to {lo/1e9} GHz")
    lo_change = True

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("bs"),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    # set up phase sweep
    phase_swp = 2 * np.pi * time_swp * sw_detuning_freq
    swp_param = [time_swp, phase_swp]

    # CPMG delays
    if n_echoes > 0:
        tau_half = time_swp / (2 * n_echoes)
        tau_full = time_swp / n_echoes

    with exp.acquire_loop_rt(
        uid="shots", count=pow(2, average_exponent), acquisition_type=acquisition_type
    ):
        with exp.sweep(
            uid="time_or_amp_sweep",
            parameter=swp_param,
            reset_oscillator_phase=True,
            chunk_count=chunk_count,
        ):
            with exp.section(uid="ge_excitation", play_after=None):
                exp.play(signal="qb_drive", pulse=ge_X90)

            with exp.section(uid="bs_ST", play_after="ge_excitation"):
                exp.play(
                    signal="bs",
                    pulse=bs,
                    length=bs_length,
                    amplitude=bs_amplitude if bs_amplitude is not None else None,
                )

            last_uid = "bs_ST"

            if n_echoes == 0:
                with exp.section(uid="ramsey_wait", play_after=last_uid):
                    exp.delay(signal="bs", time=time_swp)
                last_uid = "ramsey_wait"
            else:
                for i in range(n_echoes):
                    delay_time = tau_half if i == 0 else tau_full

                    with exp.section(uid=f"echo_{i}_delay_pre", play_after=last_uid):
                        exp.delay(signal="bs", time=delay_time)

                    with exp.section(
                        uid=f"echo_{i}_unload_snail", play_after=f"echo_{i}_delay_pre"
                    ):
                        exp.play(
                            signal="bs",
                            pulse=bs,
                            length=bs_length,
                            amplitude=bs_amplitude,
                        )

                    # PI PULSE
                    phase = np.pi / 2 if echo_axis.upper() == "Y" else 0
                    with exp.section(
                        uid=f"echo_{i}_ge_pi",
                        play_after=f"echo_{i}_unload_snail",
                        on_system_grid=True,
                    ):
                        exp.play(signal="qb_drive", pulse=ge_X180, phase=phase)

                    with exp.section(
                        uid=f"echo_{i}_load_storage", play_after=f"echo_{i}_load_buffer"
                    ):
                        exp.play(
                            signal="bs",
                            pulse=bs,
                            length=bs_length,
                            amplitude=bs_amplitude,
                        )

                    last_uid = f"echo_{i}_load_storage"

                # Final tau/2 delay
                with exp.section(uid="echo_final_delay_post", play_after=last_uid):
                    exp.delay(signal="bs", time=tau_half)
                last_uid = "echo_final_delay_post"

            # Final Unload & Readout
            with exp.section(uid="bs_unload_storage", play_after=last_uid):
                exp.play(
                    signal="bs",
                    pulse=bs,
                    length=bs_length,
                    amplitude=bs_amplitude if bs_amplitude is not None else None,
                )

            with exp.section(
                uid="ge_deexcitation",
                play_after="bs_unload_storage",
                on_system_grid=True,
            ):
                exp.play(signal="qb_drive", pulse=ge_X90, phase=phase_swp)

            with exp.section(uid="readout", play_after="ge_deexcitation"):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    acquire_signal="acquire",
                    integration_kernel=kernels,
                    handle="ac_0",
                    reset_delay=(
                        qubit_parameters["q0"]["cavity_reset_delay"]
                        if reset_delay is None
                        else reset_delay
                    ),
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
    sig_freq_map[serial_num][ch]["bs"]["frequency"] = bs_freq - lo
    sig_freq_map[serial_num][ch]["bs"]["range"] = bs_range

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

    return exp, lo
