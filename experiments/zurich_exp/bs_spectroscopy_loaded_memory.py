from laboneq.simple import (
    AcquisitionType,
    Experiment,
    ExperimentSignal,
    LinearSweepParameter,
    pulse_library,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def bs_spectroscopy_loaded_memory(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="bs_spectroscopy_loaded_memory",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    freq_swp=LinearSweepParameter(
        uid="freq_swp_param", start=-700e6, stop=700e6, count=6
    ),
    bs_length=None,
    bs_range=None,
    bs_ramp=None,
    bs_amplitude=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    buffer_spec="alice",
    storage_spec=1,
    buffer_spect="bob",
    storage_spect=3,
    max_fock_state=1,  # only works for 1 now
    rotate_ro=False,
    thresholds=None,
    sb_delay=0,
    load_spectator=True,
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X180 = qubit_params_module.ge_X180
    ef_X180 = qubit_params_module.ef_X180
    
    sb_f0g1_spec = qubit_params_module.sb_pulses[buffer_spec]["f0g1"]
    sb_f0g1_spect = qubit_params_module.sb_pulses[buffer_spect]["f0g1"]

    # bs pulses
    bs_spec = qubit_params_module.sb_pulses[buffer_spec][f"bs{storage_spec}"]
    bs_spect = qubit_params_module.sb_pulses[buffer_spect][f"bs{storage_spect}"]

    if bs_length is None:
        bs_length = bs_spect.length
    if bs_amplitude is not None:
        bs_spect.amplitude = 1
    if bs_range is None:
        bs_range = qubit_parameters["q0"][f"bs_{buffer_spect}_dBm_ranges"][storage_spect]

    bs_range_spec = qubit_parameters["q0"][f"bs_{buffer_spec}_dBm_ranges"][storage_spec]
    bs_freq_spec = qubit_parameters["q0"][f"bs_{buffer_spec}_freqs"][storage_spec]

    lo = lo_settings["q0"][serial_num]["SG4_LO"]
    lo_range = 0.5e9

    requires_shift = False
    if freq_swp.start < lo - lo_range or freq_swp.stop > lo + lo_range:
        requires_shift = True
    if bs_freq_spec < lo - lo_range or bs_freq_spec > lo + lo_range:
        requires_shift = True

    if requires_shift:
        new_lo = ((freq_swp.start + freq_swp.stop) / 2 + bs_freq_spec) / 2
        step = 200e6
        new_lo = round(new_lo / step) * step
        if new_lo < 1e9:
            new_lo = 0
        lo_settings["q0"][serial_num]["SG4_LO"] = new_lo
        lo = new_lo
        print(f"Warning: LO frequency changed to {new_lo/1e9} GHz")
        
    freq_swp.start -= lo
    freq_swp.stop -= lo

    transitions = [f"f{i}g{i+1}" for i in range(max_fock_state)]
    sb_drive_lines_spec = {}
    sb_drive_lines_spect = {}
    for transition in transitions:
        sb_drive_lines_spec[transition] = f"sb_drive_{buffer_spec}_{transition}"
        sb_drive_lines_spect[transition] = f"sb_drive_{buffer_spect}_{transition}"

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("qb_ef_drive"),
            ExperimentSignal("bs_spec"),
            ExperimentSignal("bs_spect"),
            *[ExperimentSignal(sb_drive_lines_spec[_]) for _ in transitions],
            *[ExperimentSignal(sb_drive_lines_spect[_]) for _ in transitions],
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
            with exp.section(uid="ge_excitation", play_after=None, on_system_grid=True):
                exp.play(signal="qb_drive", pulse=ge_X180)

            with exp.section(
                uid="ef_excitation", play_after="ge_excitation", on_system_grid=True
            ):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)

            if load_spectator:
                with exp.section(
                    uid="sb_transition_f0g1_1",
                    play_after="ef_excitation",
                    on_system_grid=True,
                ):
                    exp.play(
                        signal=sb_drive_lines_spec["f0g1"],
                        pulse=sb_f0g1_spec,
                    )
                    exp.delay(signal=sb_drive_lines_spec["f0g1"], time=sb_delay)

                with exp.section(
                    uid="bs_spec_park",
                    play_after="sb_transition_f0g1_1",
                    on_system_grid=True,
                ):
                    exp.play(signal="bs_spec", pulse=bs_spec)

            with exp.section(
                uid="sb_transition_f0g1_2",
                play_after="bs_spec_park" if load_spectator else "ef_excitation",
                on_system_grid=True,
            ):
                exp.play(
                    signal=sb_drive_lines_spect["f0g1"],
                    pulse=sb_f0g1_spect,
                )
                exp.delay(signal=sb_drive_lines_spect["f0g1"], time=sb_delay)

            with exp.section(uid="bs_spect", play_after="sb_transition_f0g1_2", on_system_grid=True):
                exp.play(
                    signal="bs_spect", 
                    pulse=bs_spect, 
                    length=bs_length, 
                    amplitude=bs_amplitude if bs_amplitude is not None else None
                )

            with exp.section(uid="sb_transition_f0g1_3", play_after="bs_spect", on_system_grid=True):
                exp.delay(signal=sb_drive_lines_spect["f0g1"], time=sb_delay)
                exp.play(
                    signal=sb_drive_lines_spect["f0g1"],
                    pulse=sb_f0g1_spect,
                )
                exp.delay(signal=sb_drive_lines_spect["f0g1"], time=sb_delay)

            with exp.section(
                uid="ef_excitation_2",
                play_after="sb_transition_f0g1_3",
                on_system_grid=True,
            ):
                exp.play(signal="qb_ef_drive", pulse=ef_X180)

            with exp.section(uid="readout", play_after="ef_excitation_2", on_system_grid=True):
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
    sig_freq_map[serial_num][ch]["bs_spec"] = {}
    sig_freq_map[serial_num][ch]["bs_spec"]["frequency"] = bs_freq_spec - lo
    sig_freq_map[serial_num][ch]["bs_spec"]["range"] = bs_range_spec

    sig_freq_map[serial_num][ch]["bs_spect"] = {}
    sig_freq_map[serial_num][ch]["bs_spect"]["frequency"] = freq_swp
    sig_freq_map[serial_num][ch]["bs_spect"]["range"] = bs_range
    sig_freq_map[serial_num][ch]["bs_spect"]["length"] = bs_length
    sig_freq_map[serial_num][ch]["bs_spect"]["amplitude"] = bs_amplitude

    print("bs_spec map:", sig_freq_map[serial_num][ch]["bs_spec"])
    print("bs_spect map:", sig_freq_map[serial_num][ch]["bs_spect"])

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
