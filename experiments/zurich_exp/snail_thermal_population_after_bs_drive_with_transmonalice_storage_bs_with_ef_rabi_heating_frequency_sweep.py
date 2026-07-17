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


def snail_thermal_population_after_bs_drive_with_transmonalice_storage_bs_with_ef_rabi_heating_frequency_sweep(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="snail_thermal_population_after_bs_drive_with_transmonalice_storage_bs_with_ef_rabi",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    swp_param_off_resonant=LinearSweepParameter(
        uid="swp_param_off_resonant", start=0.5e9, stop=1.5e9, count=11
    ),
    swp_param=LinearSweepParameter(
        uid="swp_param", start=1e-9, stop=10e-6, count=6
    ),
    bs_freq=None,
    bs_range=None,
    bs_length=None,
    bs_amplitude=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    alice_or_bob="alice",
    rotate_ro=False,
    thresholds=None,
    storage_mode=0,
    pi_alice_or_pi_swap="swap",
    bs_length_off_resonant=30e-6,
    ):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X180 = qubit_params_module.ge_X180
    ef_X180 = qubit_params_module.ef_X180

    # bs pulse (actual resonant one)
    bs = qubit_params_module.sb_pulses[alice_or_bob][f'bs{storage_mode}']
    if bs_length is None:
        bs_length = bs.length
    # if bs_ramp is None:
    #     bs_ramp = qubit_params_module.sb_pulses['alice'][f'bs{storage_mode}'].pulse_parameters['ramp']
    if bs_amplitude is None:
        bs_amplitude = bs.amplitude
    if bs_range is None:
        bs_range = qubit_parameters["q0"][f"bs_{alice_or_bob}_dBm_ranges"][storage_mode]
    if bs_freq is None:
        bs_freq = qubit_parameters["q0"][f"bs_{alice_or_bob}_freqs"][storage_mode]


    lo = lo_settings["q0"][serial_num]["SG4_LO"]
    lo_range = 0.5e9
    if bs_freq < lo - lo_range or bs_freq > lo + lo_range:
        old_lo = lo
        new_lo = bs_freq
        step = 200e6
        new_lo = round(new_lo / step) * step
        lo_settings["q0"][serial_num]["SG4_LO"] = new_lo
        lo = new_lo
        print(f"Warning: LO frequency changed to {new_lo/1e9} GHz")
        lo_change=True

    # Create Experiment - using single "bs" signal for both pulses with frequency switching
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("qb_ef_drive"),
            ExperimentSignal("bs"),
            ExperimentSignal("bs_off_resonant"),
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
                uid="freq_sweep_off_resonant", parameter=swp_param_off_resonant, reset_oscillator_phase=True,
                ):  

            with exp.sweep(
                uid="time_or_amp_sweep", parameter=swp_param, reset_oscillator_phase=True,
                ):

                with exp.section(uid="bs_off_resonant", play_after=None, on_system_grid=True):
                    exp.play(
                            signal="bs_off_resonant", 
                            pulse=bs,
                            length=bs_length_off_resonant
                        )

                if pi_alice_or_pi_swap == "ge":
                    with exp.section(uid="ge_excitation", play_after="bs_off_resonant", on_system_grid=True):
                        exp.play(
                            signal="qb_drive",
                            pulse=ge_X180,
                        )
                    play_after = "ge_excitation"

                elif pi_alice_or_pi_swap == "swap":
                    with exp.section(uid="bs", play_after="bs_off_resonant", on_system_grid=True):
                        exp.play(
                            signal="bs", 
                            pulse=bs,
                            length=bs_length, 
                            amplitude=bs_amplitude
                        )
                    play_after = "bs"
                
                with exp.section(uid="ef", play_after=play_after, on_system_grid=True):
                    exp.play(
                        signal="qb_ef_drive", 
                        pulse=ef_X180, 
                        length=swp_param
                    )

                with exp.section(uid="ge_excitation2", play_after="ef", on_system_grid=True):
                    exp.play(
                        signal="qb_drive", 
                        pulse=ge_X180, 
                    )

                with exp.section(uid="readout", play_after="ef"):
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
    sig_freq_map[serial_num][ch]["bs"]["frequency"] = bs_freq - lo
    sig_freq_map[serial_num][ch]["bs"]["range"] = bs_range

    sig_freq_map[serial_num][ch]["bs_off_resonant"] = {}
    sig_freq_map[serial_num][ch]["bs_off_resonant"]["frequency"] = swp_param_off_resonant - lo
    sig_freq_map[serial_num][ch]["bs_off_resonant"]["range"] = bs_range

    print(sig_freq_map[serial_num][ch])

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

    return exp,lo
