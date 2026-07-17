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


def readout_dispersive_shift(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="readout_spectroscopy",
    average_exponent=5,  # 2^n averages, n=average_exponent, maximum: n = 17. You can modify the code to average for any integer number if needed.
    freq_swp=LinearSweepParameter(
        uid="freq_swp_param",
        start=-700e6,
        stop=700e6,
        count=101,
    ),
    measure_f=False,
    resolved=False,
    ef_pulse = False,
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse_length = qubit_params_module.readout_pulse.length
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    ge_X180 = qubit_params_module.ge_X180

    qb_drive = qubit_params_module.resolved_X180 if resolved else ge_X180
    ef_drive = qubit_params_module.ef_X180

    lo = lo_settings["q0"][serial_num]['QA0_LO']
    freq_swp.start -= lo
    freq_swp.stop -= lo

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    readout_pulse_swp = pulse_library.const(
        uid="ro_pulse", length=readout_pulse_length, amplitude=1.0
    )

    with exp.acquire_loop_rt(
        uid="shots",
        count=pow(2, average_exponent),
        acquisition_type=AcquisitionType.SPECTROSCOPY,
    ):
        
        with exp.sweep(uid="spect_sweep", parameter=freq_swp):        
            with exp.section(uid="readout_0", play_after=None):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse_swp,
                    acquire_signal="acquire",
                    integration_length=qubit_parameters["q0"]["ro_len"],
                    handle="ac_0",
                    reset_delay=100e-6,
                    acquire_delay=qubit_parameters['q0']['acquire_delay']
                    )
            play_after = "readout_0"        


            with exp.section(uid="qubit_ge_excitation_0", play_after=play_after):
                exp.play(signal="qb_drive", pulse=qb_drive)
            with exp.section(uid="readout_1", play_after="qubit_ge_excitation_0"):
                exp.measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse_swp,
                    acquire_signal="acquire",
                    integration_length=qubit_parameters["q0"]["ro_len"],
                    handle="ac_1",
                    reset_delay=qubit_parameters["q0"]["reset_delay"],
                    acquire_delay=qubit_parameters['q0']['acquire_delay']
                )
            play_after = "readout_1"

            if ef_pulse:
                with exp.section(uid="qubit_ge_excitation_1", play_after=play_after):
                    exp.play(signal="qb_drive", pulse=qb_drive)
                with exp.section(uid="qubit_ef_excitation", play_after="qubit_ge_excitation_1"):
                    exp.play(signal="qb_drive", pulse=ef_drive)
                with exp.section(uid="readout_2", play_after="qubit_ef_excitation"):
                    exp.measure(
                        measure_signal="measure",
                        measure_pulse=readout_pulse_swp,
                        acquire_signal="acquire",
                        integration_length=qubit_parameters["q0"]["ro_len"],
                        handle="ac_2",
                        reset_delay=qubit_parameters["q0"]["reset_delay"]*2,
                        acquire_delay=qubit_parameters['q0']['acquire_delay']
                    )


    # setup calibration and signal map for the experiment
    sig_freq_map = create_default_map_and_calibration(exp, serial_num, qubit_parameters, lo_settings)
    # update the frequency to incorporate the frequency sweep
    sig_freq_map[serial_num]["QA0"]["measure/acquire"]["frequency"] = freq_swp

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
