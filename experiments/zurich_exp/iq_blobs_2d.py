from laboneq.simple import (
    AcquisitionType,
    AveragingMode,
    Experiment,
    ExperimentSignal,
    LinearSweepParameter,
    pulse_library,
)

from .calib_settings import create_default_map_and_calibration
from .laboneq_helper import default_signal_map_and_calibration
from .load_qubit_params import load_qubit_params


def iq_blobs_2d(
    device_setup,
    serial_num,
    qubit_params_file_path,
    exp_id="iq_blobs_2d",
    count=10000,
    amp_swp=LinearSweepParameter(
        uid="amp_swp_param",
        start=1,
        stop=1,
        count=1,
    ),
    freq_swp=LinearSweepParameter(
        uid="freq_swp_param",
        start=-700e6,
        stop=700e6,
        count=101,
    ),
    measure_f=False,
):

    # Load device and config params
    qubit_params_module = load_qubit_params(qubit_params_file_path)
    lo_settings = qubit_params_module.create_lo_settings(serial_num)
    readout_pulse = qubit_params_module.readout_pulse
    qubit_parameters = qubit_params_module.__dict__["qubit_parameters"]
    kernels = qubit_params_module.acquire_kernel
    ge_X180 = qubit_params_module.ge_X180
    ef_X180 = qubit_params_module.ef_X180
    acquire_delay = qubit_parameters['q0']['acquire_delay']
    reset_delay = qubit_parameters["q0"]["reset_delay"]

    lo = lo_settings["q0"][serial_num]['QA0_LO']
    freq_swp.start -= lo
    freq_swp.stop -= lo

    # Create Experiment
    exp = Experiment(
        uid=exp_id,
        signals=[
            ExperimentSignal("qb_drive"),
            ExperimentSignal("qb_ef_drive"),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    # the amplitude modified by 'sweep' will multiply to the default length. So here, we set the default to 1.
    readout_pulse = pulse_library.const(
        uid="ro_pulse", length=readout_pulse.length, amplitude=1.0
    )

    with exp.sweep(uid='amp_sweep', parameter=amp_swp):
        with exp.acquire_loop_rt(
            uid="shots",
            count=count,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
            averaging_mode=AveragingMode.SINGLE_SHOT,
        ):
            with exp.sweep(uid="freq_sweep", parameter=freq_swp):
                with exp.section(uid="readout_g"):
                    exp.measure(
                        measure_signal="measure",
                        measure_pulse=readout_pulse,
                        measure_pulse_amplitude=amp_swp,
                        acquire_signal="acquire",
                        integration_kernel=kernels,
                        acquire_delay=acquire_delay,
                        integration_length=readout_pulse.length - acquire_delay,
                        handle="ac_0",
                        reset_delay=reset_delay,
                    )

                with exp.section(uid="qubit_ge_excitation_0", play_after="readout_g"):
                    exp.play(signal="qb_drive", pulse=ge_X180)

                with exp.section(uid="readout_e", play_after="qubit_ge_excitation_0"):
                    exp.measure(
                        measure_signal="measure",
                        measure_pulse=readout_pulse,
                        measure_pulse_amplitude=amp_swp,
                        acquire_signal="acquire",
                        integration_kernel=kernels,
                        acquire_delay=acquire_delay,
                        integration_length=readout_pulse.length - acquire_delay,
                        handle="ac_1",
                        reset_delay=reset_delay,
                    )

                if measure_f:
                    with exp.section(uid="qubit_ge_excitation_1", play_after="readout_e"):
                        exp.play(signal="qb_drive", pulse=ge_X180)

                    with exp.section(uid="qubit_ef_excitation_0", play_after="qubit_ge_excitation_1", on_system_grid=True):
                        exp.play(signal="qb_ef_drive", pulse=ef_X180)

                    with exp.section(uid="readout_f", play_after="qubit_ef_excitation_0"):
                        exp.measure(
                            measure_signal="measure",
                            measure_pulse=readout_pulse,
                            measure_pulse_amplitude=amp_swp,
                            acquire_signal="acquire",
                            integration_kernel=kernels,
                            acquire_delay=acquire_delay,
                            integration_length=readout_pulse.length - acquire_delay,
                            handle="ac_2",
                            reset_delay=2*reset_delay,
                        )

    # setup calibration and signal map for the experiment
    sig_freq_map = create_default_map_and_calibration(
        exp, serial_num, qubit_parameters, lo_settings
    )
    sig_freq_map[serial_num]['QA0']["measure/acquire"]["frequency"] = freq_swp

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