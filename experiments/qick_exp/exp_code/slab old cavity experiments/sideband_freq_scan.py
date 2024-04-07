import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


class SidebandFreqScanProgram(RAveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        self.res_ch = cfg.hw.soc.dacs[cfg.device.readout.dac].ch
        self.qubit_ch = cfg.hw.soc.dacs[cfg.device.qubit.dac].ch
        self.sideband_ch = cfg.hw.soc.dacs[cfg.device.sideband.dac].ch

        self.sb_rp = self.ch_page(self.sideband_ch)  # get register page for manip_ch
        self.sb_freq = self.sreg(self.sideband_ch, "freq")

        self.f_res = self.freq2reg(cfg.device.readout.frequency)  # convert f_res to dac register value
        self.readout_length = self.us2cycles(cfg.device.readout.readout_length)
        # self.cfg["adc_lengths"] = [self.readout_length] * 2  # add length of adc acquisition to config
        # self.cfg["adc_freqs"] = [adcfreq(cfg.device.readout.frequency)] * 2  # add frequency of adc ddc to config

        self.sigma_ef = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma)
        self.sigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma)
        self.drive_length = self.us2cycles(cfg.expt.drive_length)
        self.f_start = self.freq2reg(cfg.expt.start)
        self.f_step = self.freq2reg(cfg.expt.step)

        self.safe_regwi(self.sb_rp, self.sb_freq, self.f_start)  # send start frequency to r_freq2
        # print(self.sigma)

        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist)
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist)
        self.declare_gen(ch=self.sideband_ch, nqz=cfg.hw.soc.dacs.sideband.nyquist)

        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        # add qubit and readout pulses to respective channels
        if cfg.expt.pulse_type == "gauss":
            self.add_gauss(ch=self.sideband_ch, name="sideband", sigma=cfg.expt.drive_length,
                           length=cfg.expt.drive_length * 4)
            self.set_pulse_registers(
                ch=self.sideband_ch,
                style="arb",
                freq=self.freq2reg(cfg.expt.start),  # set in body
                phase=0,
                gain=cfg.expt.gain,
                waveform="sideband")
        else:
            self.set_pulse_registers(
                ch=self.sideband_ch,
                style="const",
                freq=self.freq2reg(cfg.expt.start),
                phase=0,
                gain=cfg.expt.gain,
                length=self.us2cycles(cfg.expt.drive_length))

        self.add_gauss(ch=self.qubit_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)
        self.add_gauss(ch=self.qubit_ch, name="qubit_ef", sigma=self.sigma_ef, length=self.sigma_ef * 4)

        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.readout.phase, gen_ch=self.res_ch),
            gain=cfg.device.readout.gain,
            length=self.readout_length)

        self.sync_all(self.us2cycles(0.2))

    def body(self):
        cfg = AttrDict(self.cfg)

        self.set_pulse_registers(
            ch=self.qubit_ch,
            style="arb",
            freq=self.freq2reg(cfg.device.qubit.f_ge),
            phase=self.deg2reg(0),
            gain=self.cfg.device.qubit.pulses.pi_ge.gain,
            waveform="qubit_ge")
        self.pulse(ch=self.qubit_ch)
        self.set_pulse_registers(
            ch=self.qubit_ch,
            style="arb",
            freq=self.freq2reg(self.cfg.device.qubit.f_ef),
            phase=0,
            gain=self.cfg.device.qubit.pulses.pi_ef.gain,  # gain set by update
            waveform="qubit_ef")
        self.pulse(ch=self.qubit_ch)

        self.sync_all(self.us2cycles(0.01))
        self.pulse(ch=self.sideband_ch)  # play sideband pulse
        self.sync_all(self.us2cycles(0.01))  # align channels and wait 50ns

        self.set_pulse_registers(
            ch=self.qubit_ch,
            style="arb",
            freq=self.freq2reg(self.cfg.device.qubit.f_ef),
            phase=0,
            gain=self.cfg.device.qubit.pulses.pi_ef.gain,  # gain set by update
            waveform="qubit_ef")
        self.pulse(ch=self.qubit_ch)

        self.sync_all(self.us2cycles(0.01))  # align channels and wait 10ns
        self.measure(pulse_ch=self.res_ch,
                     adcs=[1, 0],
                     adc_trig_offset=cfg.device.readout.trig_offset,
                     wait=True,
                     syncdelay=self.us2cycles(cfg.device.readout.relax_delay))  # sync all channels

    def update(self):
        self.mathi(self.sb_rp, self.sb_freq, self.sb_freq, '+', self.f_step)  # update frequency


class SidebandFreqScanExperiment(Experiment):
    """Sideband Freq Scan Experiment
       Experimental Config
        expt = {"start": 5156.5, "step": 0.01, "expts": 200, "reps": 20,"rounds": 100,
          "drive_length": 15, "gain": 10000, "pulse_type" :'const'}
    """

    def __init__(self, path='', prefix='SidebandFreqScan', config_file=None, progress=None):
        super().__init__(path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        sideband_freq = SidebandFreqScanProgram(soc, self.cfg)
        x_pts, avgi, avgq = sideband_freq.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,
                                               progress=progress, debug=debug)

        data = {'xpts': x_pts, 'avgi': avgi, 'avgq': avgq}

        self.data = data

        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        data['fiti'] = dsfit.fitlor(data["xpts"], -data['avgi'][0][0])
        data['fitq'] = dsfit.fitlor(data["xpts"], -data['avgq'][0][0])

        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data = self.data

        print(self.fname)
        plt.figure(figsize=(10, 8))
        plt.subplot(211, title="f0g1 Sideband Frequency Spectroscopy", ylabel="I")
        plt.plot(data["xpts"], data["avgi"][0][0], 'o-')
        if "fiti" in data:
            plt.plot(data["xpts"], -dsfit.lorfunc(data["fiti"], data["xpts"]))
        plt.subplot(212, xlabel="Cavity Pulse Frequency (MHz)", ylabel="Q")
        plt.plot(data["xpts"], data["avgq"][0][0], 'o-')
        if "fitq" in data:
            plt.plot(data["xpts"], -dsfit.lorfunc(data["fitq"], data["xpts"]))

        plt.tight_layout()
        plt.show()
