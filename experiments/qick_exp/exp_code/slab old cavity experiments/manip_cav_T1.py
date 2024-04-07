import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


class ManipCavT1Program(RAveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        self.res_ch = cfg.hw.soc.dacs[cfg.device.readout.dac].ch
        self.qubit_ch = cfg.hw.soc.dacs[cfg.device.qubit.dac].ch
        self.manip_ch = cfg.hw.soc.dacs[cfg.device.manipulate_cav.dac].ch

        self.q_rp = self.ch_page(self.qubit_ch)  # get register page for qubit_ch
        self.r_wait = 3
        self.safe_regwi(self.q_rp, self.r_wait, self.us2cycles(cfg.expt.start))

        self.f_res = self.freq2reg(cfg.device.readout.frequency)  # convert f_res to dac register value
        self.readout_length = self.us2cycles(cfg.device.readout.readout_length)
        # self.cfg["adc_lengths"] = [self.readout_length] * 2  # add length of adc acquisition to config
        # self.cfg["adc_freqs"] = [adcfreq(cfg.device.readout.frequency)] * 2  # add frequency of adc ddc to config

        self.sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge_resolved.sigma)
        # print(self.f_start, self.f_step)

        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist)
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist)
        self.declare_gen(ch=self.manip_ch, nqz=cfg.hw.soc.dacs.manipulate_cav.nyquist)

        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        # add qubit and readout pulses to respective channels
        self.set_pulse_registers(
            ch=self.manip_ch,
            style="const",
            freq=self.freq2reg(cfg.device.manipulate_cav.freqs[cfg.expt.mode_index]),
            phase=0,
            gain=cfg.expt.gain,
            length=self.us2cycles(cfg.expt.drive_length))
        self.add_gauss(ch=self.qubit_ch, name="qubit", sigma=self.sigma, length=self.sigma * 4)
        self.set_pulse_registers(
            ch=self.qubit_ch,
            style="arb",
            freq=self.freq2reg(cfg.device.qubit.f_ge),
            phase=self.deg2reg(0),
            gain=cfg.device.qubit.pulses.pi_ge_resolved.gain,
            waveform="qubit")
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
        if cfg.expt.drive_length > 0:
            self.pulse(ch=self.manip_ch)  # play manipulate cavity pulse
        # print(cfg.device.manipulate_cav.freqs[cfg.expt.mode_index])
        self.sync_all()  # align channels
        self.sync(self.q_rp, self.r_wait)

        if cfg.expt.measure_0:
            self.pulse(ch=self.qubit_ch)
        else:
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(cfg.device.qubit.f_ge) + self.freq2reg(2*cfg.device.manipulate_cav.chis[cfg.expt.mode_index]),
                phase=self.deg2reg(0),
                gain=cfg.device.qubit.pulses.pi_ge_resolved.gain,
                waveform="qubit")
            self.pulse(ch=self.qubit_ch)
        # print(cfg.device.qubit.f_ge, cfg.device.qubit.pulses.pi_ge_resolved.gain)
        self.sync_all()  # align channels and wait 50ns
        self.measure(pulse_ch=self.res_ch,
                     adcs=[1, 0],
                     adc_trig_offset=cfg.device.readout.trig_offset,
                     wait=True,
                     syncdelay=self.us2cycles(cfg.device.readout.relax_delay))  # sync all channels

    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step))  # update wait time


class ManipCavT1Experiment(Experiment):
    """Manipulate Cavity T1 Experiment
       Experimental Config
        expt = {"start": 0, "step": 0.05, "expts": 200, "reps": 20,"rounds": 100,
          "drive_length": 1, "gain": 10000, "mode_index": 0, "measure_0": True}
    """

    def __init__(self, path='', prefix='ManipCavT1', config_file=None, progress=None):
        super().__init__(path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        manip_t1 = ManipCavT1Program(soc, self.cfg)
        x_pts, avgi, avgq = manip_t1.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,
                                             progress=progress, debug=debug)

        data = {'xpts': x_pts, 'avgi': avgi, 'avgq': avgq}

        self.data = data

        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        pI = dsfit.fitexp(data['xpts'], data['avgi'][0][0], fitparams=None, showfit=False)
        pQ = dsfit.fitexp(data['xpts'], data['avgq'][0][0], fitparams=None, showfit=False)
        # adding this due to extra parameter in decaysin that is not in fitdecaysin
        pI = np.append(pI, data['xpts'][0])
        pQ = np.append(pQ, data['xpts'][0])
        data['fiti'] = pI
        data['fitq'] = pQ
        print(data['fiti'], data['fitq'])

        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data = self.data

        # Writing in progress, may want to edit in the future to plot time instead of clock cycles
        plt.figure(figsize=(10, 8))
        plt.subplot(211, title="Manipulate T1", ylabel="I")
        plt.plot(data["xpts"], data["avgi"][0][0], 'o-')
        if "fiti" in data:
            plt.plot(data["xpts"], dsfit.expfunc(data["fiti"], data["xpts"]))
        plt.subplot(212, xlabel="Wait Time (us)", ylabel="Q")
        plt.plot(data["xpts"], data["avgq"][0][0], 'o-')
        if "fitq" in data:
            plt.plot(data["xpts"], dsfit.expfunc(data["fitq"], data["xpts"]))

        plt.tight_layout()
        plt.show()
