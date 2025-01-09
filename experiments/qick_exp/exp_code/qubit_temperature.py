import matplotlib.pyplot as plt
import numpy as np
from qick import *

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


class AmplitudeRabiEFProgram(RAveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        # self.res_ch = cfg.hw.soc.dacs[cfg.device.readout.dac].ch
        # self.qubit_ch = cfg.hw.soc.dacs[cfg.device.qubit.dac].ch

        self.res_ch = cfg.device.soc.resonator.ch
        self.qubit_ch = cfg.device.soc.qubit.ch

        self.q_rp = self.ch_page(self.qubit_ch)  # get register page for qubit_ch
        self.r_gain = self.sreg(self.qubit_ch, "gain")  # get gain register for qubit_ch
        self.r_gain2 = 4

        self.f_res=self.freq2reg(cfg.device.soc.readout.freq)  # conver f_res to dac register value
        self.readout_length=self.us2cycles(cfg.device.soc.readout.readout_length)


        self.sigma_test = self.us2cycles(cfg.expt.sigma_test)
        self.sigma = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma)
        # print(self.sigma_test)
        # initialize gain
        self.safe_regwi(self.q_rp, self.r_gain2, self.cfg.expt.start)

        # self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist)
        # self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist)

        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist)
        self.declare_gen(ch=self.qubit_ch, nqz=self.cfg.device.soc.qubit.nyqist)

        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, 
                                 length=self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0]),
                                 freq=cfg.device.soc.readout.freq, 
                                 gen_ch=self.cfg.device.soc.resonator.ch)



        # add qubit and readout pulses to respective channels
        self.add_gauss(ch=self.qubit_ch, name="qubit_pi", sigma=self.sigma, length=self.sigma * 4)
        self.add_gauss(ch=self.qubit_ch, name="qubit_ef", sigma=self.sigma_test, length=self.sigma_test * 4)
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.soc.readout.phase, gen_ch=self.res_ch),
            gain=cfg.device.soc.resonator.gain,
            length=self.readout_length)

        self.sync_all(self.us2cycles(0.2))


    def body(self):
        cfg = AttrDict(self.cfg)
        if cfg.expt.pi_qubit:
            #applying a pi pulse to go to e state of qubit
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(cfg.device.soc.qubit.f_ge),
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain,
                waveform="qubit_pi")
            self.pulse(ch=self.qubit_ch)

        # setup and play ef probe pulse
        self.set_pulse_registers(
            ch=self.qubit_ch,
            style="arb",
            freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef),
            phase=0,
            gain=0,  # gain set by update
            waveform="qubit_ef")
        self.mathi(self.q_rp, self.r_gain, self.r_gain2, '+', 0)
        self.pulse(ch=self.qubit_ch)


        if cfg.expt.ge_pi_after:
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(cfg.device.soc.qubit.f_ge),
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain,
                waveform="qubit_pi")
            self.pulse(ch=self.qubit_ch)
        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        # Readout kick pulse

        if self.cfg.device.soc.readout.kick_pulse:
            print('Playing kick pulse')
            self.set_pulse_registers(
                ch=self.cfg.device.soc.resonator.ch,
                style="const",
                freq=self.freq2reg(self.cfg.device.soc.readout.freq, gen_ch=self.cfg.device.soc.resonator.ch, ro_ch=self.cfg.device.soc.readout.ch[0]),
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.readout.kick_pulse_gain,
                length=self.us2cycles(self.cfg.device.soc.readout.kick_pulse_length))
            
            self.pulse(ch=self.cfg.device.soc.resonator.ch)
            self.sync_all()

        # Readout 

        self.set_pulse_registers(
            ch=self.cfg.device.soc.resonator.ch,
            style="const",
            freq=self.freq2reg(self.cfg.device.soc.readout.freq, gen_ch=self.cfg.device.soc.resonator.ch, ro_ch=self.cfg.device.soc.readout.ch[0]),
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.resonator.gain,
            length=self.us2cycles(self.cfg.device.soc.readout.length, gen_ch=self.cfg.device.soc.resonator.ch))
        
        self.measure(pulse_ch=self.cfg.device.soc.resonator.ch,
                     adcs=[0],
                     adc_trig_offset=self.us2cycles(self.cfg.device.soc.readout.adc_trig_offset),
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg.device.soc.readout.relax_delay))  # sync all channels

    def update(self):
        self.mathi(self.q_rp, self.r_gain2, self.r_gain2, '+', self.cfg.expt.step)  # update frequency list index


class QubitTemperatureExperiment(Experiment):
    """Amplitude Rabi EF Experiment
       Experimental Config
        expt = {"start":0, "step": 150, "expts":200, "reps": 10, "rounds": 200, "sigma_test": 0.025, "pi_ge_after": True,
        "rounds_without_pi_first": 3000}
        }
    """

    def __init__(self, path='', prefix='QubitTemperature', config_file=None, progress=None):
        super().__init__(path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        fpts = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())

        self.cfg.expt.pi_qubit = True
        amprabi_ef = AmplitudeRabiEFProgram(soc, cfg=self.cfg)
        x_pts, avgi, avgq = amprabi_ef.acquire(self.im[self.cfg.aliases.soc], threshold=None,
                                               load_pulses=True, progress=progress, debug=debug)

        self.cfg.expt.pi_qu
        bit = False
        self.cfg.expt.rounds = self.cfg.expt.rounds_without_pi_first
        amprabi_ef2 = AmplitudeRabiEFProgram(soc, cfg=self.cfg)
        x_pts2, avgi2, avgq2 = amprabi_ef2.acquire(self.im[self.cfg.aliases.soc], threshold=None,
                                                   load_pulses=True, progress=progress, debug=debug)

        data = {'xpts': x_pts, 'avgi': avgi, 'avgq': avgq, "avgi2": avgi2, "avgq2": avgq2}

        self.data = data

        return data

    def analyze(self, data=None, **kwargs):
        def temperature_q(nu, rat):
            Kb = 1.38e-23
            h = 2 * np.pi * 1.054e-34
            return h * nu / (Kb * np.log(1 / rat))

        def occupation_q(nu, T):
            Kb = 1.38e-23
            h = 2 * np.pi * 1.054e-34
            return 1 / (np.exp(h * nu / (Kb * T)) + 1)

        if data is None:
            data = self.data

        # ex: fitparams=[1.5, 1/(2*15000), -np.pi/2, 1e8, -13, 0]
        pI = dsfit.fitdecaysin(data['xpts'], data['avgi'][0][0], fitparams=None, showfit=False)
        pQ = dsfit.fitdecaysin(data['xpts'], data['avgq'][0][0], fitparams=None, showfit=False)
        # adding this due to extra parameter in decaysin that is not in fitdecaysin
        pI = np.append(pI, data['xpts'][0])
        pQ = np.append(pQ, data['xpts'][0])
        fit_freq = pI[1]
        pI2 = dsfit.fitdecaysin_fix_freq(data['xpts'], data['avgi2'][0][0], fitparams=None, showfit=False, freq=fit_freq)
        pQ2 = dsfit.fitdecaysin_fix_freq(data['xpts'], data['avgq2'][0][0], fitparams=None, showfit=False, freq=fit_freq)
        # splitting into different categories to be able to save properly
        data['fiti'] = pI
        data['fiti2'] = np.insert(pI2, 1, fit_freq)
        data['fitq'] = pQ
        data['fitq2'] = np.insert(pQ2, 1, fit_freq)

        nu_q = self.cfg.device.qubit.f_ge
        ratio = abs(pI2[0] / pI[0])
        print("ge contrast ratio from I data = ", ratio)
        print("Qubit Temp:", 1e3 * temperature_q(nu_q * 1e6, ratio), " mK")
        print("Qubit Excited State Occupation:", occupation_q(nu_q, temperature_q(nu_q, ratio)))

        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data = self.data

        print(self.fname)
        plt.figure(figsize=(10, 8))
        plt.subplot(211, title="Qubit Temperature", ylabel="I")
        plt.plot(data["xpts"], data["avgi"][0][0], 'o-')
        plt.plot(data["xpts"], data["avgi2"][0][0], 'ro-')
        if "fiti" in data:
            plt.plot(data["xpts"], dsfit.decaysin(data["fiti"], data["xpts"]))
            plt.plot(data["xpts"], dsfit.decaysin(data["fiti2"], data["xpts"]))
        plt.subplot(212, xlabel="Gain", ylabel="Q")
        plt.plot(data["xpts"], data["avgq"][0][0], 'o-')
        plt.plot(data["xpts"], data["avgq2"][0][0], 'ro-')
        if "fitq" in data:
            plt.plot(data["xpts"], dsfit.decaysin(data["fitq"], data["xpts"]))
            plt.plot(data["xpts"], dsfit.decaysin(data["fitq2"], data["xpts"]))
        plt.tight_layout()
        plt.show()







    # def body(self):
    #     cfg = AttrDict(self.cfg)
    #     if cfg.expt.pi_qubit:
    #         self.pulse(ch=self.qubit_ch, name='pi_qubit', freq=freq2reg(cfg.device.qubit.f_ge),
    #                    gain=cfg.device.qubit.pulses.pi_ge.gain, play=True)
    #     self.mathi(self.q_rp, self.r_gain, self.r_gain2, '+', 0)
    #     self.pulse(ch=self.qubit_ch, name="qubit_ef", phase=deg2reg(0), freq=freq2reg(cfg.device.qubit.f_ef),
    #                play=True)  # ef qubit pulse
    #     if cfg.expt.ge_pi_after:
    #         self.pulse(ch=self.qubit_ch, name='pi_qubit', freq=freq2reg(cfg.device.qubit.f_ge),
    #                    gain=cfg.device.qubit.pulses.pi_ge.gain, play=True)
    #     self.sync_all(us2cycles(0.05))  # align channels and wait 50ns
    #     self.trigger_adc(adc1=1, adc2=0, adc_trig_offset=cfg.device.readout.trig_offset)  # trigger measurement
    #     self.pulse(ch=self.res_ch, name="measure", freq=self.f_res, phase=deg2reg(cfg.device.readout.phase),
    #                gain=cfg.device.readout.gain, play=True)  # play readout pulse
    #     self.waiti(self.res_ch, self.readout_length)
    #     self.sync_all(us2cycles(cfg.device.readout.relax_delay))  # wait for qubit to relax
    #
    # def update(self):
    #     self.mathi(self.q_rp, self.r_gain2, self.r_gain2, '+', self.cfg.expt.step)  # update frequency list index

