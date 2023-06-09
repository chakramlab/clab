import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


class RamseyProgram(RAveragerProgram):

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        self.res_ch = cfg.device.soc.resonator.ch
        self.qubit_ch = cfg.device.soc.qubit.ch
        

        self.q_rp = self.ch_page(self.qubit_ch)     # get register page for qubit_ch
        self.r_wait = 3# self.sreg(self.qubit_ch, "time")
        self.safe_regwi(self.q_rp, self.r_wait, self.us2cycles(cfg.expt.start))

        self.r_phase2 = 4
        self.r_phase = self.sreg(self.qubit_ch, "phase")
        #self.r_phase = 0
        self.safe_regwi(self.q_rp, self.r_wait, self.us2cycles(cfg.expt.start))
        self.safe_regwi(self.q_rp, self.r_phase2, 0)
        
        self.f_res=self.freq2reg(cfg.device.soc.resonator.freq, gen_ch=self.res_ch, ro_ch=cfg.device.soc.readout.ch[0])  # convert f_res to dac register value
        self.readout_length=self.us2cycles(cfg.device.soc.readout.length)
        # self.cfg["adc_lengths"]=[self.readout_length]*2     #add length of adc acquisition to config
        # self.cfg["adc_freqs"]=[adcfreq(cfg.device.soc.readout.frequency)]*2   #add frequency of adc ddc to config
        
        self.piby2sigma = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ge.sigma)
        # print(self.sigma)

        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist) 
        self.declare_gen(ch=self.qubit_ch, nqz=self.cfg.device.soc.qubit.nyqist)

        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.soc.resonator.freq, gen_ch=self.res_ch)


        # add qubit and readout pulses to respective channels
        self.set_pulse_registers(ch=self.qubit_ch, style="const", 
                             freq=self.freq2reg(cfg.device.soc.qubit.f_ge), phase=0,
                                gain=cfg.device.soc.qubit.pulses.pi_ge.gain, 
                            length=self.piby2sigma)
        # self.add_gauss(ch=self.qubit_ch, name="qubit", sigma=self.piby2sigma, length= self.piby2sigma* 4)
        # self.set_pulse_registers(
        #     ch=self.qubit_ch,
        #     style="arb",
        #     freq=self.freq2reg(cfg.device.soc.qubit.f_ge),
        #     phase=self.deg2reg(0),
        #     gain=cfg.device.soc.qubit.pulses.pi_ge.gain,
        #     waveform="qubit")
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.soc.resonator.phase, gen_ch=self.res_ch),
            gain=cfg.device.soc.resonator.gain,
            length=self.readout_length)
        self.sync_all(self.us2cycles(0.2))

###
    # def initialize(self):
    #     cfg = AttrDict(self.cfg)
    #     self.cfg.update(cfg.expt)

    #     self.res_ch = cfg.device.soc.resonator.ch
    #     self.qubit_ch = cfg.device.soc.qubit.ch

    #     self.q_rp = self.ch_page(self.qubit_ch)  # get register page for qubit_ch
    #     self.r_wait = 3
    #     self.r_phase2 = 4
    #     # self.r_phase = self.sreg(self.qubit_ch, "phase")
    #     self.r_phase = 0
    #     self.safe_regwi(self.q_rp, self.r_wait, self.us2cycles(cfg.expt.start))
    #     self.safe_regwi(self.q_rp, self.r_phase2, 0)

    #     self.f_res = self.freq2reg(cfg.device.soc.resonator.freq)  # convert f_res to dac register value
    #     self.readout_length = self.us2cycles(cfg.device.soc.readout.length)
    #     # self.cfg["adc_lengths"] = [self.readout_length] * 2  # add length of adc acquisition to config
    #     # self.cfg["adc_freqs"] = [adcfreq(cfg.device.readout.frequency)] * 2  # add frequency of adc ddc to config

    #     self.sigma = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ge.sigma)
    #     # print(self.sigma)

    #     self.declare_gen(ch=self.res_ch, nqz=cfg.device.soc.resonator.nyqist)
    #     self.declare_gen(ch=self.qubit_ch, nqz=cfg.device.soc.qubit.nyqist)

    #     for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
    #         self.declare_readout(ch=ch, length=self.readout_length,
    #                              freq=cfg.device.soc.responator.freq, gen_ch=self.res_ch)

    #     # add qubit and readout pulses to respective channels
    #     self.add_gauss(ch=self.qubit_ch, name="qubit", sigma=self.sigma, length=self.sigma * 4)
    #     self.set_pulse_registers(
    #         ch=self.qubit_ch,
    #         style="arb",
    #         freq=self.freq2reg(cfg.device.soc.qubit.f_ge),
    #         phase=self.deg2reg(0),
    #         gain=cfg.device.soc.qubit.pulses.pi2_ge.gain,
    #         waveform="qubit")
    #     self.set_pulse_registers(
    #         ch=self.res_ch,
    #         style="const",
    #         freq=self.f_res,
    #         phase=self.deg2reg(cfg.device.soc.resonator.phase, gen_ch=self.res_ch),
    #         gain=cfg.device.soc.resonator.gain,
    #         length=self.readout_length)
    #     self.sync_all(self.us2cycles(0.2))

    def body(self):
        cfg = AttrDict(self.cfg)
        self.safe_regwi(self.q_rp, self.r_phase, 0)
        self.pulse(ch=self.qubit_ch)  # play pi/2 pulse
        self.sync_all()
        self.sync(self.q_rp, self.r_wait)
        self.mathi(self.q_rp, self.r_phase, self.r_phase2, "+", 0)
        self.pulse(ch=self.qubit_ch)  # play pi/2 pulse
        self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns
        self.measure(pulse_ch=self.res_ch,
                     adcs=[1, 0],
                     adc_trig_offset=cfg.device.soc.readout.adc_trig_offset,
                     wait=True,
                     syncdelay=self.us2cycles(cfg.device.soc.readout.relax_delay))  # sync all channels

    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+',
                   self.us2cycles(self.cfg.expt.step))  # update the time between two π/2 pulses
        self.mathi(self.q_rp, self.r_phase2, self.r_phase2, '+',
                   self.deg2reg(self.cfg.expt.phase_step, gen_ch=self.qubit_ch))  # advance the phase of the LO for the second π/2 pulse


class RamseyExperiment(Experiment):
    """Ramsey Experiment
       Experimental Config
        expt = {"start":0, "step": 1, "expts":200, "reps": 10, "rounds": 200, "phase_step": deg2reg(360/50)}
         }
    """

    def __init__(self, path='', prefix='Ramsey', config_file=None, progress=None):
        super().__init__(path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False, data_path=None, filename=None):
        fpts = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        ramsey = RamseyProgram(soc, self.cfg)
        print(self.im[self.cfg.aliases.soc], 'test0')
        x_pts, avgi, avgq = ramsey.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress, debug=debug)
        
        data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq}
        
        # x_pts, avgi, avgq = ramsey.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,
        #                                    progress=progress, debug=debug)

        # data = {'xpts': soc.cycles2us(x_pts), 'avgi': avgi, 'avgq': avgq}

        self.data = data

        data_dict = {'xpts':data['xpts'], 'avgi':data['avgi'][0][0], 'avgq':data['avgq'][0][0]}
        if data_path and filename:
            self.save_data(data_path=data_path, filename=filename, arrays=data_dict)
        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        pI = dsfit.fitdecaysin(data['xpts'], data['avgi'][0][0], fitparams=None, showfit=False)
        pQ = dsfit.fitdecaysin(data['xpts'], data['avgq'][0][0], fitparams=None, showfit=False)

        data['fiti'] = pI
        data['fitq'] = pQ
        corr_freq = (self.cfg.device.qubit.f_ge) - data['fiti'][1]
        data['corr_freq'] = corr_freq

        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data = self.data
        print(self.fname)

            # Writing in progress, may want to edit in the future to plot time instead of clock cycles
        plt.figure(figsize=(10, 8))
        plt.subplot(211, title="Ramsey", ylabel="I")
        plt.plot(data["xpts"], data["avgi"][0][0], 'o')
        if "fiti" in data:
            plt.plot(data["xpts"], dsfit.decaysin(data["fiti"], data["xpts"]))
        plt.subplot(212, xlabel="Delay (us)", ylabel="Q")
        plt.plot(data["xpts"], data["avgq"][0][0], 'o')
        if "fitq" in data:
            plt.plot(data["xpts"], dsfit.decaysin(data["fitq"], data["xpts"]))

        plt.tight_layout()
        plt.show()



