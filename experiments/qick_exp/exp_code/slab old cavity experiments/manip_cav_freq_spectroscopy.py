import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


class ManipCavFreqSpecProgram(RAveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        self.res_ch = cfg.hw.soc.dacs[cfg.device.readout.dac].ch
        self.qubit_ch = cfg.hw.soc.dacs[cfg.device.qubit.dac].ch
        self.manip_ch = cfg.hw.soc.dacs[cfg.device.manipulate_cav.dac].ch
        
        self.mc_rp = self.ch_page(self.manip_ch)     # get register page for manip_ch
        self.mc_freq=self.sreg(self.manip_ch, "freq")
        
        self.f_res=self.freq2reg(cfg.device.readout.frequency)  # convert f_res to dac register value
        self.readout_length=self.us2cycles(cfg.device.readout.readout_length)
        # self.cfg["adc_lengths"]=[self.readout_length]*2     #add length of adc acquisition to config
        # self.cfg["adc_freqs"]=[adcfreq(cfg.device.readout.frequency)]*2   #add frequency of adc ddc to config
        
        self.sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge_resolved.sigma)
        self.sigma_fast = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma)
        self.drive_length = self.us2cycles(cfg.expt.drive_length)
        self.f_start = self.freq2reg(cfg.expt.start)
        self.f_step = self.freq2reg(cfg.expt.step)
        
        self.safe_regwi(self.mc_rp, self.mc_freq, self.f_start)  # send start frequency to r_freq2
        # print(self.sigma)

        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist)
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist)
        self.declare_gen(ch=self.manip_ch, nqz=cfg.hw.soc.dacs.manipulate_cav.nyquist)

        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.readout.frequency, gen_ch=self.res_ch)
        
        # add qubit, manip, and readout pulses to respective channels
        if cfg.expt.pulse_type == "gauss":
            self.add_gauss(ch=self.manip_ch, name="manip", sigma=cfg.expt.drive_length, length=cfg.expt.drive_length * 4)
            self.set_pulse_registers(
                ch=self.manip_ch,
                style="arb",
                freq=self.freq2reg(cfg.expt.start),  # set in body
                phase=0,
                gain=cfg.expt.gain,
                waveform="manip")
        else:
            self.set_pulse_registers(
                ch=self.manip_ch,
                style="const",
                freq=self.freq2reg(cfg.expt.start),
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
        if cfg.expt.pi_qubit_first:
            self.add_gauss(ch=self.qubit_ch, name="qubit_fast", sigma=self.sigma_fast, length=self.sigma_fast * 4)
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(cfg.device.qubit.f_ge),
                phase=self.deg2reg(0),
                gain=cfg.device.qubit.pulses.pi_ge.gain,
                waveform="qubit_fast")
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.readout.phase, gen_ch=self.res_ch),
            gain=cfg.device.readout.gain,
            length=self.readout_length)
        
        self.sync_all(self.us2cycles(0.2))
    
    def body(self):
        cfg=AttrDict(self.cfg)
        if cfg.expt.pi_qubit_first:
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(cfg.device.qubit.f_ge),
                phase=self.deg2reg(0),
                gain=cfg.device.qubit.pulses.pi_ge.gain,
                waveform="qubit_fast")
            self.pulse(ch=self.qubit_ch)
        self.sync_all()
        self.pulse(ch=self.manip_ch)  # play manipulate cavity pulse
        self.sync_all(self.us2cycles(0.01))  # align channels and wait 10ns
        self.set_pulse_registers(
            ch=self.qubit_ch,
            style="arb",
            freq=self.freq2reg(cfg.device.qubit.f_ge),
            phase=self.deg2reg(0),
            gain=cfg.device.qubit.pulses.pi_ge_resolved.gain,
            waveform="qubit")
        self.pulse(ch=self.qubit_ch)
        self.sync_all(self.us2cycles(0.01)) # align channels and wait 10ns
        self.measure(pulse_ch=self.res_ch,
                     adcs=[1, 0],
                     adc_trig_offset=cfg.device.readout.trig_offset,
                     wait=True,
                     syncdelay=self.us2cycles(cfg.device.readout.relax_delay))  # sync all channels
    
    def update(self):
        self.mathi(self.mc_rp, self.mc_freq, self.mc_freq, '+', self.f_step) # update frequency
                      
                      
class ManipCavFreqSpecExperiment(Experiment):
    """Manipulate Cavity Frequency Spectroscopy Experiment
       Experimental Config
        expt = {"start": 5156.5, "step": 0.01, "expts": 200, "reps": 20,"rounds": 100,
          "drive_length": 15, "gain": 10000, "pulse_type" :'const', 'pi_qubit_first': False}
    """

    def __init__(self, path='', prefix='ManipCavFreqSpec', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        manip_freq=ManipCavFreqSpecProgram(soc, self.cfg)
        x_pts, avgi, avgq = manip_freq.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress, debug=debug)        
        
        data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq}
        
        self.data=data

        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
        
        data['fiti']=dsfit.fitlor(data["xpts"],-data['avgi'][0][0])
        data['fitq']=dsfit.fitlor(data["xpts"],-data['avgq'][0][0])
        print(data['fiti'], data['fitq'])
        
        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data=self.data 
        
        print (self.fname)
        plt.figure(figsize=(10,8))
        plt.subplot(211,title="Manipulate Cavity Frequency Spectroscopy",  ylabel="I")
        plt.plot(data["xpts"], data["avgi"][0][0],'o-')
        if "fiti" in data:
            plt.plot(data["xpts"], -dsfit.lorfunc(data["fiti"], data["xpts"]))
        plt.subplot(212, xlabel="Cavity Pulse Frequency (MHz)", ylabel="Q")
        plt.plot(data["xpts"], data["avgq"][0][0],'o-')
        if "fitq" in data:
            plt.plot(data["xpts"], -dsfit.lorfunc(data["fitq"], data["xpts"]))
            
        plt.tight_layout()
        plt.show()
        