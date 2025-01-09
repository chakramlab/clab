import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

# This program uses the RAveragerProgram class, which allows you to sweep a parameter directly on the processor 
# rather than in python as in the above example
# Because the whole sweep is done on the processor there is less downtime (especially for fast experiments)

class f0g1SpectroscopyiaqubitProgram(RAveragerProgram):
    def initialize(self):
        cfg=AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        # self.res_ch=cfg.hw.soc.dacs[cfg.device.readout.dac].ch
        # self.qubit_ch=cfg.hw.soc.dacs[cfg.device.qubit.dac].ch
        
        self.res_ch = cfg.device.soc.resonator.ch
        self.qubit_ch = cfg.device.soc.qubit.ch

        self.q_rp=self.ch_page(self.qubit_ch)     # get register page for qubit_ch
        self.r_freq=self.sreg(self.qubit_ch, "freq")   # get frequency register for qubit_ch 
        self.r_freq2 = 4
        
        self.f_res=self.freq2reg(cfg.device.soc.readout.freq, gen_ch=self.res_ch, ro_ch=self.cfg.device.soc.readout.ch[0])            # conver f_res to dac register value
        self.readout_length=self.us2cycles(cfg.device.soc.readout.readout_length)
        # self.cfg["adc_lengths"]=[self.readout_length]*2     #add length of adc acquisition to config
        # self.cfg["adc_freqs"]=[adcfreq(cfg.device.readout.frequency)]*2   #add frequency of adc ddc to config
        
        self.f_start = self.freq2reg(cfg.expt.start)
        self.f_step = self.freq2reg(cfg.expt.step)
        
        self.safe_regwi(self.q_rp, self.r_freq2, self.f_start)  # set r_freq2 to start frequency

        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist)
        self.declare_gen(ch=self.qubit_ch, nqz=self.cfg.device.soc.qubit.nyqist)

        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, 
                                 length=self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0]),
                                 freq=cfg.device.soc.readout.freq, 
                                 gen_ch=self.cfg.device.soc.resonator.ch)
        
        # add qubit and readout pulses to respective channels

        self.sigma_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ef = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)

        self.add_gauss(ch=self.qubit_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)
        self.add_gauss(ch=self.qubit_ch, name="qubit_ef", sigma=self.sigma_ef, length=self.sigma_ef * 4)

        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.soc.readout.phase, gen_ch=self.res_ch),
            gain=cfg.device.soc.resonator.gain,
            length=self.readout_length)

        self.sync_all(self.us2cycles(0.2))

    
    def body(self):
        cfg=AttrDict(self.cfg)

        self.set_pulse_registers(
            ch=self.qubit_ch,
            style="arb",
            freq=self.freq2reg(cfg.device.soc.qubit.f_ge),
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain,
            waveform="qubit_ge")
        
        self.pulse(ch=self.qubit_ch)

        self.sync_all()

        self.set_pulse_registers(
            ch=self.qubit_ch,
            style="arb",
            freq=self.freq2reg(cfg.device.soc.qubit.f_ef),
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain,
            waveform="qubit_ef")
        self.pulse(ch=self.qubit_ch)

        self.sync_all()

        # setup and play f0g1 sideband probe pulse

        self.set_pulse_registers(
            ch=self.qubit_ch,
            style="const",
            freq=0,  # freq set by update
            phase=0,
            gain=cfg.expt.gain,
            length=self.us2cycles(cfg.expt.length))
        self.mathi(self.q_rp, self.r_freq, self.r_freq2, "+", 0)

        self.pulse(ch=self.qubit_ch)

        if cfg.expt.add_pi_ef:
            self.sync_all()
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(cfg.device.soc.qubit.f_ef),
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain,
                waveform="qubit_ef")
            self.pulse(ch=self.qubit_ch)

        self.sync_all(self.us2cycles(0.05))



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
        self.mathi(self.q_rp, self.r_freq2, self.r_freq2, '+', self.f_step) # update frequency list index
        

class f0g1SpectroscopyiaqubitExperiment(Experiment):
    """f0-g1 spectroscopy experiment
       Experimental Config
        expt={"start":4020, "step":0.35, "expts":300, "reps": 200,"rounds":50,
          "length":5, "gain":300
          00
         }
    """

    def __init__(self, path='', prefix='PulseProbeEFSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False, data_path=None, filename=None):
        
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        
        qspec_ef=f0g1SpectroscopyiaqubitProgram(soc, self.cfg)
        
        x_pts, avgi, avgq = qspec_ef.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress, debug=debug)
        # x_pts, avgi, avgq = qspec_ef.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress, debug=debug)        
        
        data={'xpts':x_pts, 'avgi':avgi, 'avgq':avgq}
        
        self.data=data

        data_dict = {'xpts':data['xpts'], 'avgi':data['avgi'][0][0], 'avgq':data['avgq'][0][0]}
        if data_path and filename:
            self.save_data(data_path=data_path, filename=filename, arrays=data_dict)
        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
        
        data['fiti']=dsfit.fitlor(data["xpts"],data['avgi'][0][0])
        data['fitq']=dsfit.fitlor(data["xpts"],data['avgq'][0][0])
        print(data['fiti'], data['fitq'])
        
        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data=self.data 
        
        print (self.fname)
        plt.figure(figsize=(10,8))
        plt.subplot(211,title="Pulse Probe EF Spectroscopy",  ylabel="I")
        plt.plot(data["xpts"], data["avgi"][0][0],'o-')
        if "fiti" in data:
            plt.plot(data["xpts"], dsfit.lorfunc(data["fiti"], data["xpts"]))
        plt.subplot(212, xlabel="Pulse Frequency (MHz)", ylabel="Q")
        plt.plot(data["xpts"], data["avgq"][0][0],'o-')
        if "fitq" in data:
            plt.plot(data["xpts"], dsfit.lorfunc(data["fitq"], data["xpts"]))
            
        plt.tight_layout()
        plt.show()
        