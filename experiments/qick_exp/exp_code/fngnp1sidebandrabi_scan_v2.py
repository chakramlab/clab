import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

# This program uses the RAveragerProgram class, which allows you to sweep a parameter directly on the processor 
# rather than in python as in the above example
# Because the whole sweep is done on the processor there is less downtime (especially for fast experiments)

class fngnp1SpectroscopyProgram(RAveragerProgram):
    def initialize(self):
        cfg=AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        self.res_ch = cfg.device.soc.resonator.ch
        self.qubit_ch = cfg.device.soc.qubit.ch
        try:
            print ("Configuring sideband channel to" + str(cfg.device.soc.sideband.ch))
            self.sideband_ch = cfg.device.soc.sideband.ch
        except:
            self.sideband_ch = self.qubit_ch
            # print ("No sideband channel specified, using qubit channel = " + str(self.sideband_ch))

        self.q_rp=self.ch_page(self.qubit_ch)     # get register page for qubit_ch
        self.r_freq=self.sreg(self.qubit_ch, "freq")   # get frequency register for qubit_ch 

        try:
            self.s_freq=self.sreg(self.sideband_ch, "freq")   # get frequency register for sideband_ch
        except:
            self.s_freq=self.r_freq

        if self.qubit_ch != self.sideband_ch:
            self.s_rp = self.ch_page(self.sideband_ch)     # get register page for sideband_ch
            print('Register page for sideband_ch set')
        else:
            self.s_rp = self.q_rp

        self.s_freq2 = 4
        
        self.f_res=self.freq2reg(cfg.device.soc.readout.freq, gen_ch=self.res_ch, ro_ch=self.cfg.device.soc.readout.ch[0])            # conver f_res to dac register value
        self.readout_length=self.us2cycles(cfg.device.soc.readout.readout_length)
        
        self.f_start = self.freq2reg(cfg.expt.start)
        self.f_step = self.freq2reg(cfg.expt.step)
        
        self.safe_regwi(self.s_rp, self.s_freq2, self.f_start)  # set s_freq2 to start frequency

        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist)
        self.declare_gen(ch=self.qubit_ch, nqz=self.cfg.device.soc.qubit.nyqist)
        if self.qubit_ch != self.sideband_ch:
            self.declare_gen(ch=self.sideband_ch, nqz=self.cfg.device.soc.sideband.nyqist)

        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.soc.readout.freq, gen_ch=self.res_ch)
        
        # add qubit and readout pulses to respective channels

        self.sigma_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ef = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)

        try: self.pulse_type_ge = cfg.device.soc.qubit.pulses.pi_ge.pulse_type
        except: self.pulse_type_ge = 'const'
        try: self.pulse_type_ef = cfg.device.soc.qubit.pulses.pi_ef.pulse_type
        except: self.pulse_type_ef = 'const'

        print('Pulse type_ge: ' + self.pulse_type_ge)
        print('Pulse type_ef: ' + self.pulse_type_ef)
            
        if self.pulse_type_ge == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)
        
        if self.pulse_type_ef == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ef", sigma=self.sigma_ef, length=self.sigma_ef * 4)

        if self.cfg.expt.fngnp1_pipulse_type == 'flat_top':
            for ii in range(self.cfg.expt.n):
                self.add_gauss(ch=self.sideband_ch, name="sb_flat_top_pi"+str(ii), sigma=self.us2cycles(self.cfg.expt.sb_sigma), length=self.us2cycles(self.cfg.expt.sb_sigma) * 4)

        if self.cfg.expt.fngnp1_probepulse_type == 'flat_top':
            self.add_gauss(ch=self.sideband_ch, name="sb_flat_top_probe", sigma=self.us2cycles(self.cfg.expt.sb_sigma), length=self.us2cycles(self.cfg.expt.sb_sigma) * 4)
                


        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.soc.readout.phase, gen_ch=self.res_ch),
            gain=cfg.device.soc.resonator.gain,
            length=self.readout_length)

        self.sync_all(self.us2cycles(0.2))


    def play_pige_pulse(self, phase = 0, shift = 0):

        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'const':

            self.set_pulse_registers(
                    ch=self.qubit_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift), 
                    phase=self.deg2reg(phase),
                    gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain, 
                    length=self.sigma_ge)
            
        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'gauss':
            
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift),
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain,
                waveform="qubit_ge")
        
        self.pulse(ch=self.qubit_ch)

    def play_pief_pulse(self, phase = 0, shift = 0):
            
        if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'const':

            self.set_pulse_registers(
                    ch=self.qubit_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef + shift), 
                    phase=self.deg2reg(phase),
                    gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain, 
                    length=self.sigma_ef)
            
        if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'gauss':
            
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef + shift),
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain,
                waveform="qubit_ef")
        
        self.pulse(ch=self.qubit_ch)   

    def play_pi_sb(self, n = 0, phase=0, shift=0):

        if self.cfg.expt.fngnp1_pipulse_type == 'const':
            
            # print('Sideband const')
            self.set_pulse_registers(
                ch=self.sideband_ch,
                style="const",
                freq=self.freq2reg(self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][n] + shift),  # freq set by update
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][n],
                length=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][n]))
            
        if self.cfg.expt.fngnp1_pipulse_type == 'flat_top':
            
            # print('Sideband flat top')
            self.set_pulse_registers(
                ch=self.sideband_ch,
                style="flat_top",
                freq=self.freq2reg(self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode][n] +shift),
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode][n],
                length=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode][n]),
                waveform="sb_flat_top_pi"+str(n))
        
        # self.mathi(self.s_rp, self.s_freq, self.s_freq2, "+", 0)
        self.pulse(ch=self.sideband_ch)

    def play_sb_probe(self, freq= 1, length=1, gain=1, phase=0, shift=0):

        if self.cfg.expt.fngnp1_probepulse_type == 'const':
            
            print('Sideband const')
            self.set_pulse_registers(
                    ch=self.sideband_ch, 
                    style="const", 
                    freq=self.freq2reg(freq+shift), 
                    phase=self.deg2reg(phase),
                    gain=gain, 
                    length=self.us2cycles(length))
        
        if self.cfg.expt.fngnp1_probepulse_type == 'flat_top':
            
            print('Sideband flat top')
            self.set_pulse_registers(
                ch=self.sideband_ch,
                style="flat_top",
                freq=self.freq2reg(freq+shift),
                phase=self.deg2reg(phase),
                gain=gain,
                length=self.us2cycles(length),
                waveform="sb_flat_top_probe")
        
        self.mathi(self.s_rp, self.s_freq, self.s_freq2, "+", 0)
        self.pulse(ch=self.sideband_ch)

    
    def body(self):
        cfg=AttrDict(self.cfg)

        # Put n photons into cavity 

        for i in np.arange(cfg.expt.n):

            # setup and play qubit ge pi pulse

            self.play_pige_pulse()
            self.sync_all()

            # setup and play qubit ef pi pulse

            self.play_pief_pulse()
            self.sync_all()

            # setup and play f,n g,n+1 sideband pi pulse

            self.play_pi_sb(n=i)
            self.sync_all()

        # setup and play qubit ge pi pulse

        self.play_pige_pulse()
        self.sync_all()

        # setup and play qubit ef pi pulse

        self.play_pief_pulse()
        self.sync_all()

        # setup and play f0g1 sideband probe pulse
        print('Playing probe pulse, length = ' + str(cfg.expt.length))
        self.play_sb_probe(freq=cfg.expt.start, length=cfg.expt.length, gain=cfg.expt.gain, phase=0, shift=0)
        self.sync_all()
        
        if cfg.expt.add_pi_ef:
            self.play_pief_pulse()

        self.sync_all(self.us2cycles(0.05))

        self.measure(pulse_ch=self.res_ch,
                     adcs=[1, 0],
                     adc_trig_offset=cfg.device.soc.readout.adc_trig_offset,
                     wait=True,
                     syncdelay=self.us2cycles(cfg.device.soc.readout.relax_delay))  # sync all channels
    
    def update(self):
        self.mathi(self.s_rp, self.s_freq2, self.s_freq2, '+', self.f_step) # update frequency list index
        

class fngnp1SpectroscopyExperiment(Experiment):
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
        
        qspec_ef=fngnp1SpectroscopyProgram(soc, self.cfg)
        
        xpts, avgi, avgq = qspec_ef.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress)
        # x_pts, avgi, avgq = qspec_ef.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress, debug=debug)        
        
        data={'xpts':xpts, 'avgi':avgi, 'avgq':avgq}
        
        self.data=data

        iq_calib = self.qubit_prob_calib(path=self.path, config_file=self.config_file)
    
        i_prob, q_prob = self.get_qubit_prob(avgi[0][0], avgq[0][0], iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])

        data_dict = {'xpts': xpts, 'avgq':avgq[0][0], 'avgi':avgi[0][0], 'i_g': [iq_calib['i_g']], 'q_g': [iq_calib['q_g']], 'i_e': [iq_calib['i_e']], 'q_e': [iq_calib['q_e']], 'avgi_prob': i_prob, 'avgq_prob': q_prob}

        if data_path and filename:
            self.save_data(data_path=data_path, filename=filename, arrays=data_dict)
            
        return data_dict

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
        