import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from slab.dataanalysis import get_next_filename
from slab import SlabFile


from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

class QubitSpectroscopyProgram_v2(RAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.cfg.update(self.cfg.expt)
        # soc = self.cfg.soc
        # soc = self.im[self.cfg.aliases.soc]
        
        ############param
        self.res_ch= cfg.device.soc.resonator.ch
        self.readout_length = self.us2cycles(cfg.device.soc.readout.length)
        self.res_freq = cfg.device.soc.readout.freq
        self.res_gain = cfg.device.soc.resonator.gain
        self.readout_ch = cfg.device.soc.readout.ch
        self.adc_trig_offset = cfg.device.soc.readout.adc_trig_offset
        self.relax_delay = self.us2cycles(cfg.device.soc.readout.relax_delay)
        #################
        #print(self.res_freq)

        # # set the nyquist zone
        # self.declare_gen(ch=self.res_ch, nqz=1)

        # # configure the readout lengths and downconversion frequencies
        # for ch in self.readout_ch:  # configure the readout lengths and downconversion frequencies
        #     self.declare_readout(ch=ch, length=self.readout_length,
        #                          freq=self.res_freq, gen_ch=self.res_ch)
            

        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist) 


        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.soc.readout.freq, gen_ch=self.res_ch)

        #initializing readout pulse register
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.freq2reg(self.res_freq, gen_ch=self.res_ch, ro_ch=self.readout_ch[0]), # convert frequency to dac frequency (ensuring it is an available adc frequency)
            phase=self.deg2reg(0, gen_ch=self.res_ch), # 0 degrees
            gain=self.res_gain, 
            length=self.readout_length)

        

        # ------------------------------- Qubit Param
        self.q_ch=cfg.device.soc.qubit.ch
        self.q_length = self.us2cycles(cfg.expt.length)

        self.q_freq_start = self.freq2reg(cfg.expt.start, gen_ch = self.q_ch)
        self.q_freq_step = self.freq2reg(cfg.expt.step)
        self.q_reg_page =self.ch_page(self.q_ch)     # get register page for qubit_ch
        self.q_freq_reg = self.sreg(self.q_ch, "freq")   # get frequency register for qubit_ch
        self.q_phase_reg = self.sreg(self.q_ch, "phase")

        if cfg.expt.take_default_pi_pulse_params: 
            print ("Using default pi sigma value from cfg")
            self.length = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma)
            self.q_gain = cfg.device.soc.qubit.pulses.pi_ge.gain
            pulse_type = cfg.device.soc.qubit.pulses.pi_ge.pulse_type
        else:
            self.length = self.us2cycles(cfg.expt.length)
            self.q_gain = cfg.expt.gain
            pulse_type = cfg.expt.pulse_type


        # set the nyquist zone
        self.declare_gen(ch=self.q_ch, nqz=cfg.device.soc.qubit.nyqist)

        print ("Pulse type = ", pulse_type)

        if pulse_type == 'const':

            self.set_pulse_registers(
                ch=self.q_ch, 
                style="const", 
                freq=self.q_freq_start, 
                phase=0,
                gain=self.q_gain, 
                length=self.length)
            
        elif pulse_type == 'gauss':

            self.add_gauss(ch=self.q_ch, name="qubit", sigma=self.length, length=self.length * 4)
            self.set_pulse_registers(
                ch=self.q_ch,
                style="arb",
                freq=self.q_freq_start,
                phase=self.deg2reg(0),
                gain=self.q_gain,
                waveform="qubit")
            
        self.synci(200)  # give processor some time to configure pulses

    
    def body(self):
        cfg=AttrDict(self.cfg)

        self.pulse(ch=self.q_ch)
        self.sync_all()
        self.measure(pulse_ch=self.res_ch,
                     adcs=[1, 0],
                     adc_trig_offset=cfg.device.soc.readout.adc_trig_offset,
                     wait=True,
                     syncdelay=self.us2cycles(cfg.device.soc.readout.relax_delay))  # sync all channels

    def update(self):
        self.mathi(self.q_reg_page, self.q_freq_reg, self.q_freq_reg, '+', self.q_freq_step) # update frequency list index

class QubitSpectroscopyCalibratedPulseExperiment(Experiment):
    """Qubit Spectroscopy Experiment
       Experimental Config
        expt={"start":4020, "step":0.35, "expts":300, "reps": 200,"rounds":50,
          "length":5, "gain":400
         }
    """

    def __init__(self, path='', prefix='QubitProbeSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False, data_path=None, filename=None, prob_calib=True):
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        qspec=QubitSpectroscopyProgram_v2(soc, self.cfg)
        xpts, avgi, avgq = qspec.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress)        
        
        data={'xpts': xpts, 'avgi':avgi, 'avgq':avgq}
        self.data=data

        if prob_calib:
            # Calibrate qubit probability
            iq_calib = self.qubit_prob_calib(path=self.path, config_file=self.config_file)
            i_prob, q_prob = self.get_qubit_prob(avgi[0][0], avgq[0][0], iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])
            data_dict = {'xpts': xpts, 'avgq':avgq[0][0], 'avgi':avgi[0][0], 'i_g': [iq_calib['i_g']], 'q_g': [iq_calib['q_g']], 'i_e': [iq_calib['i_e']], 'q_e': [iq_calib['q_e']], 'avgi_prob': i_prob, 'avgq_prob': q_prob}
        else:
            data_dict = {'xpts':data['xpts'], 'avgi':data['avgi'][0][0], 'avgq':data['avgq'][0][0]}

        if data_path and filename:
            self.save_data(data_path, filename, arrays=data_dict)

        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
        
        data['fiti']=dsfit.fitlor(data["fpts"],data['avgi'][0][0])
        data['fitq']=dsfit.fitlor(data["fpts"],data['avgq'][0][0])
        print(data['fiti'], data['fitq'])
        
        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data=self.data
        print(self.fname)
        plt.figure(figsize=(10,8))
        plt.subplot(211,title="Qubit Spectroscopy",  ylabel="I")
        plt.plot(data["fpts"], data["avgi"][0][0],'o-')
        if "fiti" in data:
            plt.plot(data["fpts"], dsfit.lorfunc(data["fiti"], data["fpts"]))
        plt.subplot(212, xlabel="Pulse Frequency (MHz)", ylabel="Q")
        plt.plot(data["fpts"], data["avgq"][0][0],'o-')
        if "fitq" in data:
            plt.plot(data["fpts"], dsfit.lorfunc(data["fitq"], data["fpts"]))
            
        plt.tight_layout()
        plt.show()
        