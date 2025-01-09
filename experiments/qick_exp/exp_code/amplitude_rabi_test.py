
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from slab.dataanalysis import get_next_filename
from slab import SlabFile


from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

class AmplitudeRabiProgram(RAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.cfg.update(self.cfg.expt)
        # soc = self.cfg.soc
        # soc = self.im[self.cfg.aliases.soc]
        
        ############param
        self.res_ch= cfg.device.soc.resonator.ch
        self.readout_length = self.us2cycles(cfg.device.soc.readout.length)
        self.res_freq = cfg.device.soc.resonator.freq
        self.res_gain = cfg.device.soc.resonator.gain
        self.readout_ch = cfg.device.soc.readout.ch
        self.adc_trig_offset = cfg.device.soc.readout.adc_trig_offset
        self.relax_delay = self.us2cycles(cfg.device.soc.readout.relax_delay)
        #################
        #print(self.res_freq)

        # set the nyquist zone
        self.declare_gen(ch=self.res_ch, nqz=1)

        # configure the readout lengths and downconversion frequencies
        for ch in self.readout_ch:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, 
                                 length=self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0]),
                                 freq=cfg.device.soc.readout.freq, 
                                 gen_ch=self.cfg.device.soc.resonator.ch)


        # converrt frequency to DAC frequency
        self.freq=self.freq2reg(self.res_freq, gen_ch=self.res_ch, ro_ch=self.readout_ch[0])  # convert frequency to dac frequency (ensuring it is an available adc frequency)
        
        #initializing pulse register
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.freq, 
            phase=self.deg2reg(0, gen_ch=self.res_ch), # 0 degrees
            gain=self.res_gain, 
            length=self.readout_length)

        self.synci(200)  # give processor some time to configure pulses
    
    def body(self):
        cfg=AttrDict(self.cfg)
        # soc = self.cfg.soc
        # soc = self.im[self.cfg.aliases.soc]
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
    #     cfg = self.cfg
    #     self.cfg.update(self.cfg.expt)
        
    #     ############param
    #     self.res_ch= cfg.device.soc.resonator.ch
    #     self.readout_length = self.us2cycles(cfg.device.soc.readout.length)
    #     self.res_freq = cfg.device.soc.resonator.freq
    #     self.res_gain = cfg.device.soc.resonator.gain
    #     self.readout_ch = cfg.device.soc.readout.ch
    #     self.adc_trig_offset = cfg.device.soc.readout.adc_trig_offset
    #     self.relax_delay = self.us2cycles(cfg.device.soc.readout.relax_delay)
    #     #################

    #     # set the nyquist zone
    #     self.declare_gen(ch=self.res_ch, nqz=1)

    #     # configure the readout lengths and downconversion frequencies
    #     for ch in self.readout_ch:  # configure the readout lengths and downconversion frequencies
    #         self.declare_readout(ch=ch, length=self.readout_length,
    #                              freq=self.res_freq, gen_ch=self.res_ch)


    #     # converrt frequency to DAC frequency
    #     self.freq=self.freq2reg(self.res_freq, gen_ch=self.res_ch, ro_ch=self.readout_ch[0])  # convert frequency to dac frequency (ensuring it is an available adc frequency)
        
    #     #initializing pulse register
    #     self.set_pulse_registers(
    #         ch=self.res_ch,
    #         style="const",
    #         freq=self.freq, 
    #         phase=self.deg2reg(0, gen_ch=self.res_ch), # 0 degrees
    #         gain=self.res_gain, 
    #         length=self.readout_length)

        

    #     # # ------------------------------- Qubit Param
    #     # self.q_ch=cfg.device.soc.qubit.ch
    #     # self.q_freq_start = self.freq2reg(cfg.expt.start, gen_ch = self.q_ch)
    #     # self.q_freq_step = self.freq2reg(cfg.expt.step)
    #     # self.q_reg_page =self.ch_page(self.q_ch)     # get register page for qubit_ch
    #     # self.sigma_test = self.us2cycles(self.cfg.expt.sigma_test)
    #     # self.r_gain = self.sreg(self.q_ch, "gain")  # get gain register for qubit_ch 

    #     # # set the nyquist zone
    
    #     # self.declare_gen(ch=self.q_ch, nqz=1)

    #     # if self.cfg.expt.pulse_type == "gauss" and self.cfg.expt.sigma_test > 0:
    #     #     self.add_gauss(ch=self.q_ch, name="qubit", sigma=self.sigma_test, length=self.sigma_test * 4)
    #     #     self.set_pulse_registers(
    #     #         ch=self.q_ch,
    #     #         style="arb",
    #     #         freq=self.freq2reg(cfg.device.soc.qubit.f_ge),
    #     #         phase=0,
    #     #         gain=cfg.expt.start,
    #     #         waveform="qubit")
       
    #     # elif self.cfg.expt.sigma_test > 0:
    #     #     self.set_pulse_registers(
    #     #         ch=self.q_ch,
    #     #         style="const",
    #     #         freq=self.freq2reg(cfg.device.soc.qubit.f_ge),
    #     #         phase=0,
    #     #         gain=cfg.expt.start,
    #     #         length=self.sigma_test)
        
    #     self.synci(200)  # give processor some time to configure pulses
    #     self.synci(200)  # give processor some time to configure pulses

    
    # def body(self):
    #     cfg=AttrDict(self.cfg)

    #     # self.pulse(ch=self.q_ch)
    #     self.sync_all()
    #     self.measure(pulse_ch=self.res_ch, 
    #          adcs=self.readout_ch,
    #          pins = [0],
    #          adc_trig_offset=self.adc_trig_offset,
    #          wait=True,
    #          syncdelay=self.relax_delay)

    # def update(self):
    #     self.mathi(self.q_reg_page, self.r_gain, self.r_gain, '+', self.cfg.expt.step) # update frequency list index

class AmplitudeRabiExperiment(Experiment):
    """Qubit Spectroscopy Experiment
       Experimental Config
        expt={"start":4020, "step":0.35, "expts":300, "reps": 200,"rounds":50,
          "length":5, "gain":400
         }
    """

    def __init__(self, path='', prefix='AmplitudeRabi', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False, data_path=None, filename=None):
        fpts=self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        amprabi=AmplitudeRabiProgram(soc, self.cfg)
        x_pts, avgi, avgq = amprabi.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress, debug=debug)        
        
        data={'fpts':x_pts, 'avgi':avgi, 'avgq':avgq}
        
        self.data=data
        
        data_dict = {'xpts':x_pts, 'avgi':avgi[0][0], 'avgq':avgq[0][0]}

        # if data_path and filename:
        #     file_path = data_path + get_next_filename(data_path, filename, '.h5')
        #     with SlabFile(file_path, 'a') as f:
        #         f.append_line('freq', x_pts)
        #         f.append_line('avgi', avgi[0][0])
        #         f.append_line('avgq', avgq[0][0])
        #     print("File saved at", file_path)
        if data_path and filename:
            self.save_data(data_path=data_path, filename=filename, arrays=data_dict)
        

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
        