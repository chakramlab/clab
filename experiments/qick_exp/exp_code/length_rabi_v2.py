# 20230320 - commented out

# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm_notebook as tqdm

# from qick import *
# from qick.helpers import gauss
# from slab import Experiment, dsfit, AttrDict

# class QubitSpectroscopyProgram(RAveragerProgram):
#     def initialize(self):
#         cfg = self.cfg
#         self.cfg.update(self.cfg.expt)
#         # soc = self.cfg.soc
#         # soc = self.im[self.cfg.aliases.soc]
        
#         #--------------------------Resonator Param
#         self.res_ch=cfg.device.soc.resonator.ch
#        # print(self.res_ch)
#         self.readout_length = self.us2cycles(cfg.device.soc.readout.length)
#         self.res_gain = cfg.device.soc.resonator.gain
#         self.readout_ch = cfg.device.soc.readout.ch
#         self.res_freq = self.freq2reg(cfg.device.soc.resonator.freq, gen_ch=self.res_ch, ro_ch=self.readout_ch[0])
#         self.adc_trig_offset = cfg.device.soc.readout.adc_trig_offset
#         self.relax_delay = cfg.device.soc.readout.relax_delay
#         print(self.relax_delay)
    
#         # set the nyquist zone
#         self.declare_gen(ch=self.res_ch, nqz=1)

#         for ch in [0]:  # configure the readout lengths and downconversion frequencies
#             self.declare_readout(ch=ch, length=self.readout_length,
#                                  freq=self.res_freq, gen_ch=self.res_ch)

#         #freq=self.freq2reg(self.freq, gen_ch=self.res_ch, ro_ch=self.readout_ch)  # convert frequency to dac frequency (ensuring it is an available adc frequency)
        
#         #initializing pulse register
#         self.set_pulse_registers(
#             ch=self.res_ch,
#             style="const",
#             freq=self.res_freq, 
#             phase=self.deg2reg(0, gen_ch=self.res_ch), 
#             gain=self.res_gain, 
#             length=self.readout_length)

#         #------------------------------- Qubit Param
#         self.q_ch=cfg.device.soc.qubit.ch
#         self.q_length = self.us2cycles(cfg.expt.length)
#         self.q_freq_start = self.freq2reg(cfg.expt.start, gen_ch = self.q_ch)
#         self.q_freq_step = self.freq2reg(cfg.expt.step)
#         self.q_gain = cfg.expt.gain
#         self.q_reg_page =self.ch_page(self.q_ch)     # get register page for qubit_ch
#         self.q_freq_reg = self.sreg(self.q_ch, "freq")   # get frequency register for qubit_ch
#         self.q_phase_reg = self.sreg(self.q_ch, "phase")


#         # set the nyquist zone
#         nqz_test = 1
#         self.declare_gen(ch=self.q_ch, nqz=nqz_test)
#         print(nqz_test, 'test0')

#         #------ Update 

#         self.set_pulse_registers(ch=self.q_ch, style="const", freq=self.q_freq_start, phase=0, gain=self.q_gain, 
#                                  length=self.q_length)
#         self.synci(200)  # give processor some time to configure pulses
#         self.synci(200)  # give processor some time to configure pulses
    
#     def body(self):
#         cfg=AttrDict(self.cfg)
#         # self.safe_regwi(self.q_reg_page, self.q_phase_reg ,0) #self.deg2reg(0, gen_ch=self.q_ch))
#         self.pulse(ch=self.q_ch)
#         self.sync_all()
#         self.measure(pulse_ch=self.res_ch,
#                      adcs=[0],
#                      pins=[0],
#                      adc_trig_offset= self.adc_trig_offset,
#                      wait=True,
#                      syncdelay=self.us2cycles(self.relax_delay))  # sync all channels
    
#     def update(self):
#         self.mathi(self.q_reg_page, self.q_freq_reg, self.q_freq_reg, '+', self.q_freq_step) # update frequency list index

import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from slab.dataanalysis import get_next_filename
from slab import SlabFile


from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

class LengthRabiv2Program(RAveragerProgram):
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
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=self.res_freq, gen_ch=self.res_ch)


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

        

        # ------------------------------- Qubit Param
        self.q_ch=cfg.device.soc.qubit.ch
        self.q_length = self.us2cycles(cfg.expt.start)
        self.q_time_step = self.freq2reg(cfg.expt.step)
        self.q_gain = cfg.expt.gain
        self.q_reg_page =self.ch_page(self.q_ch)     # get register page for qubit_ch
        self.q_freq_reg = self.sreg(self.q_ch, "freq")   # get frequency register for qubit_ch
        self.q_phase_reg = self.sreg(self.q_ch, "phase")
        self.q_length_reg = self.sreg(self.q_ch, "t")
        self.q_length_temp = 4

        # set the nyquist zone
        self.declare_gen(ch=self.q_ch, nqz=1)

        self.set_pulse_registers(ch=self.q_ch, style="const", freq=self.freq2reg(cfg.device.soc.qubit.f_ge), phase=0, gain=self.q_gain, 
                                 length=self.q_length)
        
        self.synci(200)  # give processor some time to configure pulses
        self.synci(200)  # give processor some time to configure pulses

    
    def body(self):
        cfg=AttrDict(self.cfg)
        # soc = self.cfg.soc
        # soc = self.im[self.cfg.aliases.soc]
        self.mathi(self.q_reg_page, self.q_length_reg, self.q_length_temp, "+", 0)
        self.pulse(ch=self.q_ch)
        self.sync_all()
        self.measure(pulse_ch=self.res_ch, 
             adcs=self.readout_ch,
             pins = [0],
             adc_trig_offset=self.adc_trig_offset,
             wait=True,
             syncdelay=self.relax_delay)

    def update(self):
        self.mathi(self.q_reg_page, self.q_length_temp, self.q_length_temp, '+', self.q_time_step) # update frequency list index
        # self.mathi(self.q_reg_page, self.r_wait, self.r_wait, '+',
        #            self.us2cycles(self.cfg.expt.step))

class LengthRabiv2Experiment(Experiment):
    """Qubit Spectroscopy Experiment
       Experimental Config
        expt={"start":4020, "step":0.35, "expts":300, "reps": 200,"rounds":50,
          "length":5, "gain":400
         }
    """

    def __init__(self, path='', prefix='QubitProbeSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False, data_path=None, filename=None):
        fpts=self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        lrabi=LengthRabiv2Program(soc, self.cfg)
        x_pts, avgi, avgq = lrabi.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress, debug=debug)        
        
        data={'tpts':x_pts, 'avgi':avgi, 'avgq':avgq}
        
        self.data=data

        
        if data_path and filename:
            file_path = data_path + get_next_filename(data_path, filename, '.h5')
            with SlabFile(file_path, 'a') as f:
                f.append_line('time', x_pts)
                f.append_line('avgi', avgi[0][0])
                f.append_line('avgq', avgq[0][0])
            print("File saved at", file_path)
         
        

        return data
    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
        
        # ex: fitparams=[1.5, 1/(2*15000), -np.pi/2, 1e8, -13, 0]
        pI = dsfit.fitdecaysin(data["tpts"],data['avgi'][0][0],
                               fitparams=None, showfit=False)
        pQ = dsfit.fitdecaysin(data["tpts"],data['avgq'][0][0],
                               fitparams=None, showfit=False)
        # adding this due to extra parameter in decaysin that is not in fitdecaysin
     
        data['fiti'] = pI
        data['fitq'] = pQ
        
        return data
    

    def display(self, data=None, **kwargs):
        if data is None:
            data=self.data 
        print(self.fname)
        plt.figure(figsize=(10,8))
        plt.subplot(211,title="Length Rabi",  ylabel="I")
        plt.plot(data["xpts"][0], data["avgi"][0][0],'o-')
        plt.subplot(212, xlabel="Time (us)", ylabel="Q")
        plt.plot(data["xpts"][0], data["avgq"][0][0],'o-')
        plt.tight_layout()
        plt.show()   

