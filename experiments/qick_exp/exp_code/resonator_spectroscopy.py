import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from slab.dataanalysis import get_next_filename
from slab import SlabFile

from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict
# class LoopbackProgram(AveragerProgram):
#     def initialize(self):
#         cfg=self.cfg   
#         res_ch = cfg["res_ch"]

#         # set the nyquist zone
#         self.declare_gen(ch=cfg["res_ch"], nqz=1)
        
#         # configure the readout lengths and downconversion frequencies (ensuring it is an available DAC frequency)
#         for ch in cfg["ro_chs"]:
#             self.declare_readout(ch=ch, length=self.cfg["readout_length"],
#                                  freq=self.cfg["pulse_freq"], gen_ch=cfg["res_ch"])

#         # convert frequency to DAC frequency (ensuring it is an available ADC frequency)
#         freq = self.freq2reg(cfg["pulse_freq"],gen_ch=res_ch, ro_ch=cfg["ro_chs"][0])
#         phase = self.deg2reg(cfg["res_phase"], gen_ch=res_ch)
#         gain = cfg["pulse_gain"]
#         self.default_pulse_registers(ch=res_ch, freq=freq, phase=phase, gain=gain)

#         style=self.cfg["pulse_style"]

#         if style in ["flat_top","arb"]:
#             sigma = cfg["sigma"]
#             self.add_gauss(ch=res_ch, name="measure", sigma=sigma, length=sigma*5)
            
#         if style == "const":
#             self.set_pulse_registers(ch=res_ch, style=style, length=cfg["length"])
#         elif style == "flat_top":
#             # The first half of the waveform ramps up the pulse, the second half ramps down the pulse
#             self.set_pulse_registers(ch=res_ch, style=style, waveform="measure", length=cfg["length"])
#         elif style == "arb":
#             self.set_pulse_registers(ch=res_ch, style=style, waveform="measure")
        
#         self.synci(200)  # give processor some time to configure pulses
    
#     def body(self):
#         # fire the pulse
#         # trigger all declared ADCs
#         # pulse PMOD0_0 for a scope trigger
#         # pause the tProc until readout is done
#         # increment the time counter to give some time before the next measurement
#         # (the syncdelay also lets the tProc get back ahead of the clock)
#         self.measure(pulse_ch=self.cfg["res_ch"], 
#                      adcs=self.ro_chs,
#                      pins=[0], 
#                      adc_trig_offset=self.cfg["adc_trig_offset"],
#                      wait=True,
#                      syncdelay=self.us2cycles(self.cfg["relax_delay"]))
        
#         # equivalent to the following:
# #         self.trigger(adcs=self.ro_chs,
# #                      pins=[0], 
# #                      adc_trig_offset=self.cfg["adc_trig_offset"])
# #         self.pulse(ch=self.cfg["res_ch"])
# #         self.wait_all()
# #         self.sync_all(self.us2cycles(self.cfg["relax_delay"]))

class ResonatorSpectroscopyProgram(AveragerProgram):
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

        self.synci(200)  # give processor some time to configure pulses
    
    def body(self):
        cfg=AttrDict(self.cfg)
        # soc = self.cfg.soc
        # soc = self.im[self.cfg.aliases.soc]
        self.measure(pulse_ch=self.res_ch, 
             adcs=self.readout_ch,
             pins = [0],
             adc_trig_offset=self.adc_trig_offset,
             wait=True,
             syncdelay=self.relax_delay)

class ResonatorSpectroscopyExperiment(Experiment):
    """Resonator Spectroscopy Experiment
       Experimental Config
       expt_cfg={
       "start": start frequency (MHz), 
       "step": frequency step (MHz), 
       "expts": number of experiments, 
       "reps": number of reps, 
        } 
    """

    def __init__(self, path='', prefix='ResonatorSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)

        #for debuggging 
        # config={"res_ch":6, # --Fixed
        # "ro_chs":[0,1], # --Fixed
        # "reps":1, # --Fixed
        # "relax_delay":1.0, # --us
        # "res_phase":0, # --degrees
        # "pulse_style": "const", # --Fixed
        
        # "length":20, # [Clock ticks]
        # # Try varying length from 10-100 clock ticks
        
        # "readout_length":100, # [Clock ticks]
        # # Try varying readout_length from 50-1000 clock ticks

        # "pulse_gain":3000, # [DAC units]
        # # Try varying pulse_gain from 500 to 30000 DAC units

        # "pulse_freq": 3, # [MHz]
        # # In this program the signal is up and downconverted digitally so you won't see any frequency
        # # components in the I/Q traces below. But since the signal gain depends on frequency, 
        # # if you lower pulse_freq you will see an increased gain.

        # "adc_trig_offset": 100, # [Clock ticks]
        # # Try varying adc_trig_offset from 100 to 220 clock ticks

        # "soft_avgs":100
        # # Try varying soft_avgs from 1 to 200 averages

        #  }
        #print(self.cfg)
        #self.cfg.update(config)
        #print(self.cfg)
        #soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        #self.prog= LoopbackProgram(soc, self.cfg)
        #self.prog= ResonatorSpectroscopyProgram(soc, self.cfg)
        print('Successfully Initialized')

    def acquire(self, progress=False, data_path=None, filename=None):
        fpts=self.cfg.expt["start"] + self.cfg.expt["step"]*np.arange(self.cfg.expt["expts"])
        data={"fpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())

        avgi_col = []
        avgq_col = []
        
        for f in tqdm(fpts, disable = not progress):
            #print('inside res spec for loop')
            self.cfg.device.soc.resonator.freq = int(f)
            #print(self.cfg.device.resonator.freq)
            #prog = ResonatorSpectroscopyProgram(soc, self.cfg)
            self.prog= ResonatorSpectroscopyProgram(soc, self.cfg)
            avgi,avgq=self.prog.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            amp=np.abs(avgi[0][0]+1j*avgq[0][0]) # Calculating the magnitude    # what's the role of [0][0]? First index: Which expt(diff freq tested for diff expt); averaged over that many reps, second index: Which adc
            phase=np.angle(avgi[0][0]+1j*avgq[0][0]) # Calculating the phase    # Here we are calculating the mag and phase of obtained data point

            data["fpts"].append(f)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)

            avgi_col.append(avgi[0][0])
            avgq_col.append(avgq[0][0])
        
        for k, a in data.items():
            data[k]=np.array(a)
        
        self.data=data

        avgi_col = np.array(avgi_col)
        avgq_col = np.array(avgq_col)

        # if data_path and filename:
        #     file_path = data_path + get_next_filename(data_path, filename, '.h5')
        #     with SlabFile(file_path, 'a') as f:
        #         f.append_line('freq', fpts)
        #         f.append_line('avgi', avgi_col)
        #         f.append_line('avgq', avgq_col)
        #         f.append_pt('amp', amp)
        #         f.append_pt('phase', phase)
         
        #     print("File saved at", file_path)

        data_dict = {"xpts":fpts, "avgq":avgq_col, "avgi":avgi_col}
        if data_path and filename:
            self.save_data(data_path=data_path, filename=filename, arrays=data_dict)
        # return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
        
        fit=dsfit.fitlor(data["fpts"], data["amps"]**2)
        data["fit"]=fit
        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data=self.data 
        print (self.fname)
        plt.subplot(111,title="Resonator Spectroscopy", xlabel="Resonator Frequency (MHz)", ylabel="Amp. (adc level)")
        plt.plot(data["fpts"], data["amps"],'o')
        if "fit" in data:
            plt.plot(data["fpts"], np.sqrt(dsfit.lorfunc(data["fit"], data["fpts"])))
        plt.show()
        