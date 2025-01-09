import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from slab.dataanalysis import get_next_filename
from slab import SlabFile

from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

class ResonatorSpectroscopyProgram(AveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.cfg.update(self.cfg.expt)
        # soc = self.cfg.soc
        # soc = self.im[self.cfg.aliases.soc]
        
        ############param
        self.res_ch= cfg.device.soc.resonator.ch
        self.res_freq = self.cfg.expt.length_placeholder
        self.res_gain = cfg.device.soc.resonator.gain
        self.readout_ch = cfg.device.soc.readout.ch
        self.adc_trig_offset = cfg.device.soc.readout.adc_trig_offset
        self.relax_delay = self.us2cycles(cfg.device.soc.readout.relax_delay)
        self.readout_length = self.us2cycles(cfg.device.soc.readout.length, ro_ch=self.readout_ch[0])
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

        ### Qubit pi-pulse before readout

        self.qubit_pulse_type = self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type
        self.qubit_ch = cfg.device.soc.qubit.ch
        self.q_rp=self.ch_page(self.qubit_ch)     # get register page for qubit_ch
        self.declare_gen(ch=self.qubit_ch, nqz=self.cfg.device.soc.qubit.nyqist)
        self.pi_ge_sigma = self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)

        if self.qubit_pulse_type == "gauss":
            print('Pulse type: gauss')
            self.add_gauss(ch=self.qubit_ch, name="qubit_pi", sigma=self.pi_ge_sigma, length=self.pi_ge_sigma * 4)
            self.set_pulse_registers(
                            ch=self.qubit_ch,
                            style="arb",
                            freq=self.freq2reg(cfg.device.soc.qubit.f_ge),
                            phase=self.deg2reg(0),
                            gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain,
                            waveform="qubit_pi")
            
        elif self.qubit_pulse_type == "const":
            print('Pulse type: const')
            self.set_pulse_registers(
                            ch=self.qubit_ch,
                            style="const",
                            freq=self.freq2reg(cfg.device.soc.qubit.f_ge),
                            phase=self.deg2reg(0),
                            gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain,
                            length=self.pi_ge_sigma)

        self.sync_all(self.us2cycles(0.2))  # give processor some time to configure pulses
    
    def body(self):
        cfg=AttrDict(self.cfg)
        # soc = self.cfg.soc
        # soc = self.im[self.cfg.aliases.soc]

        if cfg.expt.ge_pi_before:
            print('Running pi-pulse before readout')
    
            self.pulse(ch=self.qubit_ch)
            self.sync_all()

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

    def __init__(self, path='', prefix='ResonatorSpectroscopy', config_file=None, progress=None, datapath = None,filename = None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)
        
        if datapath is None:
            self.datapath = self.path + '\\data'
        else:
            self.datapath = datapath
        
        if filename is None:
            self.filename = self.prefix
        else: 
            self.filename = filename
        

        print('Successfully Initialized')

    def acquire(self, data_path=None, filename=None, progress=False):
        fpts=self.cfg.expt["start"] + self.cfg.expt["step"]*np.arange(self.cfg.expt["expts"])
        data={"fpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())

        avgi_col = []
        avgq_col = []
        
        for f in tqdm(fpts, disable = not progress):
            #print('inside res spec for loop')
            self.cfg.expt.length_placeholder = f
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

        return data_dict

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
        