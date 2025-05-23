import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from slab.dataanalysis import get_next_filename
from slab import SlabFile


from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

class WignerTomographyProgram(AveragerProgram):
    def initialize(self):

        # --- Initialize parameters ---

        cfg = self.cfg
        self.cfg.update(self.cfg.expt)
        
        # Readout parameters
        self.res_ch= cfg.device.soc.resonator.ch
        self.res_ch_nyquist = cfg.device.soc.resonator.nyqist
        self.readout_length = self.us2cycles(cfg.device.soc.readout.length)
        self.res_freq = cfg.device.soc.readout.freq
        self.readout_ch = cfg.device.soc.readout.ch
        self.readout_freq=self.freq2reg(self.res_freq, gen_ch=self.res_ch, ro_ch=self.readout_ch[0])  # convert frequency to dac frequency (ensuring it is an available adc frequency)
        self.res_gain = cfg.device.soc.resonator.gain
        self.adc_trig_offset = cfg.device.soc.readout.adc_trig_offset
        self.relax_delay = self.us2cycles(cfg.device.soc.readout.relax_delay)

        # Qubit parameters
        self.q_ch=cfg.device.soc.qubit.ch
        self.q_reg_page =self.ch_page(self.q_ch)     # get register page for qubit_ch
        self.q_freq_reg = self.sreg(self.q_ch, "freq")   # get frequency register for qubit_ch
        self.q_phase_reg = self.sreg(self.q_ch, "phase")
        self.q_ch_nyquist = cfg.device.soc.qubit.nyqist

        self.qubit_freq = self.freq2reg(cfg.device.soc.qubit.f_ge, gen_ch = self.q_ch)
        self.qubit_pi2_gain = cfg.device.soc.qubit.pulses.pi2_ge.gain
        self.qubit_pi2_sigma = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ge.sigma, gen_ch=self.q_ch)
        self.qubit_pi2_pulsetype = cfg['device']['soc']['qubit']['pulses']['pi2_ge']['pulse_type']

        self.qubit_pi2_waittime = self.us2cycles(cfg.expt.waittime)
        
        # Cavity drive parameters
        self.cavdr_ch=cfg.device.soc.storage.ch
        self.cavdr_reg_page =self.ch_page(self.cavdr_ch)     # get register page for cavdr_ch
        self.cavdr_freq_reg = self.sreg(self.cavdr_ch, "freq")   # get frequency register for cavdr_ch
        self.cavdr_ch_nyquist = cfg.device.soc.storage.nyquist

        self.cavdr_length = self.us2cycles(cfg.expt.length)
        self.cavdr_freq = self.freq2reg(cfg.expt.cavity_drive_freq, gen_ch = self.cavdr_ch)

        # Wigner tomography sweep
        self.cavdr_gain = cfg.expt.cavdr_gain_temp
        self.cavdr_phase = cfg.expt.cavdr_phase_temp

        # --- Initialize pulses ---

        # set the nyquist zone
        self.declare_gen(ch=self.res_ch, nqz=1)
        self.declare_gen(ch=self.q_ch, nqz=2)
        self.declare_gen(ch=self.cavdr_ch, nqz=2)

        # configure the readout lengths and downconversion frequencies
        for ch in self.readout_ch:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, 
                                 length=self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0]),
                                 freq=cfg.device.soc.readout.freq, 
                                 gen_ch=self.cfg.device.soc.resonator.ch)
        
        #initializing pulse register
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.readout_freq, 
            phase=self.deg2reg(0, gen_ch=self.res_ch), # 0 degrees
            gain=self.res_gain, 
            length=self.readout_length)
        
        self.set_pulse_registers(
            ch=self.cavdr_ch,
            style="const",
            freq=self.cavdr_freq,
            phase=self.deg2reg(self.cavdr_phase, gen_ch=self.cavdr_ch), # 0 degrees
            gain=self.cavdr_gain,
            length=self.cavdr_length)
        

        if self.qubit_pi2_pulsetype == 'gauss':
            self.add_gauss(ch=self.q_ch, name="qubit_ge", sigma=self.qubit_pi2_sigma, length=self.qubit_pi2_sigma * 4)
    
            self.set_pulse_registers(
                ch=self.q_ch,
                style="arb",
                freq=self.qubit_freq,
                phase=self.deg2reg(0),
                gain=self.qubit_pi2_gain,
                waveform="qubit_ge")
        
        if self.qubit_pi2_pulsetype == 'const':
            print('Qubit pulse type set to const')
            self.set_pulse_registers(
                    ch=self.q_ch, 
                    style="const", 
                    freq=self.qubit_freq, 
                    phase=0,
                    gain=self.qubit_pi2_gain, 
                    length=self.qubit_pi2_sigma)
        
        self.synci(500)  # give processor some time to configure pulses

    def body(self):

        self.pulse(ch=self.cavdr_ch)  # Cavity displacement
        self.sync_all()
        
        self.pulse(ch=self.q_ch)  # Qubit pi/2 pulse
        self.sync_all(self.qubit_pi2_waittime)  # Wait for time tau = chi/2

        self.pulse(ch=self.q_ch)  # Qubit pi/2 pulse
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

class WignerTomographyExperiment(Experiment):
    """Qubit Spectroscopy Experiment
       Experimental Config
        expt={"start":4020, "step":0.35, "expts":300, "reps": 200,"rounds":50,
          "length":5, "gain":400
         }
    """

    def __init__(self, path='', prefix='WignerTomography', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False, data_path=None, filename=None):
        cavdr_gains = list(self.cfg.expt.cavdr_gains)
        cavdr_phases = list(self.cfg.expt.cavdr_phases)

        avgi_col = []
        avgq_col = []

        for i,j in tqdm(zip(cavdr_gains, cavdr_phases), total=len(cavdr_gains), disable = not progress):
            self.cfg.expt.cavdr_gain_temp = i
            self.cfg.expt.cavdr_phase_temp = j
            print('Gain = ', i, 'Phase = ', j)
            soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
            wigtom=WignerTomographyProgram(soc, self.cfg)
            avgi, avgq = wigtom.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=False)

            avgi_col.append(avgi[0][0])
            avgq_col.append(avgq[0][0])        
        
        avgi_col = np.array(avgi_col)
        avgq_col = np.array(avgq_col)

        # Calibrate qubit probability

        iq_calib = self.qubit_prob_calib(path=self.path, config_file=self.config_file)
    
        i_prob, q_prob = self.get_qubit_prob(avgi_col, avgq_col, iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])

        data_dict = {'cavdr_gains': cavdr_gains, 'cavdr_phases': cavdr_phases, 'avgq':avgq_col, 'avgi':avgi_col, 'i_g': [iq_calib['i_g']], 'q_g': [iq_calib['q_g']], 'i_e': [iq_calib['i_e']], 'q_e': [iq_calib['q_e']], 'avgi_prob': i_prob, 'avgq_prob': q_prob}

        # if data_path and filename:
        #     file_path = data_path + get_next_filename(data_path, filename, '.h5')
        #     with SlabFile(file_path, 'a') as f:
        #         f.append_line('freq', x_pts)
        #         f.append_line('avgi', avgi[0][0])
        #         f.append_line('avgq', avgq[0][0])
        #     print("File saved at", file_path)
        if data_path and filename:
            self.save_data(data_path=data_path, filename=filename, arrays=data_dict)
        

        return data_dict

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
        