import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from slab.dataanalysis import get_next_filename
from slab import SlabFile


from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

class WignerTomography0pNProgram(AveragerProgram):
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
        self.qubit_freq_ef = self.freq2reg(cfg.device.soc.qubit.f_ef, gen_ch = self.q_ch)

        self.qubit_pi_gain = cfg.device.soc.qubit.pulses.pi_ge.gain
        self.qubit_pi_sigma = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma, gen_ch=self.q_ch)
        self.qubit_pi_pulsetype = cfg['device']['soc']['qubit']['pulses']['pi_ge']['pulse_type']

        self.qubit_pi2_gain = cfg.device.soc.qubit.pulses.pi2_ge.gain
        self.qubit_pi2_sigma = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ge.sigma, gen_ch=self.q_ch)
        self.qubit_pi2_pulsetype = cfg['device']['soc']['qubit']['pulses']['pi2_ge']['pulse_type']

        self.qubit_pi_ef_gain = cfg.device.soc.qubit.pulses.pi_ef.gain
        self.qubit_pi_ef_sigma = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.q_ch)
        self.qubit_pi_ef_pulsetype = cfg['device']['soc']['qubit']['pulses']['pi_ef']['pulse_type']

        self.qubit_pi2_waittime = self.us2cycles(cfg.expt.waittime)
        
        # Cavity drive parameters
        self.cavdr_ch=cfg.device.soc.storage.ch
        self.cavdr_reg_page =self.ch_page(self.cavdr_ch)     # get register page for cavdr_ch
        self.cavdr_freq_reg = self.sreg(self.cavdr_ch, "freq")   # get frequency register for cavdr_ch
        self.cavdr_ch_nyquist = cfg.device.soc.storage.nyquist

        self.cavdr_length = self.us2cycles(cfg.expt.length)
        self.cavdr_freq = self.freq2reg(cfg.expt.cavity_drive_freq, gen_ch = self.cavdr_ch)

        # Sideband drive parameters

        self.sideband_ch = cfg.device.soc.sideband.ch
        self.sideband_nyquist =cfg.device.soc.sideband.nyqist

        # Wigner tomography sweep
        self.cavdr_gain = cfg.expt.cavdr_gain_temp
        self.cavdr_phase = cfg.expt.cavdr_phase_temp

        # --- Initialize pulses ---

        # set the nyquist zone
        self.declare_gen(ch=self.res_ch, nqz=self.res_ch_nyquist)
        self.declare_gen(ch=self.q_ch, nqz=self.q_ch_nyquist)
        self.declare_gen(ch=self.cavdr_ch, nqz=self.cavdr_ch_nyquist)
        self.declare_gen(ch=self.sideband_ch, nqz=self.sideband_nyquist)

        # configure the readout lengths and downconversion frequencies
        for ch in self.readout_ch:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=self.res_freq, gen_ch=self.res_ch)
        
        #initializing pulse register
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.readout_freq, 
            phase=self.deg2reg(0, gen_ch=self.res_ch), # 0 degrees
            gain=self.res_gain, 
            length=self.readout_length)
        
        print('new settings! 5')
        self.synci(500)  # give processor some time to configure pulses

    def body(self):

        # # Phase reset all channels

        for ch in self.gen_chs.keys():
            if ch != 4:
                print(ch)
                self.setup_and_pulse(ch=ch, style='const', freq=self.freq2reg(100), phase=0, gain=100, length=self.us2cycles(.05), phrst=1)

        self.sync_all(10)

        # --- Initialize parameters ---

        cfg = self.cfg
        self.cfg.update(self.cfg.expt)
        
        if cfg.expt.n == 1:

                if self.qubit_pi2_pulsetype == 'const':
                    self.set_pulse_registers(
                            ch=self.q_ch, 
                            style="const", 
                            freq=self.qubit_freq, 
                            phase=0,
                            gain=self.qubit_pi2_gain, 
                            length=self.qubit_pi2_sigma)
                
                self.pulse(ch=self.q_ch)
                self.sync_all()

                if self.qubit_pi_ef_pulsetype == 'const':  
                    self.set_pulse_registers(
                            ch=self.q_ch, 
                            style="const", 
                            freq=self.qubit_freq_ef, 
                            phase=0,
                            gain=self.qubit_pi_ef_gain, 
                            length=self.qubit_pi_ef_sigma)
            
                self.pulse(ch=self.q_ch)
                self.sync_all()

                self.set_pulse_registers(
                    ch=self.sideband_ch,
                    style="const",
                    freq=self.freq2reg(cfg.device.soc.sideband.fngnp1_freqs[cfg.expt.mode][0]),  # freq set by update
                    phase=0,
                    gain=cfg.device.soc.sideband.pulses.fngnp1pi_gains[cfg.expt.mode][0],
                    length=self.us2cycles(cfg.device.soc.sideband.pulses.fngnp1pi_times[cfg.expt.mode][0]))

                self.pulse(ch=self.sideband_ch)
                self.sync_all()

        # Prepare |0> + |n> in cavity

        else:

            for i in np.arange(cfg.expt.n):

                if i != cfg.expt.n-1:

                    if i==0:

                        if self.qubit_pi2_pulsetype == 'const':
                            self.set_pulse_registers(
                                    ch=self.q_ch, 
                                    style="const", 
                                    freq=self.qubit_freq, 
                                    phase=0,
                                    gain=self.qubit_pi2_gain, 
                                    length=self.qubit_pi2_sigma)
                        
                        self.pulse(ch=self.q_ch)
                        self.sync_all()
                    
                    else:

                        if self.qubit_pi_pulsetype == 'const':
                            self.set_pulse_registers(
                                    ch=self.q_ch, 
                                    style="const", 
                                    freq=self.qubit_freq, 
                                    phase=0,
                                    gain=self.qubit_pi_gain, 
                                    length=self.qubit_pi_sigma)
                        
                        self.pulse(ch=self.q_ch)
                        self.sync_all()

                    if self.qubit_pi_ef_pulsetype == 'const':  
                        self.set_pulse_registers(
                                ch=self.q_ch, 
                                style="const", 
                                freq=self.qubit_freq_ef, 
                                phase=0,
                                gain=self.qubit_pi_ef_gain, 
                                length=self.qubit_pi_ef_sigma)
                
                    self.pulse(ch=self.q_ch)
                    self.sync_all()

                    if self.qubit_pi_pulsetype == 'const':
                        self.set_pulse_registers(
                                ch=self.q_ch, 
                                style="const", 
                                freq=self.qubit_freq, 
                                phase=0,
                                gain=self.qubit_pi_gain, 
                                length=self.qubit_pi_sigma)
                
                    self.pulse(ch=self.q_ch)
                    self.sync_all()

                    self.set_pulse_registers(
                        ch=self.sideband_ch,
                        style="const",
                        freq=self.freq2reg(cfg.device.soc.sideband.fngnp1_freqs[cfg.expt.mode][i]),  # freq set by update
                        phase=0,
                        gain=cfg.device.soc.sideband.pulses.fngnp1pi_gains[cfg.expt.mode][i],
                        length=self.us2cycles(cfg.device.soc.sideband.pulses.fngnp1pi_times[cfg.expt.mode][i]))

                    self.pulse(ch=self.sideband_ch)
                    self.sync_all()
                
                else:

                    if i==0:

                        if self.qubit_pi2_pulsetype == 'const':
                            self.set_pulse_registers(
                                    ch=self.q_ch, 
                                    style="const", 
                                    freq=self.qubit_freq, 
                                    phase=0,
                                    gain=self.qubit_pi2_gain, 
                                    length=self.qubit_pi2_sigma)
                        
                        self.pulse(ch=self.q_ch)
                        self.sync_all()
                    
                    else:

                        if self.qubit_pi_pulsetype == 'const':
                            self.set_pulse_registers(
                                    ch=self.q_ch, 
                                    style="const", 
                                    freq=self.qubit_freq, 
                                    phase=0,
                                    gain=self.qubit_pi_gain, 
                                    length=self.qubit_pi_sigma)
                        
                        self.pulse(ch=self.q_ch)
                        self.sync_all()

                    if self.qubit_pi_ef_pulsetype == 'const':  
                        self.set_pulse_registers(
                                ch=self.q_ch, 
                                style="const", 
                                freq=self.qubit_freq_ef, 
                                phase=0,
                                gain=self.qubit_pi_ef_gain, 
                                length=self.qubit_pi_ef_sigma)
                
                    self.pulse(ch=self.q_ch)
                    self.sync_all()

                    self.set_pulse_registers(
                        ch=self.sideband_ch,
                        style="const",
                        freq=self.freq2reg(cfg.device.soc.sideband.fngnp1_freqs[cfg.expt.mode][i]),  # freq set by update
                        phase=0,
                        gain=cfg.device.soc.sideband.pulses.fngnp1pi_gains[cfg.expt.mode][i],
                        length=self.us2cycles(cfg.device.soc.sideband.pulses.fngnp1pi_times[cfg.expt.mode][i]))

                    self.pulse(ch=self.sideband_ch)
                    self.sync_all()
        
        # Cavity displacement
        
        self.set_pulse_registers(
            ch=self.cavdr_ch,
            style="const",
            freq=self.cavdr_freq,
            phase=self.deg2reg(self.cavdr_phase, gen_ch=self.cavdr_ch),
            gain=self.cavdr_gain,
            length=self.cavdr_length)
        
        self.pulse(ch=self.cavdr_ch)
        self.sync_all()  
        
        # Parity Measurement
        
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
            
        self.pulse(ch=self.q_ch)  # Qubit pi/2 pulse
        self.sync_all(self.qubit_pi2_waittime)  # Wait for time tau = chi/2

        self.pulse(ch=self.q_ch)  # Qubit pi/2 pulse
        self.sync_all()

        self.measure(pulse_ch=self.res_ch, 
             adcs=self.readout_ch,
             pins = [0],
             adc_trig_offset=self.adc_trig_offset,
             wait=True,
             syncdelay=self.relax_delay)

class WignerTomography0pNExperiment(Experiment):
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
            wigtom=WignerTomography0pNProgram(soc, self.cfg)
            avgi, avgq = wigtom.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=False)

            avgi_col.append(avgi[0][0])
            avgq_col.append(avgq[0][0])        
        
        # Calibrate qubit probability

        avgi_col = np.array(avgi_col)
        avgq_col = np.array(avgq_col)

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
        