import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from slab.dataanalysis import get_next_filename
from slab import SlabFile


from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

class PNRQSNNStateProgram(AveragerProgram):
    def initialize(self):

        # --- Initialize parameters ---

        cfg = self.cfg
        self.cfg.update(self.cfg.expt)
        
        # --- Readout parameters
        self.res_ch= cfg.device.soc.resonator.ch
        self.res_ch_nyquist = cfg.device.soc.resonator.nyqist
        self.readout_length = self.us2cycles(cfg.device.soc.readout.length)
        self.res_freq = cfg.device.soc.readout.freq
        self.res_gain = cfg.device.soc.resonator.gain
        self.readout_ch = cfg.device.soc.readout.ch
        self.adc_trig_offset = cfg.device.soc.readout.adc_trig_offset
        self.relax_delay = self.us2cycles(cfg.device.soc.readout.relax_delay)

        # --- Sideband parameters

        self.sideband_ch = cfg.device.soc.sideband.ch
        self.sideband_nyquist = cfg.device.soc.sideband.nyqist

        # --- Qubit parameters
        self.q_ch=cfg.device.soc.qubit.ch
        self.q_ch_nyquist = cfg.device.soc.qubit.nyqist
        
        self.qubit_freq_placeholder = self.freq2reg(cfg.expt.freq_placeholder)
        self.qubit_gf_freq = self.freq2reg(cfg.device.soc.qubit.f_gf, gen_ch = self.sideband_ch)
        self.qubit_gf_gain = cfg.device.soc.qubit.pulses.pi_gf.gain

        # --- Initialize pulses ---

        # set the nyquist zone
        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist)
        self.declare_gen(ch=self.q_ch, nqz=self.cfg.device.soc.qubit.nyqist)
        self.declare_gen(ch=self.sideband_ch, nqz=self.sideband_nyquist)

        # configure the readout lengths and downconversion frequencies
        for ch in self.readout_ch:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, 
                                 length=self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0]),
                                 freq=cfg.device.soc.readout.freq, 
                                 gen_ch=self.cfg.device.soc.resonator.ch)

        #initializing pulse register
        # add qubit and readout pulses to respective channels
            
        self.sigma_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma, gen_ch=self.q_ch)
        self.sigma_ge2 = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ge.sigma, gen_ch=self.q_ch)
        self.sigma_ef = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.q_ch)

        try: 
            self.chi_e = self.cfg.expt.chi_e
        except:
            self.chi_e = 0
        try:
            self.chi_ef = self.cfg.expt.chi_ef
        except:
            self.chi_ef = 0


        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'gauss':
            self.add_gauss(ch=self.q_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)
        if self.cfg.device.soc.qubit.pulses.pi2_ge.pulse_type == 'gauss':
            self.add_gauss(ch=self.q_ch, name="qubit_ge2", sigma=self.sigma_ge2, length=self.sigma_ge2 * 4)
        if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'gauss':
            self.add_gauss(ch=self.q_ch, name="qubit_ef", sigma=self.sigma_ef, length=self.sigma_ef * 4)
        if self.cfg.expt.qubit_prep_pulse_type == 'gauss':
            self.add_gauss(ch=self.q_ch, name="qubit_prep", sigma=self.us2cycles(self.cfg.expt.qubit_prep_length), length=self.us2cycles(self.cfg.expt.qubit_prep_length) * 4)
        self.add_gauss(ch=self.sideband_ch, name="sb_flat_top", sigma=self.us2cycles(cfg.expt.sb_ramp_sigma), length=self.us2cycles(cfg.expt.sb_ramp_sigma) * 4)
        print('sb ramp sigma', cfg.expt.sb_ramp_sigma)
        # convert frequency to DAC frequency
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

    def play_pige_pulse(self, phase = 0, shift = 0):

        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'const':

            self.set_pulse_registers(
                    ch=self.q_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift), 
                    phase=self.deg2reg(phase),
                    gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain, 
                    length=self.sigma_ge)
            
        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'gauss':
            
            self.set_pulse_registers(
                ch=self.q_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift),
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain,
                waveform="qubit_ge")
        
        self.pulse(ch=self.q_ch)

    def play_pief_pulse(self, phase = 0, shift = 0):
            
        if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'const':

            self.set_pulse_registers(
                    ch=self.q_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef + shift), 
                    phase=self.deg2reg(phase),
                    gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain, 
                    length=self.sigma_ef)
            
        if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'gauss':
            
            self.set_pulse_registers(
                ch=self.q_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef + shift),
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain,
                waveform="qubit_ef")
        
        self.pulse(ch=self.q_ch)
    
    def play_sb(self, freq= 1, length=1, gain=1, phase=0, shift=0):

        if self.cfg.expt.pulse_type == 'const':
            
            print('Sideband const')
            self.set_pulse_registers(
                    ch=self.sideband_ch, 
                    style="const", 
                    freq=self.freq2reg(freq+shift), 
                    phase=self.deg2reg(phase),
                    gain=gain, 
                    length=self.us2cycles(length))
        
        if self.cfg.expt.pulse_type == 'flat_top':
            
            print('Sideband flat top')
            self.set_pulse_registers(
                ch=self.sideband_ch,
                style="flat_top",
                freq=self.freq2reg(freq+shift),
                phase=self.deg2reg(phase),
                gain=gain,
                length=self.us2cycles(length),
                waveform="sb_flat_top")
        
        self.pulse(ch=self.sideband_ch)


    def play_pige_resolved_pulse(self, phase = 0, shift = 0):

        self.sigma_ge = self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ge_resolved.sigma, gen_ch=self.q_ch)
        self.qubit_pulsetype = self.cfg['device']['soc']['qubit']['pulses']['pi_ge_resolved']['pulse_type']

        if self.qubit_pulsetype == 'gauss':
            self.add_gauss(ch=self.q_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)
    
            self.set_pulse_registers(
                ch=self.q_ch,
                style="arb",
                freq=self.qubit_freq_placeholder,
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.qubit.pulses.pi_ge_resolved.gain,
                waveform="qubit_ge")
        
        if self.qubit_pulsetype == 'const':
            self.set_pulse_registers(
                    ch=self.q_ch, 
                    style="const", 
                    freq=self.qubit_freq_placeholder,
                    phase=0,
                    gain=self.cfg.device.soc.qubit.pulses.pi_ge_resolved.gain, 
                    length=self.sigma_ge)
            
        self.pulse(ch=self.q_ch)


    def body(self):
        # Phase reset all channels

        for ch in self.gen_chs.keys():
            if ch != 4:
                print(ch)
                self.setup_and_pulse(ch=ch, style='const', freq=self.freq2reg(100), phase=0, gain=100, length=self.us2cycles(.05), phrst=1)

        self.sync_all(10)

        # --- Initialize parameters ---

        cfg = self.cfg
        self.cfg.update(self.cfg.expt)

        # State Preparation

        print("State Preparation")
        
        chie_mode1 = self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode1]
        chie_mode2 = self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode2]
        
        chif_mode1 = self.cfg.device.soc.storage.chi_f[self.cfg.expt.mode1]
        chif_mode2 = self.cfg.device.soc.storage.chi_f[self.cfg.expt.mode2]

        chi_ef_mode1 = chif_mode1-chie_mode1
        chi_ef_mode2 = chif_mode2-chie_mode2

        for ii in range(0, self.cfg.expt.n):
            
            print('ii:', ii)

            sb_mode1_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode1][ii]
            sb_mode1_sigma =self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode1][ii]
            sb_mode1_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode1][ii]
            sb_mode2_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode2][ii]
            sb_mode2_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode2][ii]
            sb_mode2_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode2][ii]

            print(sb_mode1_freq)
            print(sb_mode2_freq)
            print(sb_mode1_sigma)
            print(sb_mode2_sigma)
            print(sb_mode1_gain)
            print(sb_mode2_gain)

            # pi_ge

            self.play_pige_pulse(phase = 0, shift = ii*(chie_mode1 + chie_mode2)/2.0)
            self.sync_all()
        
            # pi_ef

            self.play_pief_pulse(phase = 0, shift = ii*(chi_ef_mode1 + chi_ef_mode2)/2.0)
            self.sync_all()

            # pi_f0g1 on Mode 1

            self.play_sb(freq=sb_mode1_freq, length=sb_mode1_sigma, gain=sb_mode1_gain)
            self.sync_all()

            # pi_ge

            self.play_pige_pulse(phase = 0, shift = (ii+1)*chie_mode1 + ii*(chie_mode1 + chie_mode2)/2.0)
            self.sync_all()
        
            # pi_ef

            self.play_pief_pulse(phase = 0, shift = (ii+1)*chi_ef_mode1 + ii*(chie_mode1 + chie_mode2)/2.0)
            self.sync_all()

            # pi_f0g1 on Mode 2

            self.play_sb(freq=sb_mode2_freq, length=sb_mode2_sigma, gain=sb_mode2_gain)
            self.sync_all()
        
        # PNQRS 
        self.play_pige_resolved_pulse(phase = 0, shift = 0)
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
                     adc_trig_offset=self.us2cycles(self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0]),
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg.device.soc.readout.relax_delay))  # sync all channels

class PNRQSNNStateExperiment(Experiment):
    """Qubit Spectroscopy Experiment
       Experimental Config
        expt={"start":4020, "step":0.35, "expts":300, "reps": 200,"rounds":50,
          "length":5, "gain":400
         }
    """

    def __init__(self, path='', prefix='QubitProbeSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False, data_path=None, filename=None):
        
        avgi_col = []
        avgq_col = []

        fpts=self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        
        for i in tqdm(fpts, disable = not progress):
            self.cfg.expt.freq_placeholder = i
            soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
            qspec=PNRQSNNStateProgram(soc, self.cfg)
            avgi, avgq = qspec.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=False) 

            avgi_col.append(avgi[0][0])
            avgq_col.append(avgq[0][0])       
        
        avgi_col = np.array(avgi_col)
        avgq_col = np.array(avgq_col)

        iq_calib = self.qubit_prob_calib(path=self.path, config_file=self.config_file)
    
        i_prob, q_prob = self.get_qubit_prob(avgi_col, avgq_col, iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])

        data_dict = {'xpts':fpts, 'avgi':avgi_col, 'avgq':avgq_col, 'avgi_prob': i_prob, 'avgq_prob': q_prob}

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
        