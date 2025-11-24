import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from qick import *
from slab import Experiment, dsfit, AttrDict

class fngnp1RabiProgram(AveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        self.res_ch = cfg.device.soc.resonator.ch
        self.qubit_ch = cfg.device.soc.qubit.ch

        try:
            self.sideband_ch = cfg.device.soc.sideband.ch
            # print('Sideband channel found')
        except:
            self.sideband_ch = self.qubit_ch



        
        self.q_rp = self.ch_page(self.qubit_ch)     # get register page for qubit_ch
        # self.r_gain = self.sreg(self.qubit_ch, "gain")   # get gain register for qubit_ch    
        
        self.f_res=self.freq2reg(self.cfg.device.soc.readout.freq, gen_ch=self.res_ch, ro_ch=cfg.device.soc.readout.ch[0])            # conver f_res to dac register value
        self.readout_length= self.us2cycles(self.cfg.device.soc.readout.length)
        



        self.sigma_test = self.cfg.expt.length_placeholder
        # print('Sigma test (us):', self.cfg.expt.length_placeholder)
        # print('Sigma test (cycles)', self.sigma_test)

        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist)
        self.declare_gen(ch=self.qubit_ch, nqz=self.cfg.device.soc.qubit.nyqist)
        self.declare_gen(ch=self.sideband_ch, nqz=self.cfg.device.soc.sideband.nyqist)


        for ch in [0]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.soc.readout.freq, gen_ch=self.res_ch)
        
        # qubit ge and ef pulse parameters

        self.sigma_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ef = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)

        try: self.pulse_type_ge = cfg.device.soc.qubit.pulses.pi_ge.pulse_type
        except: self.pulse_type_ge = 'const'
        try: self.pulse_type_ef = cfg.device.soc.qubit.pulses.pi_ef.pulse_type
        except: self.pulse_type_ef = 'const'

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
            phase=self.deg2reg(cfg.device.soc.resonator.phase, gen_ch=self.res_ch),
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

    def play_gen_sb_pulse(self, freq= 1, length=1, gain=1, phase=0, shift=0):

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
        
        self.pulse(ch=self.sideband_ch)


    def body(self):

        cfg=AttrDict(self.cfg)
        self.sigma_fngnp1 = self.cfg.expt.length_placeholder

        # Phase reset all channels

        for ch in self.gen_chs.keys():
            if ch != 4:
                print(ch)
                self.setup_and_pulse(ch=ch, style='const', freq=self.freq2reg(100), phase=0, gain=100, length=self.us2cycles(.05), phrst=1)

        self.sync_all(10)

        chi_e = self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode]
        chi_f = self.cfg.device.soc.storage.chi_f[self.cfg.expt.mode]

        # Put n photons into cavity 

        for i in np.arange(cfg.expt.n):
            
            if self.cfg.expt.chi_correction:
                chi_ge_cor = chi_e * i
                chi_ef_cor = (chi_f - chi_e) * i
            else:
                chi_ge_cor = 0
                chi_ef_cor = 0

            # pi_ge
            self.play_pige_pulse(phase=0, shift=chi_ge_cor)
            self.sync_all()

            # pi_ef

            self.play_pief_pulse(phase=0, shift=chi_ef_cor)
            self.sync_all()

            # pi_fngnp1 
            self.play_pi_sb(n=i)
            self.sync_all()

        if self.cfg.expt.chi_correction:
            chi_ge_cor = chi_e * self.cfg.expt.n
            chi_ef_cor = (chi_f - chi_e) * self.cfg.expt.n
        else:
            chi_ge_cor = 0
            chi_ef_cor = 0

        # pi_ge
            
        self.play_pige_pulse(phase=0, shift=chi_ge_cor) 
        self.sync_all()

        # pi_ef 

        self.play_pief_pulse(phase=0, shift=chi_ef_cor)
        self.sync_all()

        # fngnp1 vs time pulse

        self.play_gen_sb_pulse(freq=cfg.expt.freq, length=self.sigma_fngnp1, gain=cfg.expt.gain, phase=0)
        self.sync_all()

        if cfg.expt.add_pi_ef:

            self.play_pief_pulse(phase=0, shift=chi_ef_cor)

        self.sync_all(self.us2cycles(0.05))


        self.measure(pulse_ch=self.res_ch,
                     adcs=[0],
                     pins=[0],
                     adc_trig_offset=cfg.device.soc.readout.adc_trig_offset,
                     wait=True,
                     syncdelay=self.us2cycles(cfg.device.soc.readout.relax_delay))  # sync all channels
        
        
        
        
class fngnp1RabiExperiment(Experiment):
    """Length Rabi Experiment
       Experimental Config
       expt_cfg={
       "start": start length, 
       "step": length step, 
       "expts": number of different length experiments, 
       "reps": number of reps,
       "gain": gain to use for the pulse
       "length_placeholder": used for iterating over lengths, initial specified value does not matter
        } 
    """

    def __init__(self, path='', prefix='LengthRabi', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, data_path=None, filename=None):
        lengths = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.length_placeholder = float(length)
            lenrabi = fngnp1RabiProgram(soc, self.cfg)
            self.prog=lenrabi
            avgi,avgq=lenrabi.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            amp=np.abs(avgi[0][0]+1j*avgq[0][0]) # Calculating the magnitude
            phase=np.angle(avgi[0][0]+1j*avgq[0][0]) # Calculating the phase
            data["xpts"].append(lengths)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)

        for k, a in data.items():
            data[k]=np.array(a)
        
        self.data = data

        avgq_col = np.array([data['avgq'][i][0][0] for i in np.arange(len(data['avgq']))])
        avgi_col = np.array([data['avgi'][i][0][0] for i in np.arange(len(data['avgi']))])

        iq_calib = self.qubit_prob_calib(path=self.path, config_file=self.config_file)
    
        i_prob, q_prob = self.get_qubit_prob(avgi_col, avgq_col, iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])

        data_dict = {'xpts': data['xpts'][0], 'avgq':avgq_col, 'avgi':avgi_col, 'i_g': [iq_calib['i_g']], 'q_g': [iq_calib['q_g']], 'i_e': [iq_calib['i_e']], 'q_e': [iq_calib['q_e']], 'avgi_prob': i_prob, 'avgq_prob': q_prob}

        if data_path and filename:
            self.save_data(data_path=data_path, filename=filename, arrays=data_dict)

        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
        
        # ex: fitparams=[1.5, 1/(2*15000), -np.pi/2, 1e8, -13, 0]
        pI = dsfit.fitdecaysin(data['xpts'][0],
                               np.array([data["avgi"][i][0][0] for i in range(len(data['avgi']))]),
                               fitparams=None, showfit=False)
        pQ = dsfit.fitdecaysin(data['xpts'][0],
                               np.array([data["avgq"][i][0][0] for i in range(len(data['avgq']))]),
                               fitparams=None, showfit=False)
        # adding this due to extra parameter in decaysin that is not in fitdecaysin
        pI = np.append(pI, data['xpts'][0][0])
        pQ = np.append(pQ, data['xpts'][0][0]) 
        data['fiti'] = pI
        data['fitq'] = pQ
        
        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data=self.data 
        print(self.fname)
        plt.figure(figsize=(10,8))
        plt.subplot(211,title="Length Rabi",  ylabel="I")
        plt.plot(data["xpts"][0], [data["avgi"][i][0][0] for i in range(len(data['avgi']))],'o-')
        plt.subplot(212, xlabel="Time (us)", ylabel="Q")
        plt.plot(data["xpts"][0], [data["avgq"][i][0][0] for i in range(len(data['avgq']))],'o-')
        plt.tight_layout()
        plt.show()


