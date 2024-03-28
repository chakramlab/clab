import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


class f0g1SidebandRamseyProgram(RAveragerProgram):
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
        self.sb_rp = self.ch_page(self.sideband_ch)  # Get register page for sideband_ch
        self.sb_phase_rp = self.sreg(self.sideband_ch, "phase")  # Get register for sideband phase

        self.sb_phase = 3
        self.safe_regwi(self.sb_rp, self.sb_phase, 0)

        self.r_wait = 3
        self.safe_regwi(self.q_rp, self.r_wait, self.us2cycles(cfg.expt.start))
        
        self.f_res=self.freq2reg(cfg.device.soc.readout.freq, gen_ch=self.res_ch, ro_ch=cfg.device.soc.readout.ch[0])  # convert f_res to dac register value
        self.readout_length=self.us2cycles(cfg.device.soc.readout.length)
        # self.cfg["adc_lengths"]=[self.readout_length]*2     #add length of adc acquisition to config
        # self.cfg["adc_freqs"]=[adcfreq(cfg.device.soc.readout.frequency)]*2   #add frequency of adc ddc to config
        
        self.pisigma = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma)
        # print(self.sigma)

        self.mode_idx = cfg.expt.mode
        self.pisblength = self.us2cycles(cfg.device.soc.sideband.pulses.f0g1pi_times[self.mode_idx])
        self.pisbgain = cfg.device.soc.sideband.pulses.f0g1pi_gains[self.mode_idx]
        print(self.pisblength, self.pisbgain, 'test')
        self.sbfreq = cfg.device.soc.sideband.f0g1_freqs[self.mode_idx]


        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist) 
        self.declare_gen(ch=self.qubit_ch, nqz=self.cfg.device.soc.qubit.nyqist)
        self.declare_gen(ch=self.sideband_ch, nqz=self.cfg.device.soc.sideband.nyqist)


        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.soc.readout.freq, gen_ch=self.res_ch)

        # qubit ge and ef pulse parameters

        self.sigma_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ge2 = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ef = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)
        self.sigma_ef2 = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ef.sigma, gen_ch=self.qubit_ch)

        try: self.pulse_type_ge = cfg.device.soc.qubit.pulses.pi_ge.pulse_type
        except: self.pulse_type_ge = 'const'
        try: self.pulse_type_ge2 = cfg.device.soc.qubit.pulses.pi2_ge.pulse_type
        except: self.pulse_type_ge2 = 'const'

        try: self.pulse_type_ef = cfg.device.soc.qubit.pulses.pi_ef.pulse_type
        except: self.pulse_type_ef = 'const'
        try: self.pulse_type_ef2 = cfg.device.soc.qubit.pulses.pi2_ef.pulse_type
        except: self.pulse_type_ef2 = 'const'

        print('Pulse type_ge: ' + self.pulse_type_ge)
        print('Pulse type_ef: ' + self.pulse_type_ef)
        print('Pulse type_ge2: ' + self.pulse_type_ge2)
        print('Pulse type_ef2: ' + self.pulse_type_ef2)

        if self.pulse_type_ge == 'gauss':

            self.add_gauss(ch=self.qubit_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)

        if self.pulse_type_ge2 == 'gauss':

            self.add_gauss(ch=self.qubit_ch, name="qubit_ge2", sigma=self.sigma_ge2, length=self.sigma_ge2 * 4)
        
        if self.pulse_type_ef == 'gauss':

            self.add_gauss(ch=self.qubit_ch, name="qubit_ef", sigma=self.sigma_ef, length=self.sigma_ef * 4)

        if self.pulse_type_ef2 == 'gauss':

            self.add_gauss(ch=self.qubit_ch, name="qubit_ef2", sigma=self.sigma_ef2, length=self.sigma_ef2 * 4)

        if self.cfg.device.soc.sideband.pulses.f0g1pi_pulse_types[self.cfg.expt.mode] == 'flat_top':
            self.add_gauss(ch=self.sideband_ch, name="sb_flat_top_pi", sigma=self.us2cycles(self.cfg.expt.sb_sigma), length=self.us2cycles(self.cfg.expt.sb_sigma) * 4)

        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.soc.resonator.phase, gen_ch=self.res_ch),
            gain=cfg.device.soc.resonator.gain,
            length=self.readout_length)
        
        print('updated3')

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
    
    def play_pi2ge_pulse(self, phase = 0, shift = 0):

        if self.cfg.device.soc.qubit.pulses.pi2_ge.pulse_type == 'const':

            self.set_pulse_registers(
                    ch=self.qubit_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift), 
                    phase=self.deg2reg(phase),
                    gain=self.cfg.device.soc.qubit.pulses.pi2_ge.gain, 
                    length=self.sigma_ge2)
            
        if self.cfg.device.soc.qubit.pulses.pi2_ge.pulse_type == 'gauss':
            
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift),
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.qubit.pulses.pi2_ge.gain,
                waveform="qubit_ge2")
        
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

    def play_pi2ef_pulse(self, phase = 0, shift = 0):
            
        if self.cfg.device.soc.qubit.pulses.pi2_ef.pulse_type == 'const':

            self.set_pulse_registers(
                    ch=self.qubit_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef + shift), 
                    phase=self.deg2reg(phase),
                    gain=self.cfg.device.soc.qubit.pulses.pi2_ef.gain, 
                    length=self.sigma_ef2)
            
        if self.cfg.device.soc.qubit.pulses.pi2_ef.pulse_type == 'gauss':
            
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef + shift),
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.qubit.pulses.pi2_ef.gain,
                waveform="qubit_ef2")
        
        self.pulse(ch=self.qubit_ch)

    def init_pi_sb(self, phase=0, shift=0):

        if self.cfg.device.soc.sideband.pulses.f0g1pi_pulse_types[self.cfg.expt.mode] == 'const':
            
            print('Sideband const')
            self.set_pulse_registers(
                ch=self.sideband_ch,
                style="const",
                freq=self.freq2reg(self.cfg.device.soc.sideband.f0g1_freqs[self.cfg.expt.mode] + shift),  # freq set by update
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.sideband.pulses.f0g1pi_gains[self.cfg.expt.mode],
                length=self.us2cycles(self.cfg.device.soc.sideband.pulses.f0g1pi_times[self.cfg.expt.mode]))
            
        if self.cfg.device.soc.sideband.pulses.f0g1pi_pulse_types[self.cfg.expt.mode] == 'flat_top':
            
            print('Sideband flat top')
            self.set_pulse_registers(
                ch=self.sideband_ch,
                style="flat_top",
                freq=self.freq2reg(self.cfg.device.soc.sideband.f0g1_freqs[self.cfg.expt.mode] +shift),
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.sideband.pulses.f0g1pi_gains[self.cfg.expt.mode],
                length=self.us2cycles(self.cfg.device.soc.sideband.pulses.f0g1pi_times[self.cfg.expt.mode]),
                waveform="sb_flat_top_pi")
            print(self.cfg.device.soc.sideband.f0g1_freqs[self.cfg.expt.mode], self.cfg.device.soc.sideband.pulses.f0g1pi_gains[self.cfg.expt.mode], self.cfg.device.soc.sideband.pulses.f0g1pi_times[self.cfg.expt.mode])

    def body(self):
        cfg=AttrDict(self.cfg)

        # Phase reset

        for ch in self.gen_chs.keys():
            if ch != 4:
                print(ch)
                self.setup_and_pulse(ch=ch, style='const', freq=self.freq2reg(100), phase=0, gain=100, length=self.us2cycles(.05), phrst=1)

        self.sync_all(10)

        # setup and play pi/2 ge qubit pulse

        self.play_pi2ge_pulse()
        self.sync_all()

        # setup and play pi ef qubit pulse

        self.play_pief_pulse()
        self.sync_all()

        # setup and play f0g1 sideband pi pulse

        self.init_pi_sb()
        self.safe_regwi(self.sb_rp, self.sb_phase_rp, 0)
        self.pulse(ch=self.sideband_ch)
        self.sync_all()

        # idle
        self.sync(self.q_rp, self.r_wait) # sets internal time offset to value stored in register self.r_wait in page self.q_rp
        
        # Advance sideband phase
        self.mathi(self.sb_rp, self.sb_phase_rp, self.sb_phase, "+", 0)

        # Done in function update(self)

        # swap back
        self.pulse(ch=self.sideband_ch)

        self.sync_all()

        # Setup ef pi pulse

        self.play_pief_pulse()
        self.sync_all()

        # setup and play pi/2 ge qubit pulse

        self.play_pi2ge_pulse()
        self.sync_all()

        self.measure(pulse_ch=self.res_ch,
                     adcs=[1, 0],
                     adc_trig_offset=cfg.device.soc.readout.adc_trig_offset,
                     wait=True,
                     syncdelay=self.us2cycles(cfg.device.soc.readout.relax_delay))  # sync all channels
    
    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+', 
                   self.us2cycles(self.cfg.expt.step))  # update wait time
        self.mathi(self.sb_rp, self.sb_phase, self.sb_phase, "+", 
                   self.deg2reg(self.cfg.expt.phase_step, gen_ch=self.sideband_ch))  # Update sideband phase
                      
class f0g1SidebandRamseyExperiment(Experiment):
    """T1 Experiment
       Experimental Config
        expt =  {"start":0, "step": 1, "expts":200, "reps": 10, "rounds": 200}
    """

    def __init__(self, path='', prefix='T1', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)
    def acquire(self, progress=False, debug=False, data_path=None, filename=None):
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        t1 = f0g1SidebandRamseyProgram(soc, self.cfg)
        x_pts, avgi, avgq = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress)
                
        iq_calib = self.qubit_prob_calib(path=self.path, config_file=self.config_file)
    
        i_prob, q_prob = self.get_qubit_prob(avgi[0][0], avgq[0][0], iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])

        data_dict = {'xpts': x_pts, 'avgq':avgq[0][0], 'avgi':avgi[0][0], 'i_g': [iq_calib['i_g']], 'q_g': [iq_calib['q_g']], 'i_e': [iq_calib['i_e']], 'q_e': [iq_calib['q_e']], 'avgi_prob': i_prob, 'avgq_prob': q_prob}

        if data_path and filename:
            self.save_data(data_path, filename, arrays=data_dict)

        return data_dict

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
        
        pI = dsfit.fitexp(data['xpts'], data['avgi'][0][0], fitparams=None, showfit=False)
        pQ = dsfit.fitexp(data['xpts'], data['avgq'][0][0], fitparams=None, showfit=False)
        # adding this due to extra parameter in decaysin that is not in fitdecaysin
        pI = np.append(pI, data['xpts'][0])
        pQ = np.append(pQ, data['xpts'][0]) 
        data['fiti'] = pI
        data['fitq'] = pQ
        print("T1:", data['fiti'][3], data['fitq'][3])
        
        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data=self.data
        print(self.fname)
        # Writing in progress, may want to edit in the future to plot time instead of clock cycles
        plt.figure(figsize=(10,8))
        plt.subplot(211,title="T1",  ylabel="I")
        plt.plot(data["xpts"], data["avgi"][0][0],'o-')
        if "fiti" in data:
            plt.plot(data["xpts"], dsfit.expfunc(data["fiti"], data["xpts"]))
        plt.subplot(212, xlabel="Wait Time (us)", ylabel="Q")
        plt.plot(data["xpts"], data["avgq"][0][0],'o-')
        if "fitq" in data:
            plt.plot(data["xpts"], dsfit.expfunc(data["fitq"], data["xpts"]))
            
        plt.tight_layout()
        plt.show()
