import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from slab.dataanalysis import get_next_filename
from slab import SlabFile


from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

class NOONStateCoherenceProgram(AveragerProgram):
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
        self.adc_trig_offset = self.us2cycles(cfg.device.soc.readout.adc_trig_offset)
        self.relax_delay = self.us2cycles(cfg.device.soc.readout.relax_delay)

        # Qubit parameters
        self.q_ch=cfg.device.soc.qubit.ch
        self.qubit_ch=cfg.device.soc.qubit.ch
        self.q_reg_page =self.ch_page(self.q_ch)     # get register page for qubit_ch
        self.q_freq_reg = self.sreg(self.q_ch, "freq")   # get frequency register for qubit_ch
        self.q_phase_reg = self.sreg(self.q_ch, "phase")
        self.q_ch_nyquist = cfg.device.soc.qubit.nyqist

        self.qubit_freq = self.freq2reg(cfg.device.soc.qubit.f_ge, gen_ch = self.q_ch)
        self.qubit_pi2_gain = cfg.device.soc.qubit.pulses.pi2_ge.gain
        self.qubit_pi2_sigma = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ge.sigma, gen_ch=self.q_ch)
        self.qubit_pi2_pulsetype = cfg['device']['soc']['qubit']['pulses']['pi2_ge']['pulse_type']

        # --- Initialize pulses ---

        # set the nyquist zone
        self.declare_gen(ch=self.res_ch, nqz=1)
        self.declare_gen(ch=self.q_ch, nqz=2)

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
        
        self.sigma_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ge2 = self.us2cycles(cfg.device.soc.qubit.pulses.pi2_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_ef = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)

        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ge", sigma=self.sigma_ge, length=self.sigma_ge * 4)
        
        self.add_gauss(ch=self.qubit_ch, name="qubit_ge2", sigma=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi2_ge.sigma, gen_ch=self.qubit_ch), length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi2_ge.sigma, gen_ch=self.qubit_ch) * 4)

        if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="qubit_ef", sigma=self.sigma_ef, length=self.sigma_ef * 4)

        self.add_gauss(ch=self.qubit_ch, name="qubit_ef_theta", sigma=self.us2cycles(self.cfg.expt.ef_length), length=self.us2cycles(self.cfg.expt.ef_length) * 4)

        # Initialize sideband pulses

        try:
            self.sideband_ch = cfg.device.soc.sideband.ch
            print('Sideband channel found')
        except:
            print('No sideband channel found')

        self.declare_gen(ch=self.sideband_ch, nqz=self.cfg.device.soc.sideband.nyqist)
        
        self.synci(500)  # give processor some time to configure pulses

        

    def play_sb(self, freq= 1, length=1, gain=1, pulse_type='flat_top', ramp_type='sin_squared', ramp_sigma=0.01, phase=0, shift=0):
        
        # why not add this inside the if statement?
        self.add_gauss(ch=self.sideband_ch, name="sb_flat_top_gaussian", sigma=self.us2cycles(ramp_sigma), length=self.us2cycles(ramp_sigma) * 4)
        self.add_cosine(ch=self.sideband_ch, name="sb_flat_top_sin_squared", length=self.us2cycles(ramp_sigma) * 2)

        if pulse_type == 'const':
            
            print('Sideband const')
            self.set_pulse_registers(
                    ch=self.sideband_ch, 
                    style="const", 
                    freq=self.freq2reg(freq+shift), 
                    phase=self.deg2reg(phase),
                    gain=gain, 
                    length=self.us2cycles(length))
        
        elif pulse_type == 'flat_top':
            
            if ramp_type == 'sin_squared':
                print('Sideband flat top sin squared')
                self.set_pulse_registers(
                    ch=self.sideband_ch,
                    style="flat_top",
                    freq=self.freq2reg(freq+shift),
                    phase=self.deg2reg(phase),
                    gain=gain,
                    length=self.us2cycles(length),
                    waveform="sb_flat_top_sin_squared")

            elif ramp_type == 'gaussian':
                print('Sideband flat top gaussian')
                self.set_pulse_registers(
                    ch=self.sideband_ch,
                    style="flat_top",
                    freq=self.freq2reg(freq+shift),
                    phase=self.deg2reg(phase),
                    gain=gain,
                    length=self.us2cycles(length),
                    waveform="sb_flat_top_gaussian")
        
        self.pulse(ch=self.sideband_ch)

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
    
    def play_pi2_ge_pulse(self, phase = 0, shift = 0):

        
        
        if self.cfg.device.soc.qubit.pulses.pi2_ge.pulse_type == 'const':
            print('pi2_ge const')
            self.set_pulse_registers(
                    ch=self.qubit_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift), 
                    phase=self.deg2reg(phase),
                    gain=self.cfg.device.soc.qubit.pulses.pi2_ge.gain, 
                    length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi2_ge.sigma, gen_ch=self.qubit_ch))
            
        if self.cfg.device.soc.qubit.pulses.pi2_ge.pulse_type == 'gauss':
            print('pi2_ge gauss')
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge + shift),
                phase=self.deg2reg(phase),
                gain=self.cfg.device.soc.qubit.pulses.pi2_ge.gain,
                waveform="qubit_ge2")
        
        print('Playing pi2_ge pulse, length = ' + str(self.cfg.device.soc.qubit.pulses.pi2_ge.sigma) + ' us' + ', gain = ' + str(self.cfg.device.soc.qubit.pulses.pi2_ge.gain))
    
        self.pulse(ch=self.qubit_ch)

    def play_thetaef_pulse(self, phase = 0, shift = 0):


        if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'const':

            self.set_pulse_registers(
                    ch=self.qubit_ch, 
                    style="const", 
                    freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef + shift), 
                    phase=self.deg2reg(phase),
                    gain=self.cfg.expt.ef_gain,
                    length=self.us2cycles(self.cfg.expt.ef_length))
            
        if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'gauss':
            
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef + shift),
                phase=self.deg2reg(phase),
                gain=self.cfg.expt.ef_gain,
                waveform="qubit_ef_theta")

        print ('Playing theta_ef pulse, length = ' + str(self.cfg.expt.ef_length) + ' us')
        
        self.pulse(ch=self.qubit_ch)

    def body(self):
        
        cfg = self.cfg
        self.cfg.update(self.cfg.expt)
        
        # Phase reset all channels

        for ch in self.gen_chs.keys():
            if ch != 4:  # Ch 4 is the readout channel
                print(ch)
                self.setup_and_pulse(ch=ch, style='const', freq=self.freq2reg(100), phase=0, gain=100, length=self.us2cycles(.05), phrst=1)

        self.sync_all(100)

        # Prepare NOON State

        # --- Initialize parameters ---

        cfg = self.cfg
        self.cfg.update(self.cfg.expt)

        if self.cfg.expt.chi_correction:
            chi_e_mode1 = self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode1]
            chi_f_mode1 = self.cfg.device.soc.storage.chi_f[self.cfg.expt.mode1]
            chi_ef_mode1 = chi_f_mode1 - chi_e_mode1
            chi_e_mode2 = self.cfg.device.soc.storage.chi_e[self.cfg.expt.mode2]
            chi_f_mode2 = self.cfg.device.soc.storage.chi_f[self.cfg.expt.mode2]
            chi_ef_mode2 = chi_f_mode2 - chi_e_mode2
        else:
            chi_e_mode1 = 0
            chi_f_mode1 = 0
            chi_ef_mode1 = 0
            chi_e_mode2 = 0
            chi_f_mode2 = 0
            chi_ef_mode2 = 0

        print('Chi e mode 1:', chi_e_mode1)
        print('Chi f mode 1:', chi_f_mode1)
        print('Chi ef mode 1:', chi_ef_mode1)
        print('Chi e mode 2:', chi_e_mode2)
        print('Chi f mode 2:', chi_f_mode2)
        print('Chi ef mode 2:', chi_ef_mode2)

        # --- State preparation ---
        
        for i in range(self.cfg.expt.n):
            print('i:', i)
            if i == 0:
                # Step 1
                # pi_ge 
                self.play_pige_pulse()
                self.sync_all()

                # Step 2
                # theta_ef
                self.play_thetaef_pulse()
                self.sync_all()
            else:
                # Step (1+6i)
                # pi_ef (shifted by i*chi_ef_mode1)
                self.play_pief_pulse(shift=i*chi_ef_mode1)
                self.sync_all()

                # Step (2+6i)
                # pi_ge (shifted by i*chi_e_mode2)
                # Shelving in the e manifold
                self.play_pige_pulse(shift=i*chi_e_mode2)
                self.sync_all()

            # Step (3 + 6i)
            # pi_sb (fi-g(i+1)) mode 1

            sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode1][i]
            sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode1][i]
            sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode1][i]
            sb_pulse_type = 'flat_top'
            sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode1][i]
            sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode1]
            print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma), ', ramp_type = ' + str(sb_ramp_type))

            self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
            self.sync_all()

            # Step (4 + 6i)
            # pi_ef (shifted by i*chi_ef_mode2)
            self.play_pief_pulse(shift=i*chi_ef_mode2)
            self.sync_all()

            # Step (5+6i) 
            # pi_ge (shifted by (i+1)*chi_e_mode1)
            # Shelving in the e manifold. Except: NO SHELVING IN THE LAST STEP
            if i != self.cfg.expt.n-1:
                self.play_pige_pulse(shift=(i+1)*chi_e_mode1)
                self.sync_all()

            # Step (6+6i) or Final step (5+6n):  
            # pi_sb (fi-g(i+1)) mode 2
            sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode2][i]
            sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode2][i]
            sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode2][i]
            sb_pulse_type = 'flat_top'
            sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode2][i]
            sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode2]
            print('Playing sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma), ', ramp_type = ' + str(sb_ramp_type))

            self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
            self.sync_all()

        if self.cfg.expt.swap_mode1 == True:
            
            sb_freq = self.cfg.device.soc.sideband.fngnp1_freqs[self.cfg.expt.mode1][self.cfg.expt.n-1]
            sb_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_times[self.cfg.expt.mode1][self.cfg.expt.n-1]
            sb_gain = self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[self.cfg.expt.mode1][self.cfg.expt.n-1]
            sb_pulse_type = 'flat_top'
            sb_ramp_sigma = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_sigmas[self.cfg.expt.mode1][self.cfg.expt.n-1]
            sb_ramp_type = self.cfg.device.soc.sideband.pulses.fngnp1pi_ramp_types[self.cfg.expt.mode1]
            print('Playing measure sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma), ', ramp_type = ' + str(sb_ramp_type))

            self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
            self.sync_all()

            if cfg.expt.add_pi_ef:
                print('Adding pi_ef pulse for calibration')
                self.play_pief_pulse(shift=(self.cfg.expt.n-1)*(chi_ef_mode1))
                self.sync_all()
        
        else:
            # Same as final prep sideband pulse
            print('Playing measure sideband pulse, freq = ' + str(sb_freq) + ', length = ' + str(sb_sigma) + ', gain = ' + str(sb_gain), ', ramp_sigma = ' + str(sb_ramp_sigma), ', ramp_type = ' + str(sb_ramp_type))

            self.play_sb(freq=sb_freq, length=sb_sigma, gain=sb_gain, pulse_type=sb_pulse_type, ramp_type=sb_ramp_type, ramp_sigma=sb_ramp_sigma)
            self.sync_all()

            if cfg.expt.add_pi_ef:
                print('Adding pi_ef pulse')
                self.play_pief_pulse(shift=(self.cfg.expt.n-1)*(chi_ef_mode2))
                self.sync_all()

        # # Prepare qubit state

        # self.add_gauss(ch=self.qubit_ch, name="qubit_prep", sigma=self.us2cycles(0.023251488095238096), length=self.us2cycles(0.023251488095238096) * 4)
        
        # self.set_pulse_registers(
        #     ch=self.q_ch,
        #     style="arb",
        #     freq=self.qubit_freq,
        #     phase=self.deg2reg(0),
        #     gain=20367,
        #     waveform="qubit_prep")
        
        # self.pulse(ch=self.q_ch)  
        # self.sync_all()

        # Qubit tomography 

        if self.cfg.expt.tomography_pulsetype == 'pi2_x':

            print('Playing pi2_x pulse')

            self.play_pi2_ge_pulse(phase=0)
            self.sync_all()  
    
        elif self.cfg.expt.tomography_pulsetype == 'pi2_y':
            
            print('Playing pi2_y pulse')

            self.play_pi2_ge_pulse(phase=90) 
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

class NOONStateCoherenceExperiment(Experiment):
    """Qubit Spectroscopy Experiment
       Experimental Config
        expt={"start":4020, "step":0.35, "expts":300, "reps": 200,"rounds":50,
          "length":5, "gain":400
         }
    """

    def __init__(self, path='', prefix='WignerTomography', config_file=None, progress=None):
        super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False, data_path=None, filename=None):

        avgi_col = []
        avgq_col = []
        avgi_pi2_x_col = []
        avgq_pi2_x_col = []
        avgi_pi2_y_col = []
        avgq_pi2_y_col = []

        # Qubit tomography for identity
        
        self.cfg.expt.tomography_pulsetype = 'I'
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        wigtom=NOONStateCoherenceProgram(soc, self.cfg)
        avgi, avgq = wigtom.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=False)

        avgi_col.append(avgi[0][0])
        avgq_col.append(avgq[0][0])        
        
        avgi_col = np.array(avgi_col)
        avgq_col = np.array(avgq_col)

        # Qubit tomography for pi2_x

        self.cfg.expt.tomography_pulsetype = 'pi2_x'
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        wigtom=NOONStateCoherenceProgram(soc, self.cfg)
        avgi, avgq = wigtom.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=False)

        avgi_pi2_x_col.append(avgi[0][0])
        avgq_pi2_x_col.append(avgq[0][0])        
        
        avgi_pi2_x_col = np.array(avgi_pi2_x_col)
        avgq_pi2_x_col = np.array(avgq_pi2_x_col)

        # Qubit tomography for pi2_y

        self.cfg.expt.tomography_pulsetype = 'pi2_y'
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        wigtom=NOONStateCoherenceProgram(soc, self.cfg)
        avgi, avgq = wigtom.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=False)

        avgi_pi2_y_col.append(avgi[0][0])
        avgq_pi2_y_col.append(avgq[0][0])        
        
        avgi_pi2_y_col = np.array(avgi_pi2_y_col)
        avgq_pi2_y_col = np.array(avgq_pi2_y_col)

        # Calibrate qubit probability

        iq_calib = self.qubit_prob_calib(path=self.path, config_file=self.config_file)
    
        i_prob, q_prob = self.get_qubit_prob(avgi_col, avgq_col, iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])
        i_pi2_x_prob, q_pi2_x_prob = self.get_qubit_prob(avgi_pi2_x_col, avgq_pi2_x_col, iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])
        i_pi2_y_prob, q_pi2_y_prob = self.get_qubit_prob(avgi_pi2_y_col, avgq_pi2_y_col, iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])

        data_dict = {
            'avgq':avgq_col, 'avgi':avgi_col, 
            'avgq_pi2_x':avgq_pi2_x_col, 'avgi_pi2_x':avgi_pi2_x_col,
            'avgq_pi2_y':avgq_pi2_y_col, 'avgi_pi2_y':avgi_pi2_y_col,
            'i_g': [iq_calib['i_g']], 'q_g': [iq_calib['q_g']], 'i_e': [iq_calib['i_e']], 'q_e': [iq_calib['q_e']], 
            'avgi_prob': i_prob, 'avgq_prob': q_prob,
            'avgi_pi2_x_prob': i_pi2_x_prob, 'avgq_pi2_x_prob': q_pi2_x_prob,
            'avgi_pi2_y_prob': i_pi2_y_prob, 'avgq_pi2_y_prob': q_pi2_y_prob}

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
        