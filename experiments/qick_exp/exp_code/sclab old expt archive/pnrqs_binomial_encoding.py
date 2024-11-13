import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from slab.dataanalysis import get_next_filename
from slab import SlabFile


from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

class PNRQSBinomialEncodingProgram(AveragerProgram):
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
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=self.res_freq, gen_ch=self.res_ch)


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
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        cfg=AttrDict(self.cfg)
        
        # for i in np.arange(len(cfg.expt.sideband_freqs)):

        #     qubit_phase = self.deg2reg(cfg.expt.qubit_phases[i])
        #     qubit_length = self.us2cycles(cfg.expt.qubit_lengths[i])

        #     sideband_freq = self.freq2reg(cfg.expt.sideband_freqs[i], gen_ch = self.sideband_ch)
        #     sideband_phase = self.deg2reg(cfg.expt.sideband_phases[i])
        #     sideband_length = self.us2cycles(cfg.expt.sideband_lengths[i])

        #     print('Sideband freq.:', cfg.expt.sideband_freqs[i])
        #     print('Sideband phase:', cfg.expt.sideband_phases[i])
        #     print('Sideband length:', cfg.expt.sideband_lengths[i])
        #     print('Qubit phase:', cfg.expt.qubit_phases[i])
        #     print('Qubit length:', cfg.expt.qubit_lengths[i])

        #     # Qubit_gf pulse

        #     self.set_pulse_registers(
        #                     ch=self.sideband_ch, 
        #                     style="const", 
        #                     freq=self.qubit_gf_freq, 
        #                     phase=qubit_phase,
        #                     gain=self.qubit_gf_gain, 
        #                     length=qubit_length)
        
        #     self.pulse(ch=self.sideband_ch)
        #     self.sync_all()
        
        #     # Sideband pulse

        #     self.set_pulse_registers(
        #             ch=self.sideband_ch,
        #             style="const",
        #             freq=sideband_freq, 
        #             phase=sideband_phase,
        #             gain=cfg.expt.sideband_gain,
        #             length=sideband_length)
            
        #     self.pulse(ch=self.sideband_ch)
        #     self.sync_all()

        # State Preparation

        print("State Preparation")

        # qubit_ge

        print("qubit_ge")
        print('Freq.:', self.cfg.device.soc.qubit.f_ge)
        print('Gain:', self.cfg.qubit_prep_gain)
        print('Length:', self.cfg.qubit_prep_length)
        print('Phase:', self.cfg.qubit_prep_phase)

        self.set_pulse_registers(
            ch=self.q_ch, 
            style="const", 
            freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge),
            phase=self.deg2reg(self.cfg.qubit_prep_phase),
            gain=self.cfg.qubit_prep_gain, 
            length=self.us2cycles(self.cfg.qubit_prep_length))
            
        self.pulse(ch=self.q_ch)
        self.sync_all()

        # Encoding Operation

        print('Encoding Operation')

        # 1. pi_ef
        
        print("pi_ef")
        print('Freq.:', self.cfg.device.soc.qubit.f_ef)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ef.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ef.sigma)

        self.set_pulse_registers(
            ch=self.q_ch, 
            style="const", 
            freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef),
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain, 
            length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ef.sigma))
            
        self.pulse(ch=self.q_ch)
        self.sync_all()

        # 2. pi_ge

        print("pi_ge")
        print('Freq.:', self.cfg.device.soc.qubit.f_ge)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ge.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ge.sigma)
        
        self.set_pulse_registers(
            ch=self.q_ch, 
            style="const", 
            freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge),
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain, 
            length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ge.sigma))
            
        self.pulse(ch=self.q_ch)
        self.sync_all()

        # 3. pi_f0g1

        print("pi_f0g1")
        print('Freq.:', self.cfg.device.soc.sideband.fngnp1_freqs[0][0])
        print('Gain:', self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[0][0])
        print('Length:', self.cfg.device.soc.sideband.pulses.fngnp1pi_times[0][0])
        
        self.set_pulse_registers(
            ch=self.sideband_ch,
            style="const",
            freq=self.freq2reg(self.cfg.device.soc.sideband.fngnp1_freqs[0][0]), 
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[0][0],
            length=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1pi_times[0][0]))
            
        self.pulse(ch=self.sideband_ch)
        self.sync_all()

        # 4. pi_ge

        print("pi_ge")
        print('Freq.:', self.cfg.device.soc.qubit.f_ge)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ge.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ge.sigma)

        self.set_pulse_registers(
            ch=self.q_ch, 
            style="const", 
            freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge),
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain, 
            length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ge.sigma))
            
        self.pulse(ch=self.q_ch)
        self.sync_all()

        # 5. pi_ef

        print("pi_ef")
        print('Freq.:', self.cfg.device.soc.qubit.f_ef)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ef.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ef.sigma)

        self.set_pulse_registers(
            ch=self.q_ch, 
            style="const", 
            freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef),
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain, 
            length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ef.sigma))
            
        self.pulse(ch=self.q_ch)
        self.sync_all()

        # 6. pi_ge

        print("pi_ge")
        print('Freq.:', self.cfg.device.soc.qubit.f_ge)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ge.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ge.sigma)

        self.set_pulse_registers(
            ch=self.q_ch, 
            style="const", 
            freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge),
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain, 
            length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ge.sigma))
            
        self.pulse(ch=self.q_ch)
        self.sync_all()

        # 7. pi2_f1g2

        print("pi2_f1g2")
        print('Freq.:', self.cfg.device.soc.sideband.fngnp1_freqs[0][1])
        print('Gain:', self.cfg.device.soc.sideband.pulses.pi2_fngnp1_gains[0][1])
        print('Length:', self.cfg.device.soc.sideband.pulses.pi2_fngnp1_times[0][1])

        self.set_pulse_registers(
            ch=self.sideband_ch,
            style="const",
            freq=self.freq2reg(self.cfg.device.soc.sideband.fngnp1_freqs[0][1]), 
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.sideband.pulses.pi2_fngnp1_gains[0][1],
            length=self.us2cycles(self.cfg.device.soc.sideband.pulses.pi2_fngnp1_times[0][1]))
            
        self.pulse(ch=self.sideband_ch)
        self.sync_all()

        # 8. pi_ge

        print("pi_ge")
        print('Freq.:', self.cfg.device.soc.qubit.f_ge)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ge.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ge.sigma)

        self.set_pulse_registers(
            ch=self.q_ch, 
            style="const", 
            freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge),
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain, 
            length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ge.sigma))
            
        self.pulse(ch=self.q_ch)
        self.sync_all()

        # 9. pi_ef

        print("pi_ef")
        print('Freq.:', self.cfg.device.soc.qubit.f_ef)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ef.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ef.sigma)

        self.set_pulse_registers(
            ch=self.q_ch, 
            style="const", 
            freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef),
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain, 
            length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ef.sigma))
            
        self.pulse(ch=self.q_ch)
        self.sync_all()

        # 10. pi_f2g3

        print("pi_f2g3")
        print('Freq.:', self.cfg.device.soc.sideband.fngnp1_freqs[0][2])
        print('Gain:', self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[0][2])
        print('Length:', self.cfg.device.soc.sideband.pulses.fngnp1pi_times[0][2])
        
        self.set_pulse_registers(
            ch=self.sideband_ch,
            style="const",
            freq=self.freq2reg(self.cfg.device.soc.sideband.fngnp1_freqs[0][2]), 
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[0][2],
            length=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1pi_times[0][2]))
            
        self.pulse(ch=self.sideband_ch)
        self.sync_all()

        # 11. pi_ge

        print("pi_ge")
        print('Freq.:', self.cfg.device.soc.qubit.f_ge)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ge.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ge.sigma)

        self.set_pulse_registers(
            ch=self.q_ch, 
            style="const", 
            freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge),
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain, 
            length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ge.sigma))
            
        self.pulse(ch=self.q_ch)
        self.sync_all()

        # 12. pi_f0g1

        print("pi_f0g1")
        print('Freq.:', self.cfg.device.soc.sideband.fngnp1_freqs[0][0])
        print('Gain:', self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[0][0])
        print('Length:', self.cfg.device.soc.sideband.pulses.fngnp1pi_times[0][0])

        self.set_pulse_registers(
            ch=self.sideband_ch,
            style="const",
            freq=self.freq2reg(self.cfg.device.soc.sideband.fngnp1_freqs[0][0]), 
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[0][0],
            length=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1pi_times[0][0]))
            
        self.pulse(ch=self.sideband_ch)
        self.sync_all()

        # 13. pi_ef

        print("pi_ef")
        print('Freq.:', self.cfg.device.soc.qubit.f_ef)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ef.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ef.sigma)

        self.set_pulse_registers(
            ch=self.q_ch, 
            style="const", 
            freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef),
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain, 
            length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ef.sigma))
            
        self.pulse(ch=self.q_ch)
        self.sync_all()

        # 14. pi_f0g1 and 2pi_f3g4

        print("pi_f0g1 and 2pi_f3g4")
        print('Freq.:', self.cfg.device.soc.sideband.fngnp1_freqs[0][0])
        print('Gain:', self.cfg.device.soc.sideband.pulses.pi_f0g1_2pi_f3g4_gains[0])
        print('Length:', self.cfg.device.soc.sideband.pulses.pi_f0g1_2pi_f3g4_times[0])

        self.set_pulse_registers(
            ch=self.sideband_ch,
            style="const",
            freq=self.freq2reg(self.cfg.device.soc.sideband.pulses.pi_f0g1_2pi_f3g4_freqs[0]), 
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.sideband.pulses.pi_f0g1_2pi_f3g4_gains[0],
            length=self.us2cycles(self.cfg.device.soc.sideband.pulses.pi_f0g1_2pi_f3g4_times[0]))
                
        self.pulse(ch=self.sideband_ch)
        self.sync_all()

        # 15. pi_ge

        print("pi_ge")
        print('Freq.:', self.cfg.device.soc.qubit.f_ge)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ge.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ge.sigma)

        self.set_pulse_registers(
            ch=self.q_ch, 
            style="const", 
            freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge),
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain, 
            length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ge.sigma))
            
        self.pulse(ch=self.q_ch)
        self.sync_all()

        # 16. pi_f3g4

        print("pi_f3g4")
        print('Freq.:', self.cfg.device.soc.sideband.fngnp1_freqs[0][3])
        print('Gain:', self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[0][3])
        print('Length:', self.cfg.device.soc.sideband.pulses.fngnp1pi_times[0][3])
        
        self.set_pulse_registers(
            ch=self.sideband_ch,
            style="const",
            freq=self.freq2reg(self.cfg.device.soc.sideband.fngnp1_freqs[0][3]), 
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.sideband.pulses.fngnp1pi_gains[0][3],
            length=self.us2cycles(self.cfg.device.soc.sideband.pulses.fngnp1pi_times[0][3]))

        self.pulse(ch=self.sideband_ch)
        self.sync_all()

        # 17. pi_ef

        print("pi_ef")
        print('Freq.:', self.cfg.device.soc.qubit.f_ef)
        print('Gain:', self.cfg.device.soc.qubit.pulses.pi_ef.gain)
        print('Length:', self.cfg.device.soc.qubit.pulses.pi_ef.sigma)

        self.set_pulse_registers(
            ch=self.q_ch, 
            style="const", 
            freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef),
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain, 
            length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ef.sigma))
            
        self.pulse(ch=self.q_ch)
        self.sync_all()

        # 18. pi_f1g2 and 2pi_f3g4

        print("pi_f1g2 and 2pi_f3g4")
        print('Freq.:', self.cfg.device.soc.sideband.fngnp1_freqs[0][0])
        print('Gain:', self.cfg.device.soc.sideband.pulses.pi_f1g2_2pi_f3g4_gains[0])
        print('Length:', self.cfg.device.soc.sideband.pulses.pi_f1g2_2pi_f3g4_times[0])

        self.set_pulse_registers(
            ch=self.sideband_ch,
            style="const",
            freq=self.freq2reg(self.cfg.device.soc.sideband.pulses.pi_f1g2_2pi_f3g4_freqs[0]), 
            phase=self.deg2reg(0),
            gain=self.cfg.device.soc.sideband.pulses.pi_f1g2_2pi_f3g4_gains[0],
            length=self.us2cycles(self.cfg.device.soc.sideband.pulses.pi_f1g2_2pi_f3g4_times[0]))
                
        self.pulse(ch=self.sideband_ch)
        self.sync_all()

        # PNQRS 
        
        self.sigma_ge = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge_resolved.sigma, gen_ch=self.q_ch)

        self.qubit_pulsetype = cfg['device']['soc']['qubit']['pulses']['pi_ge_resolved']['pulse_type']

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
        self.sync_all()

        self.measure(pulse_ch=self.res_ch, 
             adcs=self.readout_ch,
             pins = [0],
             adc_trig_offset=self.adc_trig_offset,
             wait=True,
             syncdelay=self.relax_delay)

class PNRQSBinomialEncodingExperiment(Experiment):
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
            qspec=PNRQSBinomialEncodingProgram(soc, self.cfg)
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
        