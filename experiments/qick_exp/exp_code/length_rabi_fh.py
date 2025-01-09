import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from qick import *
from slab import Experiment, dsfit, AttrDict


class LengthRabiFHProgram(AveragerProgram):
    def initialize(self):
            cfg = AttrDict(self.cfg)
            self.cfg.update(cfg.expt)

            # self.res_ch = cfg.hw.soc.dacs[cfg.device.readout.dac].ch
            # self.qubit_ch = cfg.hw.soc.dacs[cfg.device.qubit.dac].ch

            self.res_ch = cfg.device.soc.resonator.ch
            self.qubit_ch = cfg.device.soc.qubit.ch

            self.q_rp = self.ch_page(self.qubit_ch)     # get register page for qubit_ch
            self.r_gain = self.sreg(self.qubit_ch, "gain")   # get gain register for qubit_ch

            self.f_res=self.freq2reg(self.cfg.device.soc.readout.freq, gen_ch=self.res_ch, ro_ch=self.cfg.device.soc.readout.ch[0]) # convert f_res to dac register value
            self.readout_length= self.us2cycles(self.cfg.device.soc.readout.length)

            self.sigma_test = self.us2cycles(self.cfg.expt.length_placeholder)
            self.pi_ge_sigma = self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ge.sigma)
            self.sigma_ef = self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)


            # self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist)
            # self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist)

            self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist)
            self.declare_gen(ch=self.qubit_ch, nqz=self.cfg.device.soc.qubit.nyqist)

            for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
                self.declare_readout(ch=ch, 
                                 length=self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0]),
                                 freq=cfg.device.soc.readout.freq, 
                                 gen_ch=self.cfg.device.soc.resonator.ch)


            # add readout pulses to respective channels


            self.set_pulse_registers(
                ch=self.res_ch,
                style="const",
                freq=self.f_res,
                phase=self.deg2reg(cfg.device.soc.readout.phase, gen_ch=self.res_ch),
                gain=cfg.device.soc.resonator.gain,
                length=self.readout_length)

            self.sync_all(self.us2cycles(0.2))

    def play_pi_ge(self):

        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'const':
            
            self.set_pulse_registers(
                ch=self.qubit_ch, 
                style="const", 
                freq=self.freq2reg(cfg.device.soc.qubit.f_ge), 
                phase=0,
                gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain, 
                length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ge.sigma))

        if self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type == 'gauss':

            self.add_gauss(ch=self.qubit_ch, name="qubit_ge", sigma=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ge.sigma), length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ge.sigma) * 4)
    
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ge),
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.qubit.pulses.pi_ge.gain,
                waveform="qubit_ge")
        
        self.pulse(ch=self.qubit_ch)
    
    def play_pi_ef(self):

        if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'const':
            
            self.set_pulse_registers(
                ch=self.qubit_ch, 
                style="const", 
                freq=self.freq2reg(cfg.device.soc.qubit.f_ef), 
                phase=0,
                gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain, 
                length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ef.sigma))
        
        if self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type == 'gauss':

            self.add_gauss(ch=self.qubit_ch, name="qubit_ef", sigma=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ef.sigma), length=self.us2cycles(self.cfg.device.soc.qubit.pulses.pi_ef.sigma) * 4)
    
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_ef),
                phase=self.deg2reg(0),
                gain=self.cfg.device.soc.qubit.pulses.pi_ef.gain,
                waveform="qubit_ef")

        self.pulse(ch=self.qubit_ch)

    def body(self):
        cfg=AttrDict(self.cfg)
        if cfg.expt.pi_qubit_before:
            # Play pi_ge
            self.play_pi_ge()
            self.sync_all()

            # Play pi_ef pulse

            self.play_pi_ef()
            self.sync_all()

        
        # setup and play fh probe pulse

        if self.cfg.expt.pulse_type == 'const':
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="const",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_fh),  
                phase=0,
                gain=self.cfg.expt.gain,
                length=self.sigma_test)

        elif self.cfg.expt.pulse_type == 'gauss':

            self.add_gauss(ch=self.qubit_ch, name="qubit_fh", sigma=self.sigma_test, length=self.sigma_test * 4)
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.freq2reg(self.cfg.device.soc.qubit.f_fh),  # freq set by update
                phase=0,
                gain=self.cfg.expt.gain,
                waveform="qubit_fh")

        self.pulse(ch=self.qubit_ch)
        self.sync_all()

        if cfg.expt.pi_qubit_after:

            # Play pi_ef pulse

            self.play_pi_ef()
            self.sync_all()

            # Play pi_ge pulse
        
            self.play_pi_ge()
            self.sync_all()

        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
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


class LengthRabiFHExperiment(Experiment):
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
        #print('updated')
        super().__init__(path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, data_path=None, filename=None, prob_calib=True):
        
        lengths = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        data = {"xpts": [], "avgi": [], "avgq": [], "amps": [], "phases": []}
        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.length_placeholder = float(length)
            rspec = LengthRabiFHProgram(soc, self.cfg)
            self.prog = rspec
            avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            amp = np.abs(avgi[0][0] + 1j * avgq[0][0])  # Calculating the magnitude
            phase = np.angle(avgi[0][0] + 1j * avgq[0][0])  # Calculating the phase
            data["xpts"].append(lengths)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)

        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data

        avgi_col = np.array([data["avgi"][i][0][0] for i in range(len(data['avgi']))])
        avgq_col = np.array([data["avgq"][i][0][0] for i in range(len(data['avgq']))])

        if prob_calib:

            # Calibrate qubit probability

            iq_calib = self.qubit_prob_calib(path=self.path, config_file=self.config_file)
            i_prob, q_prob = self.get_qubit_prob(avgi_col, avgq_col, iq_calib['i_g'], iq_calib['q_g'], iq_calib['i_e'], iq_calib['q_e'])
            data_dict = {'xpts': data['xpts'][0], 'avgq':avgq_col, 'avgi':avgi_col, 'i_g': [iq_calib['i_g']], 'q_g': [iq_calib['q_g']], 'i_e': [iq_calib['i_e']], 'q_e': [iq_calib['q_e']], 'avgi_prob': i_prob, 'avgq_prob': q_prob}
        
        else:

            data_dict = {'xpts': data['xpts'][0], 'avgq':avgq_col, 'avgi':avgi_col}

        if data_path and filename:
            self.save_data(data_path=data_path, filename=filename, arrays=data_dict)

        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

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
            data = self.data
        print(self.fname)
        plt.figure(figsize=(10, 8))
        plt.subplot(211, title="Length Rabi", ylabel="I")
        plt.plot(data["xpts"][0], [data["avgi"][i][0][0] for i in range(len(data['avgi']))], 'o-')
        plt.subplot(212, xlabel="Time (us)", ylabel="Q")
        plt.plot(data["xpts"][0], [data["avgq"][i][0][0] for i in range(len(data['avgq']))], 'o-')
        plt.tight_layout()
        plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm_notebook as tqdm
#
# from qick import *
# from qick.helpers import gauss
# from slab import Experiment, dsfit, AttrDict
#
# class LengthRabiEFProgram(AveragerProgram):
#     def initialize(self):
#         cfg = AttrDict(self.cfg)
#         self.cfg.update(cfg.expt)
#
#         self.res_ch = cfg.hw.soc.dacs[cfg.device.readout.dac].ch
#         self.qubit_ch = cfg.hw.soc.dacs[cfg.device.qubit.dac].ch
#
#         self.q_rp = self.ch_page(self.qubit_ch)     # get register page for qubit_ch
#         self.r_gain = self.sreg(self.qubit_ch, "gain")   # get gain register for qubit_ch
#
#         self.f_res=self.freq2reg(self.cfg.device.readout.frequency)            # conver f_res to dac register value
#         self.readout_length= self.us2cycles(self.cfg.device.readout.readout_length)
#
#         self.sigma_test = self.us2cycles(self.cfg.expt.length_placeholder)
#         self.pi_ge_sigma = self.us2cycles(self.cfg.device.qubit.pulses.pi_ge.sigma)
#
#         self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist)
#         self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist)
#
#         for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
#             self.declare_readout(ch=ch, length=self.readout_length,
#                                  freq=cfg.device.readout.frequency, gen_ch=self.res_ch)
#
#
#         # add qubit and readout pulses to respective channels
#         self.add_gauss(ch=self.qubit_ch, name="qubit_pi", sigma=self.pi_ge_sigma, length=self.pi_ge_sigma * 4)
#         self.add_gauss(ch=self.qubit_ch, name="qubit_ef", sigma=self.sigma_test, length=self.sigma_test * 4)
#         self.set_pulse_registers(
#             ch=self.res_ch,
#             style="const",
#             freq=self.f_res,
#             phase=self.deg2reg(cfg.device.readout.phase, gen_ch=self.res_ch),
#             gain=cfg.device.readout.gain,
#             length=self.readout_length)
#
#         self.sync_all(self.us2cycles(0.2))
#
#
#     def body(self):
#         cfg=AttrDict(self.cfg)
#         self.set_pulse_registers(
#             ch=self.qubit_ch,
#             style="arb",
#             freq=self.freq2reg(cfg.device.qubit.f_ge),
#             phase=self.deg2reg(0),
#             gain=self.cfg.device.qubit.pulses.pi_ge.gain,
#             waveform="qubit_pi")
#         self.pulse(ch=self.qubit_ch)
#         self.sync_all()
#
#         if self.cfg.expt.length_placeholder > 0:
#             self.set_pulse_registers(
#                 ch=self.qubit_ch,
#                 style="arb",
#                 freq=self.freq2reg(self.cfg.device.qubit.f_ef),
#                 phase=self.deg2reg(0),
#                 gain=self.cfg.expt.gain,
#                 length=self.sigma_test,
#                 waveform="qubit_ef")
#             self.pulse(ch=self.qubit_ch)
#             self.sync_all()
#
#         if cfg.expt.ge_pi_after:
#             self.set_pulse_registers(
#                 ch=self.qubit_ch,
#                 style="arb",
#                 freq=self.freq2reg(cfg.device.qubit.f_ge),
#                 phase=self.deg2reg(0),
#                 gain=self.cfg.device.qubit.pulses.pi_ge.gain,
#                 waveform="qubit_pi")
#             self.pulse(ch=self.qubit_ch)
#
#         self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
#         self.measure(pulse_ch=self.res_ch,
#                      adcs=[1, 0],
#                      adc_trig_offset=cfg.device.readout.trig_offset,
#                      wait=True,
#                      syncdelay=self.us2cycles(cfg.device.readout.relax_delay))  # sync all channels
#
#
# class LengthRabiEFExperiment(Experiment):
#     """Length Rabi Experiment
#        Experimental Config
#        expt_cfg={
#        "start": start length,
#        "step": length step,
#        "expts": number of different length experiments,
#        "reps": number of reps,
#        "gain": gain to use for the pulse
#        "length_placeholder": used for iterating over lengths, initial specified value does not matter
#         }
#     """
#
#     def __init__(self, path='', prefix='LengthRabiEF', config_file=None, progress=None):
#         super().__init__(path=path,prefix=prefix, config_file=config_file, progress=progress)
#
#     def acquire(self, progress=False):
#         lengths = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
#         soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
#         data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
#         for length in tqdm(lengths, disable=not progress):
#             self.cfg.expt.length_placeholder = float(length)
#             rspec = LengthRabiEFProgram(soc, self.cfg)
#             self.prog=rspec
#             avgi,avgq=rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
#             amp=np.abs(avgi[0][0]+1j*avgq[0][0]) # Calculating the magnitude
#             phase=np.angle(avgi[0][0]+1j*avgq[0][0]) # Calculating the phase
#             data["xpts"].append(lengths)
#             data["avgi"].append(avgi)
#             data["avgq"].append(avgq)
#             data["amps"].append(amp)
#             data["phases"].append(phase)
#
#         for k, a in data.items():
#             data[k]=np.array(a)
#
#         self.data = data
#
#         return data
#
#     def analyze(self, data=None, **kwargs):
#         if data is None:
#             data=self.data
#
#         # ex: fitparams=[1.5, 1/(2*15000), -np.pi/2, 1e8, -13, 0]
#         pI = dsfit.fitdecaysin(data['xpts'][0], np.array([data["avgi"][i][0][0] for i in range(len(data['avgi']))]), fitparams=None, showfit=False)
#         pQ = dsfit.fitdecaysin(data['xpts'][0], np.array([data["avgq"][i][0][0] for i in range(len(data['avgq']))]), fitparams=None, showfit=False)
#         # adding this due to extra parameter in decaysin that is not in fitdecaysin
#         pI = np.append(pI, data['xpts'][0][0])
#         pQ = np.append(pQ, data['xpts'][0][0])
#         data['fiti'] = pI
#         data['fitq'] = pQ
#
#         return data
#
#     def display(self, data=None, **kwargs):
#         if data is None:
#             data=self.data
#         print(self.fname)
#         plt.figure(figsize=(10,8))
#         plt.subplot(211,title="Length Rabi",  ylabel="I")
#         plt.plot(data["xpts"][0], [data["avgi"][i][0][0] for i in range(len(data['avgi']))],'o')
#         plt.subplot(212, xlabel="Time (us)", ylabel="Q")
#         plt.plot(data["xpts"][0], [data["avgq"][i][0][0] for i in range(len(data['avgq']))],'o')
#         plt.tight_layout()
#         plt.show()
#
#
#
#
#
#
#
#         # plt.subplot(111, title="Length Rabi", xlabel="Time (us)", ylabel="Q")
#         # plt.plot(data["xpts"][0], [data["avgq"][i][0][0] for i in range(len(data['avgq']))],'o-')
#         # if "fiti" in data:
#         #     plt.plot(data["xpts"][0], dsfit.decaysin(data["fiti"], data["xpts"][0]))
#         # plt.show()
#