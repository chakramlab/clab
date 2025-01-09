import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


class HistogramProgram(RAveragerProgram):
    def initialize(self):
        self.cfg.expt.expts = 2
        self.cfg.expt.rounds = 1
        self.cfg.expt.start = 0
        if self.cfg.expt.f_state:
            self.cfg.expt.step = self.cfg.device.soc.qubit.pulses.pi_ef.gain
        else:
            self.cfg.expt.step = self.cfg.device.soc.qubit.pulses.pi_ge.gain    
        self.cfg.update(self.cfg.expt)
        self.cfg = AttrDict(self.cfg)

        cfg = self.cfg

        # self.res_ch = cfg.hw.soc.dacs[cfg.device.readout.dac].ch
        # self.qubit_ch = cfg.hw.soc.dacs[cfg.device.qubit.dac].ch

        self.res_ch = cfg.device.soc.resonator.ch
        self.qubit_ch = cfg.device.soc.qubit.ch

        self.q_rp = self.ch_page(self.qubit_ch)  # get register page for qubit_ch
        self.q_gain = self.sreg(self.qubit_ch, "gain")

        # self.safe_regwi(self.q_rp, self.q_gain, cfg.expt.start)

        self.f_res = self.freq2reg(cfg.device.soc.readout.freq, gen_ch=self.res_ch, ro_ch=cfg.device.soc.readout.ch[0])  # convert f_res to dac register value
        self.readout_length = self.us2cycles(cfg.device.soc.readout.readout_length)
        # self.cfg["adc_lengths"] = [self.readout_length] * 2  # add length of adc acquisition to config
        # self.cfg["adc_freqs"] = [adcfreq(cfg.device.readout.frequency)] * 2  # add frequency of adc ddc to config

        self.pisigma = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ge.sigma)
        self.piefsigma = self.us2cycles(cfg.device.soc.qubit.pulses.pi_ef.sigma)
        # print(self.sigma)

        self.declare_gen(ch=self.res_ch, nqz=self.cfg.device.soc.resonator.nyqist) 
        self.declare_gen(ch=self.qubit_ch, nqz=self.cfg.device.soc.qubit.nyqist)

        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, 
                                 length=self.us2cycles(self.cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0]),
                                 freq=self.cfg.device.soc.readout.freq, 
                                 gen_ch=self.cfg.device.soc.resonator.ch)

        # add qubit and readout pulses to respective channels

        if self.cfg.expt.f_state:
            self.qubit_pulse_type = self.cfg.device.soc.qubit.pulses.pi_ef.pulse_type

            if self.qubit_pulse_type == "gauss":
                print('Pulse type: gauss')
                self.add_gauss(ch=self.qubit_ch, name="qubit_pief", sigma=self.piefsigma, length=self.piefsigma * 4)
                self.set_pulse_registers(
                                ch=self.qubit_ch,
                                style="arb",
                                freq=self.freq2reg(cfg.device.soc.qubit.f_ef),
                                phase=self.deg2reg(0),
                                gain=0,
                                waveform="qubit_pief")
                
            elif self.qubit_pulse_type == "const":
                print('Pulse type: const')
                self.set_pulse_registers(
                                ch=self.qubit_ch,
                                style="const",
                                freq=self.freq2reg(cfg.device.soc.qubit.f_ef),
                                phase=self.deg2reg(0),
                                gain=0,
                                length=self.piefsigma)

        else:
            self.qubit_pulse_type = self.cfg.device.soc.qubit.pulses.pi_ge.pulse_type

            if self.qubit_pulse_type == "gauss":
                print('Pulse type: gauss')
                self.add_gauss(ch=self.qubit_ch, name="qubit_pi", sigma=self.pisigma, length=self.pisigma * 4)
                self.set_pulse_registers(
                                ch=self.qubit_ch,
                                style="arb",
                                freq=self.freq2reg(cfg.device.soc.qubit.f_ge),
                                phase=self.deg2reg(0),
                                gain=0,
                                waveform="qubit_pi")
                
            elif self.qubit_pulse_type == "const":
                print('Pulse type: const')
                self.set_pulse_registers(
                                ch=self.qubit_ch,
                                style="const",
                                freq=self.freq2reg(cfg.device.soc.qubit.f_ge),
                                phase=self.deg2reg(0),
                                gain=0,
                                length=self.pisigma)
            
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(0),
            gain=cfg.device.soc.resonator.gain,
            length=self.readout_length)
        self.sync_all(self.us2cycles(0.2))

    def body(self):
        cfg = self.cfg
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

    def update(self):
        self.mathi(self.q_rp, self.q_gain, self.q_gain, '+', self.cfg.step)  # update frequency list index

    def collect_shots(self):
        # collect shots for 2 adcs (0 and 1 indexed) and I and Q channels
        cfg = self.cfg

        # print(self.di_buf[0].reshape((cfg.expt.expts, cfg.expt.reps)))
        # print(cfg.device.readout.readout_length)
        shots_i0 = self.di_buf[0].reshape((cfg.expt.expts, cfg.expt.reps)) / self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0])
        shots_q0 = self.dq_buf[0].reshape((cfg.expt.expts, cfg.expt.reps)) / self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0])
        shots_i1 = self.di_buf[1].reshape((cfg.expt.expts, cfg.expt.reps)) / self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0])
        shots_q1 = self.dq_buf[1].reshape((cfg.expt.expts, cfg.expt.reps)) / self.us2cycles(cfg.device.soc.readout.length - self.cfg.device.soc.readout.adc_trig_offset, ro_ch=self.cfg.device.soc.readout.ch[0])
        return shots_i0, shots_q0, shots_i1, shots_q1


class HistogramExperiment(Experiment):
    """Histogram Experiment
       Experimental Config
        expt =  {"reps": 10000}
    """

    def __init__(self, path='', prefix='Histogram', config_file=None, progress=None, datapath=None, filename=None):
        super().__init__(path=path, prefix=prefix, config_file=config_file, progress=progress)

        if datapath is None:
            self.datapath = None
        else:
            self.datapath = datapath

        if filename is None:
            self.filename = None
        else: 
            self.filename = filename

    def acquire(self, progress=False, debug=False):
        soc = QickConfig(self.im[self.cfg.aliases.soc].get_cfg())
        histpro = HistogramProgram(soc, self.cfg)
        x_pts, avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,
                                            progress=progress)

        data = {'xpts': x_pts, 'avgi': avgi, 'avgq': avgq}

        self.data = data
        i0, q0, i1, q1 = histpro.collect_shots()
        self.data['i0'] = i0
        self.data['q0'] = q0
        self.data['i1'] = i1
        self.data['q1'] = q1

        data_dict = {'ig': i0[0], 'qg': q0[0], 'ie': i0[1], 'qe': q0[1]}

        if self.datapath and self.filename:
            self.save_data(data_path=self.datapath, filename=self.filename, arrays=data_dict)
        return data_dict

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        i_g = data['i0'][0]
        q_g = data['q0'][0]
        i_e = data['i0'][1]
        q_e = data['q0'][1]

        fid, threshold, angle = self.hist(data=[i_g, q_g, i_e, q_e], plot=False, ran=300)
        data['fid'] = fid
        data['angle'] = angle
        data['threshold'] = threshold

        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data = self.data
        print(self.fname)

        i_g = data['i0'][0]
        q_g = data['q0'][0]
        i_e = data['i0'][1]
        q_e = data['q0'][1]

        fid, threshold, angle = self.hist(data=[i_g, q_g, i_e, q_e], plot=True, ran=200)

        plt.tight_layout()
        plt.show()
        print("fidelity:", fid)
        print("angle:", -angle*180/np.pi)
        print("threshold:", threshold)
        data['fid'] = fid
        data['angle'] = -angle*180/np.pi
        data['threshold'] = threshold

    def hist(self, data=None, plot=True, ran=1.0):
        ig = data[0]
        qg = data[1]
        ie = data[2]
        qe = data[3]

        numbins = 200

        xg, yg = np.median(ig), np.median(qg)
        xe, ye = np.median(ie), np.median(qe)

        if plot == True:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
            fig.tight_layout()

            axs[0].scatter(ig, qg, label='g', color='b', marker='*')
            axs[0].scatter(ie, qe, label='e', color='r', marker='*')
            axs[0].scatter(xg, yg, color='k', marker='o')
            axs[0].scatter(xe, ye, color='k', marker='o')
            axs[0].set_xlabel('I (a.u.)')
            axs[0].set_ylabel('Q (a.u.)')
            axs[0].legend(loc='upper right')
            axs[0].set_title('Unrotated')
            axs[0].axis('equal')
        """Compute the rotation angle"""
        theta = -np.arctan2((ye - yg), (xe - xg))
        """Rotate the IQ data"""
        ig_new = ig * np.cos(theta) - qg * np.sin(theta)
        qg_new = ig * np.sin(theta) + qg * np.cos(theta)
        ie_new = ie * np.cos(theta) - qe * np.sin(theta)
        qe_new = ie * np.sin(theta) + qe * np.cos(theta)

        """New means of each blob"""
        xg, yg = np.median(ig_new), np.median(qg_new)
        xe, ye = np.median(ie_new), np.median(qe_new)

        # print(xg, xe)

        xlims = [xg - ran, xg + ran]
        ylims = [yg - ran, yg + ran]

        if plot == True:
            axs[1].scatter(ig_new, qg_new, label='g', color='b', marker='*')
            axs[1].scatter(ie_new, qe_new, label='e', color='r', marker='*')
            axs[1].scatter(xg, yg, color='k', marker='o')
            axs[1].scatter(xe, ye, color='k', marker='o')
            axs[1].set_xlabel('I (a.u.)')
            axs[1].legend(loc='lower right')
            axs[1].set_title('Rotated')
            axs[1].axis('equal')

            """X and Y ranges for histogram"""

            ng, binsg, pg = axs[2].hist(ig_new, bins=numbins, range=xlims, color='b', label='g', alpha=0.5)
            ne, binse, pe = axs[2].hist(ie_new, bins=numbins, range=xlims, color='r', label='e', alpha=0.5)
        #        axs[2].set_xlabel('I(a.u.)')

        else:
            ng, binsg = np.histogram(ig_new, bins=numbins, range=xlims)
            ne, binse = np.histogram(ie_new, bins=numbins, range=xlims)

        """Compute the fidelity using overlap of the histograms"""
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())))
        tind = contrast.argmax()
        threshold = binsg[tind]
        fid = contrast[tind]
        # axs[2].set_title(f"Fidelity = {fid*100:.2f}%")

        return fid, threshold, theta, ig_new, qg_new, ie_new, qe_new
