import numpy as np
import matplotlib.pyplot as plt

import slab.dsfit as dsf
from scipy.fft import rfft, rfftfreq

def iq_phase_unwrap(freq, i, q):

    amp = np.sqrt(i**2 + q**2)
    phase = np.angle(i + 1j*q)

    # Unwrap phase

    complex = i + 1j*q
    phase_temp = np.unwrap(phase)

    coef = np.polyfit(freq, phase_temp, 1)
    exp_unwrap = np.exp(-1j*coef[0]*freq)
    complex_unwrap = complex*exp_unwrap

    i_unwrap = np.real(complex_unwrap)
    q_unwrap = np.imag(complex_unwrap)

    amp_unwrap = np.sqrt(i_unwrap**2 + q_unwrap**2)
    phase_unwrap = np.angle(i_unwrap + 1j*q_unwrap)

    # Plot phase unwrapping

    fig, axs = plt.subplots(3, 2, figsize=(10, 9), layout='constrained')

    ax = axs[0][0]

    ax.plot(freq, phase_temp, marker='o', linestyle='--', markersize=3., label='Phase')
    ax.plot(freq, np.poly1d(coef)(freq), color='r', linestyle='--', label='Fit, slope = ' + str(np.round(coef[0], 2)))

    ax.set_title('Unwrapped Phase')
    ax.set_xlabel('Freq. (MHz)')
    ax.set_ylabel('Phase (Rad.)')
    ax.legend()

    # Plot original data

    ax = axs[1][0]
    ax.plot(freq, amp, marker='o', c='tab:blue', markersize=3., label='Lin. Mag. Squared')
    ax2 = ax.twinx()
    ax2.plot(freq, phase, marker='o', c='tab:orange', markersize=3., label='Phase')
    ax.set_zorder(1)
    ax.set_frame_on(False)

    ax.set_title('Lin. Mag. Squared and Phase')
    ax.set_xlabel('Freq. (MHz)')
    ax.set_ylabel('Lin. Mag. Squared (DAC Units^2)', color='tab:blue')
    ax.tick_params(axis='y', colors='tab:blue')
    ax2.set_ylabel('Phase (Rad.)', color='tab:orange')
    ax2.tick_params(axis='y', colors='tab:orange')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2)

    ax = axs[1][1]
    ax.plot(freq, i, marker='o', c='tab:blue', markersize=3., label='I')
    ax.plot(freq, q, marker='o', c='tab:orange', markersize=3., label='Q')

    ax.set_title('I and Q')
    ax.set_xlabel('Freq. (MHz)')
    ax.set_ylabel('Lin. Mag. (DAC Units^2)')
    ax.legend()
    
    # Plot phase-unwrapped data

    ax = axs[2][0]
    ax.plot(freq, amp_unwrap, marker='o', c='tab:blue', markersize=3., label='Lin. Mag. Squared')
    ax2 = ax.twinx()
    ax2.plot(freq, phase_unwrap, marker='o', c='tab:orange', markersize=3., label='Phase')
    ax.set_zorder(1)
    ax.set_frame_on(False)

    ax.set_title('Lin. Mag. Squared and Phase (Phase-Unwrapped)')
    ax.set_xlabel('Freq. (MHz)')
    ax.set_ylabel('Lin. Mag. Squared (DAC Units^2)', color='tab:blue')
    ax.tick_params(axis='y', colors='tab:blue')
    ax2.set_ylabel('Phase (Rad.)', color='tab:orange')
    ax2.tick_params(axis='y', colors='tab:orange')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2)

    ax = axs[2][1]
    ax.plot(freq, i_unwrap, marker='o', c='tab:blue', markersize=3., label='I')
    ax.plot(freq, q_unwrap, marker='o', c='tab:orange', markersize=3., label='Q')

    ax.set_title('I and Q (Phase-Unwrapped)')
    ax.set_xlabel('Freq. (MHz)')
    ax.set_ylabel('Lin. Mag. (DAC Units^2)')
    ax.legend()

    return i_unwrap, q_unwrap

def iq_rotation(x_sweep, i, q):
    '''
    Rotate I and Q to maximize contrast in I

    x_sweep: list or numpy array
    i: numpy array
    q: numpy array
    '''

    # Get rotation angle

    theta_list = np.linspace(0, 2*np.pi, 200)
    i_rot = [i * np.cos(j) - q * np.sin(j) for j in theta_list]
    q_rot = [i * np.sin(j) + q * np.cos(j) for j in theta_list]

    i_contrast = [np.max(j) - np.min(j) for j in i_rot]
    q_contrast = [np.max(j) - np.min(j) for j in q_rot]

    theta = theta_list[np.argmax(i_contrast)]

    i_rot = i * np.cos(theta) - q * np.sin(theta)
    q_rot = i * np.sin(theta) + q * np.cos(theta)

    # Plot

    fig, axs = plt.subplots(3, 1, figsize=(7, 6), layout='constrained')
    
    ax = axs[0]
    ax.plot(theta_list, i_contrast, c='tab:blue', label='I contrast')
    ax.plot(theta_list, q_contrast, c='tab:orange', label='Q contrast')
    ax.axvline(theta, linestyle='--', c='r', label='Max I contrast')

    ax.set_title('Contrast vs. Rotation Angle')
    ax.set_xlabel('Rotation Angle (Rad.)')
    ax.set_ylabel('Contrast')
    ax.legend()

    ax = axs[1]
    ax.plot(x_sweep, i, marker='o', c='tab:blue', markersize=3., label='I')
    ax.plot(x_sweep, q, marker='o', c='tab:orange', markersize=3., label='Q')

    ax.set_title('I and Q')
    ax.set_xlabel('Sweep Variable')
    ax.set_ylabel('Lin. Mag. (DAC Units)')
    ax.legend()

    ax = axs[2]
    ax.plot(x_sweep, i_rot, marker='o', c='tab:blue', markersize=3., label='I')
    ax.plot(x_sweep, q_rot, marker='o', c='tab:orange', markersize=3., label='Q')

    ax.set_title('I and Q (Rotated)')
    ax.set_xlabel('Sweep Variable')
    ax.set_ylabel('Lin. Mag. (DAC Units)')
    ax.legend()

    return i_rot, q_rot
    
    
class ResSpecAnalysis():
    def __init__(self, freq, i, q, config=None):
        self.freq = freq
        self.i = i
        self.q = q
        self.amp = np.sqrt(i**2 + q**2)
        self.phase = np.angle(i + 1j*q)
        self.config = config

    def analyze(self):
        
        # Fit data

        p = dsf.fitlor(self.freq, self.amp**2)
        res_freq = p[2]
        res_hwhm = p[3]
        res_q = res_freq / 2 / res_hwhm

        print("Resonator Frequency (MHz):", res_freq)
        print("Resonator HWHM (MHz):", res_hwhm)
        print("Q:", res_freq/2/res_hwhm)

        # Plot 

        fig, axs = plt.subplots(3, 1, figsize=(7, 6), layout='constrained')

        ax = axs[0]
        ax.plot(self.freq, self.amp**2, marker='o', c='tab:blue', markersize=3., label='Lin. Mag. Squared')
        ax.plot(self.freq, dsf.lorfunc(p, self.freq), linestyle='--', c='r', label='Fit') 
        ax.axvline(res_freq, linestyle='--', c='r')  

        ax.set_title('Lin. Mag. Squared')
        ax.set_xlabel('Freq. (MHz)')
        ax.set_ylabel('Lin. Mag. Squared (DAC Units^2)')
        ax.legend()

        ax=axs[1]
        ax.plot(self.freq, self.phase, marker='o', c='tab:blue', markersize=3., label='Phase')

        ax.set_title('Phase')
        ax.set_xlabel('Freq. (MHz)')
        ax.set_ylabel('Phase (Rad.)')
        ax.legend()
        
        ax = axs[2]
        ax.plot(self.freq, self.i, marker='o', c='tab:blue', markersize=3., label='I')
        ax.plot(self.freq, self.q, marker='o', c='tab:orange', markersize=3., label='Q')

        ax.set_title('I and Q')
        ax.set_xlabel('Freq. (MHz)')
        ax.set_ylabel('Lin. Mag. (DAC Units)')
        ax.legend()

        # underline suptitle of plot
        fig.suptitle('Resonator Spectroscopy', y=1.075, fontweight='bold')

    
class QubitSpecAnalysis():
    def __init__(self, freq, i, q, config=None):
        self.freq = freq
        self.i = i
        self.q = q
        self.amp = np.sqrt(i**2 + q**2)
        self.phase = np.angle(i + 1j*q)
        self.config = config

    def analyze(self):
        
        # Fit data

        p = dsf.fitlor(self.freq, -self.i)
        qubit_freq = p[2]
        qubit_hwhm = p[3]
        qubit_pipulsetime = 1/qubit_hwhm/2/np.sqrt(2)

        print("Qubit Frequency (MHz):", qubit_freq)
        print("Qubit HWHM (MHz):", qubit_hwhm)
        print("Pi-pulse time (expected) (us):", 1/qubit_hwhm/2/np.sqrt(2))

        # Plot 

        fig, axs = plt.subplots(2, 1, figsize=(7, 6), layout='constrained')

        ax = axs[0]
        ax.plot(self.freq, self.i, marker='o', c='tab:blue', markersize=3., label='I')
        ax.plot(self.freq, self.q, marker='o', c='tab:orange', markersize=3., label='Q')
        ax.plot(self.freq, -dsf.lorfunc(p, self.freq), linestyle='--', c='r', label='Fit') 
        ax.axvline(qubit_freq, linestyle='--', c='r')  

        ax.set_title('I and Q')
        ax.set_xlabel('Freq. (MHz)')
        ax.set_ylabel('Lin. Mag. (DAC Units)')
        ax.legend()

        ax = axs[1]
        ax.plot(self.freq, self.amp, marker='o', c='tab:blue', markersize=3., label='Lin. Mag.')
        ax2 = ax.twinx()
        ax2.plot(self.freq, self.phase, marker='o', c='tab:orange', markersize=3., label='Phase')
        ax.set_zorder(1)
        ax.set_frame_on(False)

        ax.set_title('Lin. Mag. and Phase')
        ax.set_xlabel('Freq. (MHz)')
        ax.set_ylabel('Lin. Mag. (DAC Units)', color='tab:blue')
        ax.tick_params(axis='y', colors='tab:blue')
        ax2.set_ylabel('Phase (Rad.)', color='tab:orange')
        ax2.tick_params(axis='y', colors='tab:orange')

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2)

class LengthRabiAnalysis():
    def __init__(self, time, i, q, config=None):
        self.time = time
        self.i = i
        self.q = q
        self.amp = np.sqrt(i**2 + q**2)
        self.phase = np.angle(i + 1j*q)
        self.config = config

    def analyze(self):
            
        # Fit data

        p = dsf.fitdecaysin(self.time, self.i, fitparams=None, showfit=False)
        
        t_pi = 1 / 2 / p[1]
        t_half_pi = t_pi / 2

        print("Pi-pulse time (us):", t_pi)
        print("Pi/2-pulse time (us):", t_half_pi)

        # Plot 

        fig, axs = plt.subplots(2, 1, figsize=(7, 6), layout='constrained')

        ax = axs[0]
        ax.plot(self.time, self.i, marker='o', c='tab:blue', markersize=3., label='I')
        ax.plot(self.time, self.q, marker='o', c='tab:orange', markersize=3., label='Q')
        ax.plot(self.time, dsf.decaysin(p, self.time), linestyle='--', c='r', label='Fit') 

        ax.set_title('I and Q')
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Lin. Mag. (DAC Units)')
        ax.legend()

        ax = axs[1]
        ax.plot(self.time, self.amp, marker='o', c='tab:blue', markersize=3., label='Lin. Mag.')
        ax2 = ax.twinx()
        ax2.plot(self.time, self.phase, marker='o', c='tab:orange', markersize=3., label='Phase')
        ax.set_zorder(1)
        ax.set_frame_on(False)

        ax.set_title('Lin. Mag. and Phase')
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Lin. Mag. (DAC Units)', color='tab:blue')
        ax.tick_params(axis='y', colors='tab:blue')
        ax2.set_ylabel('Phase (Rad.)', color='tab:orange')
        ax2.tick_params(axis='y', colors='tab:orange')

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2)

class T1Analysis():
    def __init__(self, time, i, q, config=None):
        self.time = time
        self.i = i
        self.q = q
        self.amp = np.sqrt(i**2 + q**2)
        self.phase = np.angle(i + 1j*q)
        self.config = config

    def analyze(self):

        # Fit data

        p = dsf.fitexp(self.time, self.i, fitparams=None, showfit=False)
        t1 = p[3]

        print("T1 (us):", t1)

        # Plot

        fig, axs = plt.subplots(2, 1, figsize=(7, 6), layout='constrained')

        ax = axs[0]
        ax.plot(self.time, self.i, marker='o', c='tab:blue', markersize=3., label='I')
        ax.plot(self.time, self.q, marker='o', c='tab:orange', markersize=3., label='Q')
        ax.plot(self.time, dsf.expfunc(p, self.time), linestyle='--', c='r', label='Fit')

        ax.set_title('I and Q')
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Lin. Mag. (DAC Units)')
        ax.legend()

        ax = axs[1]
        ax.plot(self.time, self.amp, marker='o', c='tab:blue', markersize=3., label='Lin. Mag.')
        ax2 = ax.twinx()
        ax2.plot(self.time, self.phase, marker='o', c='tab:orange', markersize=3., label='Phase')
        ax.set_zorder(1)
        ax.set_frame_on(False)

        ax.set_title('Lin. Mag. and Phase')
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Lin. Mag. (DAC Units)', color='tab:blue')
        ax.tick_params(axis='y', colors='tab:blue')
        ax2.set_ylabel('Phase (Rad.)', color='tab:orange')
        ax2.tick_params(axis='y', colors='tab:orange')

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2)

class PhaseRamseyAnalysis():
    def __init__(self, time, i, q, config=None):
        self.time = time
        self.i = i
        self.q = q
        self.amp = np.sqrt(i**2 + q**2)
        self.phase = np.angle(i + 1j*q)

        if not config:
            print('Error: no config file provided.')
        self.config = config
        self.qubit_freq = config['device']['soc']['qubit']['f_ge']
        step = self.config['expt']['step']
        phase_step = self.config['expt']['phase_step']
        self.ramsey_freq = phase_step/360/step

    def analyze(self):

        # Fit data
        p = dsf.fitdecaysin(self.time, self.i)
        osc_freq = p[1]
        t2 = p[3]
        offset = self.ramsey_freq - osc_freq
        qubit_freq_correction = self.qubit_freq + offset

        print('Qubit Frequency Guess (MHz):', self.qubit_freq)
        print('Offset (MHz):', offset)
        print('Suggested Qubit Frequency (MHz):', qubit_freq_correction)
        print('T2 (us):', t2)

        # Plot data

        fig, axs = plt.subplots(2, 1, figsize=(7, 6), layout='constrained')

        ax = axs[0]
        ax.plot(self.time, self.i, marker='o', c='tab:blue', markersize=3., label='I')
        ax.plot(self.time, self.q, marker='o', c='tab:orange', markersize=3., label='Q')
        ax.plot(self.time, dsf.decaysin(p, self.time), linestyle='--', c='r', label='Fit')
        
        ax.set_title('I and Q')
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Lin. Mag. (DAC Units)')
        ax.legend()

        ax = axs[1]
        ax.plot(self.time, self.amp, marker='o', c='tab:blue', markersize=3., label='Lin. Mag.')
        ax2 = ax.twinx()
        ax2.plot(self.time, self.phase, marker='o', c='tab:orange', markersize=3., label='Phase')
        ax.set_zorder(1)
        ax.set_frame_on(False)

        ax.set_title('Lin. Mag. and Phase')
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Lin. Mag. (DAC Units)', color='tab:blue')
        ax.tick_params(axis='y', colors='tab:blue')
        ax2.set_ylabel('Phase (Rad.)', color='tab:orange')
        ax2.tick_params(axis='y', colors='tab:orange')

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2)

class IQHistAnalysis():
    def __init__(self, i_g, q_g, i_e, q_e , config=None):
        self.i_g = i_g
        self.q_g = q_g
        self.i_e = i_e
        self.q_e = q_e
        self.config = config
    
    def analyze(self, bins=500, plot=True):
        i_g_med, q_g_med = np.median(self.i_g), np.median(self.q_g)
        i_e_med, q_e_med = np.median(self.i_e), np.median(self.q_e)

        if plot:  # Plot original I and Q
            fig, ax = plt.subplots(3,2, figsize=(10,10), layout='constrained')
            ax[0][0].scatter(self.i_g, self.q_g, label='Ground', c='#1f77b4')
            ax[0][0].scatter(self.i_e, self.q_e, label='Excited', c='#ff7f0e', alpha=.25)
            ax[0][0].scatter(i_g_med, q_g_med, c='#1f77b4', edgecolors='black', linewidths=.75, s=100, marker='*', label="Ground Median")
            ax[0][0].scatter(i_e_med, q_e_med, c='#ff7f0e', edgecolors='black', linewidths=.75, s=100, marker='*', label="Excited Median")

            ax[0][0].set_title('I and Q (Raw)')
            ax[0][0].set_xlabel('I')
            ax[0][0].set_ylabel('Q')
            ax[0][0].legend()

        theta = -np.arctan2(q_e_med - q_g_med, i_e_med - i_g_med)  # in radians

        i_g_rot = self.i_g * np.cos(theta) - self.q_g * np.sin(theta)
        q_g_rot = self.i_g * np.sin(theta) + self.q_g * np.cos(theta)
        i_e_rot = self.i_e * np.cos(theta) - self.q_e * np.sin(theta)
        q_e_rot = self.i_e * np.sin(theta) + self.q_e * np.cos(theta)

        i_g_med, q_g_med = np.median(i_g_rot), np.median(q_g_rot)
        i_e_med, q_e_med = np.median(i_e_rot), np.median(q_e_rot)

        if plot:  # Plot rotated I and Q
            ax[0][1].scatter(i_g_rot, q_g_rot, label='Ground', c='#1f77b4')
            ax[0][1].scatter(i_e_rot, q_e_rot, label='Excited', c='#ff7f0e', alpha=.25)
            ax[0][1].scatter(i_g_med, q_g_med, c='#1f77b4', edgecolors='black', linewidths=.75, s=100, marker='*', label="Ground Median")
            ax[0][1].scatter(i_e_med, q_e_med, c='#ff7f0e', edgecolors='black', linewidths=.75, s=100, marker='*', label="Excited Median")

            ax[0][1].set_title('I and Q (Rotated)')
            ax[0][1].set_xlabel('I')
            ax[0][1].set_ylabel('Q')
            ax[0][1].legend()

        i_lim = (np.min([np.min(i_g_rot), np.min(i_e_rot)]), np.max([np.max(i_g_rot), np.max(i_e_rot)]))
        n_g = np.histogram(i_g_rot, bins=bins, range=i_lim)
        n_e = np.histogram(i_e_rot, bins=bins, range=i_lim)
        q_lim = (np.min([np.min(q_g_rot), np.min(q_e_rot)]), np.max([np.max(q_g_rot), np.max(q_e_rot)]))
        n_g_q = np.histogram(q_g_rot, bins=bins, range=q_lim)
        n_e_q = np.histogram(q_e_rot, bins=bins, range=q_lim)

        if plot: # Plot histograms of one-dimensional I and Q
            ax[1][0].hist(i_g_rot, bins=bins, range=i_lim, label='Ground')
            ax[1][0].hist(i_e_rot, bins=bins, range=i_lim, label='Excited')

            ax[1][0].set_title('I Histogram (After Rotation)')
            ax[1][0].set_xlabel('I')
            ax[1][0].set_ylabel('Counts')
            ax[1][0].legend()

            ax[1][1].hist(q_g_rot, bins=bins, range=q_lim, label='Ground')
            ax[1][1].hist(q_e_rot, bins=bins, range=q_lim, label='Excited')

            ax[1][1].set_title('Q Histogram (After Rotation)')
            ax[1][1].set_xlabel('Q')
            ax[1][1].set_ylabel('Counts')
            ax[1][1].legend()

        cumsum_g = np.cumsum(n_g[0])
        cumsum_e = np.cumsum(n_e[0])
        diff = cumsum_g - cumsum_e

        cumsum_g_q = np.cumsum(n_g_q[0])
        cumsum_e_q = np.cumsum(n_e_q[0])
        diff_q = cumsum_g_q - cumsum_e_q

        norm = cumsum_g[-1]  # Normalize by total number of points

        if plot:  # Plot normalized cumulative sum and difference
            ax[2][0].plot(n_g[1][:-1], cumsum_g/norm, label='Ground')
            ax[2][0].plot(n_e[1][:-1], cumsum_e/norm, label='Excited')
            ax[2][0].plot(n_g[1][:-1], diff/norm, label='Difference')
            
            ax[2][0].set_title('I, Normalized Cumulative Sum')
            ax[2][0].set_xlabel('I')
            ax[2][0].set_ylabel('Normalized Cumulative Sum')
            
            ax[2][0].legend()

            ax[2][1].plot(n_g_q[1][:-1], cumsum_g_q/norm, label='Ground')
            ax[2][1].plot(n_e_q[1][:-1], cumsum_e_q/norm, label='Excited')
            ax[2][1].plot(n_g_q[1][:-1], diff_q/norm, label='Difference')

            ax[2][1].set_title('Q, Normalized Cumulative Sum')
            ax[2][1].set_xlabel('Q')
            ax[2][1].set_ylabel('Normalized Cumulative Sum')
            ax[2][1].legend()

        fid = np.max(diff/norm)
        rot_angle = theta  # radians
        print('Readout Fidelity:', fid)
        print('Rotation Angle:', rot_angle, '[rad.]', '|', rot_angle*180/np.pi, '[deg.]')



