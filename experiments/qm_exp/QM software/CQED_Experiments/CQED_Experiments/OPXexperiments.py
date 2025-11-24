"""Plamen's Old codebase with all of the OPX experiments. Going to update these for Octave and OPX functionality"""


import numpy as np
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from oldConfig import *
# from dsfit import *
import matplotlib.pyplot as plt
import time
from IPython import display
import importlib, sys
import json
import matplotlib.pyplot as plt
# qmm = QuantumMachinesManager()
fres=29.5e6
class OPXexp():
    def __init__(self, *args):
    
        super().__init__()


        """ 
        Below I have a list of working experiments, each experiment has 
        a lambda function associated with it that takes a params dictionary.
        This params dictionary passes the dictionary's values into the 
        lambda function and returns the program that you need to run using
        quantum machines.

        Note that these params dictionaries are in the experiments.json file
        that is in this directory. Also note that you need to manually pass the 
        params into the lambda function by accessing the dictionary item 
        directly when implementing
        """
        


    def resspec(self, params):
        fmin = params["IF frequency range (Hz)"][0]
        fmax = params["IF frequency range (Hz)"][1] 
        df = params["frequency intervals (Hz)"]
        reset_time = params["T1 time (ns)"] * params["reset time (units of T1 time)"]
        avgs = params["averages"]
        fvec=np.arange(fmin,fmax,df)

        with program() as prog:
            f = declare(int)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_st = declare_stream()
            Q_st = declare_stream()
            update_frequency('qubit', 400e6 - 24.5e6)

            with for_(n,0,n<avgs,n+1):
                with for_(f,fmin,f<fmax,f+df):
                    update_frequency("resonator",f)
                    # update_correction("resonator",  1.0038154684007168,
                    #                                 -0.08353806287050247,
                    #                                 -0.08248425275087357,
                    #                                 1.0166401080787182)
                    
                    # align("qubit","resonator")
                    measure("readout"*amp(1),'resonator',None,
                            demod.full('integW1',I,"out1"),
                            demod.full('integW2',Q,"out1")
                           )
                    save(I,I_st)
                    save(Q,Q_st)
                    wait(int(reset_time//4),'resonator')
            with stream_processing():
                I_st.buffer(len(fvec)).average().save("I")
                Q_st.buffer(len(fvec)).average().save("Q")

        return {"prog": prog, "IF frequencies": fvec}

    def resspec_V(self,*args):
        fmin=29e6
        fmax=31e6
        df=0.05e6
        fvec=np.arange(fmin,fmax,df)

        amin=-0.5
        amax=0.04
        da=0.05
        avec=np.arange(amin,amax+da/2,da)

        with program() as prog:
            f = declare(int)
            n = declare(int)
            a = declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)
            I_st = declare_stream()
            Q_st = declare_stream()
            update_frequency('qubit', 400e6 - 24.5e6)

            ramp_to_zero("Vsource", 100)
            play("CW" * amp(-0.5), "Vsource", duration=1e4)  # 0.8 V for amp=
            wait(int(1e9))
            with for_(a, amin, a < amax + da / 2, a + da):
                play("CW" * amp(0.05), "Vsource", duration=1e4)  # 0.8 V for amp=
                wait(int(1e9))
                align("Vsource", "resonator")
                with for_(n,0,n<5000,n+1):
                    with for_(f,fmin,f<fmax,f+df):
                        update_frequency("resonator",f)
                        align("qubit","resonator")
                       # reset_phase('resonator')
                        measure("readout"*amp(0.5),'resonator',None,
                                demod.full('integW1',I,"out1"),
                                demod.full('integW2',Q,"out1")
                               )
                        save(I,I_st)
                        save(Q,Q_st)
                        wait(int(20e3//4),'resonator')
            with stream_processing():
                I_st.buffer(len(fvec)).buffer(len(avec)).average().save("I")
                Q_st.buffer(len(fvec)).buffer(len(avec)).average().save("Q")

        return prog,fvec,avec

    def resspec_pi(self, *args):
        fmin = 28.5e6
        fmax = 31.5e6
        df = 0.05e6
        fvec = np.arange(fmin, fmax, df)
        Tpi=1e3
        reset_time=8e3*5
        with program() as prog:
            f = declare(int)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_st = declare_stream()
            Q_st = declare_stream()
            update_frequency('qubit', 300e6)

            with for_(n, 0, n < 2000, n + 1):
                with for_(f, fmin, f < fmax, f + df):
                    update_frequency("resonator", f)
                    wait(int(reset_time//4))
                #    play('marker','qubit')
                    play('saturation' * amp(1), 'qubit', length=Tpi// 4)
                    align("qubit", "resonator")
                    measure("readout", 'resonator', None,
                            demod.full('integW1', I, "out1"),
                            demod.full('integW2', Q, "out1")
                            )
                    save(I, I_st)
                    save(Q, Q_st)

                    wait(int(reset_time//4))
                    align("qubit", "resonator")
                    measure("readout", 'resonator', None,
                            demod.full('integW1', I, "out1"),
                            demod.full('integW2', Q, "out1")
                            )
                    save(I, I_st)
                    save(Q, Q_st)
            with stream_processing():
                I_st.buffer(2*len(fvec)).average().save("I")
                Q_st.buffer(2*len(fvec)).average().save("Q")

        return prog, fvec

    def resspec_efpi(self, *args):
        fmin = 25e6
        fmax = 31.5e6
        df = 0.05e6
        fvec = np.arange(fmin, fmax, df)
        Tpi=375
        efTpi=335
        reset_time=8e3*5
        with program() as prog:
            f = declare(int)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_st = declare_stream()
            Q_st = declare_stream()

            with for_(n, 0, n < 2000, n + 1):
                with for_(f, fmin, f < fmax, f + df):
                    update_frequency("resonator", f)
                    wait(int(reset_time//4))
                    update_frequency('qubit', -145e6)
                    play('saturation' * amp(1), 'qubit', length=Tpi// 4)
                    update_frequency('qubit', -302e6)
                    play('saturation' * amp(1), 'qubit', length=efTpi// 4)
                    align("qubit", "resonator")
                    measure("readout" * amp(0.2), 'resonator', None,
                            demod.full('integW1', I, "out1"),
                            demod.full('integW2', Q, "out1")
                            )
                    save(I, I_st)
                    save(Q, Q_st)

                    wait(int(reset_time//4))
                    align("qubit", "resonator")
                    measure("readout" * amp(0.2), 'resonator', None,
                            demod.full('integW1', I, "out1"),
                            demod.full('integW2', Q, "out1")
                            )
                    save(I, I_st)
                    save(Q, Q_st)
            with stream_processing():
                I_st.buffer(2*len(fvec)).average().save("I")
                Q_st.buffer(2*len(fvec)).average().save("Q")

        return prog, fvec

    def resspec_pi_fscan(self, *args):
        fmin = 28.5e6
        fmax = 30.5e6
        df = 0.05e6
        fvec = np.arange(fmin, fmax, df)

        fqmin = 360e6
        fqmax = 390e6
        dfq = 1e6
        fqvec = np.arange(fqmin, fqmax, dfq)

        reset_time=2e3*5
        with program() as prog:
            f = declare(int)
            fq = declare(int)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_st = declare_stream()
            Q_st = declare_stream()
            update_frequency('qubit', 360e6)

            with for_(n, 0, n < 2000, n + 1):
                with for_(f, fmin, f < fmax, f + df):
                    update_frequency("resonator", f)
                    with for_(fq, fqmin, fq < fqmax, fq + dfq):
                        update_frequency("qubit", fq)
                        wait(int(reset_time//4))
                        play('saturation' * amp(1), 'qubit', length=1e3// 4)
                        align("qubit", "resonator")
                        measure("readout" * amp(0.5), 'resonator', None,
                                demod.full('integW1', I, "out1"),
                                demod.full('integW2', Q, "out1")
                                )
                        save(I, I_st)
                        save(Q, Q_st)

                        wait(int(reset_time//4))
                        align("qubit", "resonator")
                        measure("readout" * amp(0.5), 'resonator', None,
                                demod.full('integW1', I, "out1"),
                                demod.full('integW2', Q, "out1")
                                )
                        save(I, I_st)
                        save(Q, Q_st)
            with stream_processing():
                I_st.buffer(2*len(fqvec)).buffer(len(fvec)).average().save("I")
                Q_st.buffer(2*len(fqvec)).buffer(len(fvec)).average().save("Q")

        return prog, fvec, fqvec
    def resspec_amp(self,*args):
        fmin = 29e6
        fmax = 33e6
        df = .2e6
        fvec = np.arange(fmin, fmax, df)

        amin = 0.05
        amax = 1
        da = 0.05
        avec = np.arange(amin, amax+da/2, da)
        reset_time=5*12e3

        with program() as prog:
            fr = declare(int)
            n = declare(int)
            a = declare(fixed)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I1_stream = declare_stream()
            Q1_stream = declare_stream()

            with for_(n, 0, n < 2000, n + 1):
                with for_(fr, fmin, fr < fmax, fr + df):
                    with for_(a, amin, a < amax + da / 2, a + da):
                        update_frequency("resonator", fr)
                        measure("readout" * amp(a), 'resonator', None,
                                demod.full('integW1', I1, "out1"),
                                demod.full('integW2', Q1, "out1")
                                )
                        save(I1, I1_stream)
                        save(Q1, Q1_stream)
                        wait(int(reset_time//4), 'resonator')
            with stream_processing():
                I1_stream.buffer(len(avec)).buffer(len(fvec)).average().save("I")
                Q1_stream.buffer(len(avec)).buffer(len(fvec)).average().save("Q")
        return prog, fvec, avec

    def plotIQ(self,I,Q,fvec):
        fig = plt.figure(figsize = (10,5))
        ax = fig.add_subplot(111)
        
        ax.plot(fvec,I,label='I')
        ax.plot(fvec,Q,label='Q')
        mags = np.sqrt(I**2+Q**2)

        ax.plot(fvec,mags,'g.')

        ax.axvline(fvec[np.argmax(mags)])
        p = fitlor(fvec,mags**2,showfit=False)
        ax.plot(fvec,np.sqrt(lorfunc(p,fvec)),'r-')

        nu_res = p[2] + 7.33e9
        Qfac = nu_res/p[3]/2

        return (fig.show(),print('fr=',float('%.5g' % nu_res),print('frIF=',float('%.5g' % p[2]),'Q=',float('%.4g' % Qfac))))
        
    def qubitspec2d(self,*args):
        fmin = 380e6
        fmax = 410e6
        df=0.2e6
        reset_time=5*35e3
        fvec=np.arange(fmin,fmax,df)

        amin=0.0
        amax=1.0
        da=0.05
        avec=np.arange(amin,amax+da/2,da)
        avgs=5000
        Tpi=15e3

        with program() as prog:
            fr = declare(int)
            n = declare(int)
            I = declare(fixed)
            a = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n,0,n<avgs,n+1):
                with for_(a,amin,a<amax+da/2,a+da):
                    with for_(fr,fmin,fr<fmax,fr+df):
                        update_frequency("resonator",29.4e6)
                        update_frequency("qubit",fr)
                        wait(int(reset_time//4), "qubit")# wait for the qubit to relax, several T1s
                        #play("saturation"*amp(1), "qubit",duration = 15e3)
                        play("saturation"*amp(a), "qubit",duration =Tpi//4)
                        align("qubit", "resonator")
      #                  reset_phase('resonator')
                        measure("readout",'resonator',None,
                                demod.full('integW1',I,"out1"),
                                demod.full('integW2',Q,"out1")
                               )
                        save(I,I_stream)
                        save(Q,Q_stream)

            with stream_processing():
                I_stream.buffer(len(fvec)).buffer(len(avec)).average().save("I")
                Q_stream.buffer(len(fvec)).buffer(len(avec)).average().save("Q")
        
        return prog,fvec+4.0e9,avec

    def qubitspec(self, params):
        fmin = params["IF frequency range (Hz)"][0]
        fmax = params["IF frequency range (Hz)"][1] 
        df = params["frequency intervals (Hz)"]
        Stime = params["saturation time (ns)"]
        reset_time = params["T1 time (ns)"] * params["reset time (units of T1 time)"]
        avgs = params["averages"]
    
        fvec = np.arange(fmin, fmax, df)

        with program() as prog:
            fr = declare(int)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < avgs, n + 1):
                with for_(fr, fmin, fr < fmax, fr + df):
                    wait(int(reset_time // 4),"qubit")
                    #update_frequency("resonator", 29.4e6)
                    update_frequency("qubit", fr)
             #       update_frequency("qubit_multi", fr+9e6)
                    wait(int(reset_time // 4), "qubit")  # wait for the qubit to relax, several T1s
                    play("saturation"*amp(0.3), "qubit", duration=Stime//4)
                  #  play("saturation", "qubit_multi", duration=640)
                    align("qubit", "resonator")
                    measure("readout", 'resonator', None,
                            demod.full('integW1', I, "out1"),
                            demod.full('integW2', Q, "out1")
                            )
                    save(I, I_stream)
                    save(Q, Q_stream)


            with stream_processing():
                I_stream.buffer(len(fvec)).average().save("I")
                Q_stream.buffer(len(fvec)).average().save("Q")

        return {"prog": prog, "frequencies": fvec} #+4e9

    def qubitspec_t(self, fmin = 20e6, fmax = 330e6, df = 1e6, *args):

        reset_time = 5 * 12e3
        fvec = np.arange(fmin, fmax, df)

        avgs = 10

        with program() as prog:
            fr = declare(int)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < avgs, n + 1):
                with for_(fr, fmin, fr < fmax, fr + df):
                    update_frequency("resonator", 30.2e6)
                    update_frequency("qubit", fr)
                    wait(int(reset_time // 4), "qubit")  # wait for the qubit to relax, several T1s
                    play("saturation", "qubit", duration=246 // 4)
                    align("qubit", "resonator")
                    measure("readout" * amp(.5), 'resonator', None,
                            demod.full('integW1', I, "out1"),
                            demod.full('integW2', Q, "out1")
                            )
                    save(I, I_stream)
                    save(Q, Q_stream)

            with stream_processing():
                I_stream.buffer(len(fvec)).average().save("I")
                Q_stream.buffer(len(fvec)).average().save("Q")

        return prog, fvec + 4.7e9

    def qubitspec_gauss(self, *args):
        fmin = 400e6 - 27e6
        fmax = 400e6 - 25.5e6
        df = 0.1e6
        reset_time = 2e3
        fvec = np.arange(fmin, fmax, df)

        avgs = 10000

        with program() as prog:
            fr = declare(int)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < avgs, n + 1):
                with for_(fr, fmin, fr < fmax, fr + df):
                    update_frequency("resonator", 59.8e6)
                    update_frequency("qubit", fr)
                    wait(int(reset_time // 4), "qubit")  # wait for the qubit to relax, several T1s
                    # play("saturation" * amp(1), "qubit", duration=5e4, chirp=(1e6,'MHz/sec'))
                    play("gaussian" * amp(1), "qubit", len=1e3 // 4)
                    align("qubit", "resonator")
                    reset_phase('resonator')
                    measure("readout" * amp(0.6), 'resonator', None,
                            demod.full('integW1', I, "out1"),
                            demod.full('integW2', Q, "out1")
                            )
                    save(I, I_stream)
                    save(Q, Q_stream)

            with stream_processing():
                I_stream.buffer(len(fvec)).average().save("I")
                Q_stream.buffer(len(fvec)).average().save("Q")

        return prog, fvec + 3.7e9

    def qubitspec_V(self, *args):
        fmin=250e6
        fmax=267e6
        df=0.2e6
        reset_time=13.5e3*5

        fvec = np.arange(fmin, fmax, df)

        avgs = 5000

        amin=-1
        amax=0.05
        da=0.05
        avec=np.arange(amin,amax+da/2,da)

        with program() as prog:
            fr = declare(int)
            n = declare(int)
            a= declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()
            update_frequency("resonator", 29.5e6)

            ramp_to_zero("Vsource",100)
            play("CW" * amp(-0.3), "Vsource", duration=1e4)  # 0.8 V for amp=
            wait(int(1e9))
            with for_(a,amin,a<amax+da/2,a+da):
                play("CW" * amp(0.03), "Vsource", duration=1e4)  # 0.8 V for amp=
                wait(int(2e9))
                align("Vsource", "resonator")
                with for_(n, 0, n < avgs, n + 1):
                    with for_(fr, fmin, fr < fmax, fr + df):
                        update_frequency("qubit", fr)
                        wait(int(reset_time // 4), "qubit")  # wait for the qubit to relax, several T1s
                        play("saturation", "qubit",duration = 5e3//4)
                        align("resonator", "qubit")
                        reset_phase('resonator')
                        measure("readout" * amp(0.5), 'resonator', None,
                                demod.full('integW1', I, "out1"),
                                demod.full('integW2', Q, "out1")
                                )
                        save(I, I_stream)
                        save(Q, Q_stream)

            with stream_processing():
                I_stream.buffer(len(fvec)).buffer(len(avec)).average().save("I")
                Q_stream.buffer(len(fvec)).buffer(len(avec)).average().save("Q")

        return prog, fvec+4.7e9, avec
    
    def rabi2d(self,*args):
        dt = 20//4
        T_min = 4
        T_max = 2000//4
        times = np.arange(T_min, T_max, dt)*4

        T1=35e3

        fmin = 390e6
        fmax = 410e6
        df = 1e6
        fvec=np.arange(fmin,fmax,df)
        reset_time = 5*T1

        avgs = 5000
        a=1.0
        with program() as prog:

            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)      # Averaging
            i = declare(int)      # Amplitudes
            t = declare(int)     #array of time delays
            f = declare(int)
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()


            ###############
            # the sequence:
            ###############
            update_frequency("resonator",29.4e6)
            with for_(n, 0, n < avgs, n + 1):
                with for_(f, fmin,f<fmax, f+df):
                    update_frequency("qubit", f)

                    with for_(t, T_min, t < T_max, t + dt):
                        wait(int(reset_time//4), "qubit")
                        align("qubit", "qubit_multi")
                        play("saturation"*amp(1), "qubit", duration=t)
                     #   play("saturation", "qubit_multi",duration = t)
                        align("qubit","qubit_multi", "resonator")
                        measure("readout",'resonator',None,
                                demod.full('integW1',I,"out1"),
                                demod.full('integW2',Q,"out1")
                               )
                        save(I,I_st)
                        save(Q,Q_st)

            with stream_processing():
                I_st.buffer(len(times)).buffer(len(fvec)).average().save('I')
                Q_st.buffer(len(times)).buffer(len(fvec)).average().save('Q')

        return prog,fvec+4.0e9,times
    def HelloOctaveRes(self, params):
        dt = 20//4
        T_min = 4
        T_max = 2000//4
        times = np.arange(T_min, T_max, dt)*4

        T1=35e3

        fmin = 390e6
        fmax = 410e6
        df = 1e6
        fvec=np.arange(fmin,fmax,df)
        reset_time = 5*T1

        avgs = 5000
        a=1.0
        
        

        with program() as hello_octave_readout_1:
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()
            measure("readout"*amp(1),'resonator',None,
                            demod.full('integW1',I,"out1"),
                            demod.full('integW2',Q,"out1")
                           )
            save(I,I_st)
            save(Q,Q_st)
            with stream_processing():
                I_st.buffer(1).average().save("I")
                Q_st.buffer(1).average().save("Q")
        return {"prog": hello_octave_readout_1}
    def rabi1d(self, params):
        timeRange = params["time range (ns)"]
        timeInterval = params["time interval (ns)"]
        T1 = params["T1 time (ns)"]
        reset_time = params["reset time (units of T1 time)" ] * T1
        # Tpi = params["Tpi"]
        avgs = params["averages"]


        #  Times are given in units of the clock cycle, need to 
        # divide the time range by 4 ns before we pass it into the program
        timeRange = np.array(timeRange)/4
        dt = timeInterval/4

        ### Passing in values to T_min, T_max from time ranges/interval
        T_min = timeRange[0]
        T_max = timeRange[1]
        times = np.arange(T_min, T_max, dt) * 4

        # T1 = 35e3
        # reset_time = 5 * T1
        # Tpi=130
        # avgs =3000
        # a=1.0
        with program() as prog1:
            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)  # Averaging
            t = declare(int)  # array of time delays
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()

            ###############
            # the sequence:
            ###############
            # update_frequency("resonator", 29.4e6)
            # update_frequency("qubit", 400e6)
            # update_frequency("qubit_multi", 392e6)

            # update_frequency("qubit_multi", 448e6)
            with for_(n, 0, n < avgs, n + 1):
                # wait(int(reset_time // 4), "qubit")
                # wait(int(reset_time // 4), "resonator")
                with for_(t, T_min, t < T_max, t + dt):
                    wait(int(reset_time // 4), "qubit")
                    # align("qubit", "qubit_multi")
                    play("saturation", "qubit", duration = t)
                    # play("saturation_multi" , "qubit_multi", duration=t)
                    align("qubit", "resonator")
                    measure("readout", 'resonator', None,
                            demod.full('integW1', I, "out1"),
                            demod.full('integW2', Q, "out1")
                            )
                    save(I, I_st)
                    save(Q, Q_st)

            #     """Play a ge pi pulse and then readout"""
            #     wait(int(reset_time // 4), "qubit")
            #     # align("qubit", "qubit_multi")
            #     play("saturation", "qubit", duration=Tpi//4)
            #     # play("saturation_multi", "qubit_multi", duration=Tpi//4)
            #     align('qubit','resonator')
            # #    reset_phase('resonator')
            #     measure("readout",'resonator',None,
            #             demod.full('integW1',I,"out1"),
            #             demod.full('integW2',Q,"out1"))
            #     save(I,I_st)
            #     save(Q,Q_st)

            #     """Just readout without playing anything"""
            #     wait(int(reset_time // 4), "qubit")
            #     align('qubit','resonator')
            #  #   reset_phase('resonator')
            #     measure("readout",'resonator',None,
            #             demod.full('integW1',I,"out1"),
            #             demod.full('integW2',Q,"out1"))
            #     save(I,I_st)
            #     save(Q,Q_st)

            with stream_processing():
                I_st.buffer(len(times)+2).average().save('I')
                Q_st.buffer(len(times)+2).average().save('Q')

        return {"prog": prog1, "times": times}

    def rabi1d_discriminate(self, *args):
        dt = 4//4
        T_min = 4
        T_max = 2000 // 4
        times = np.arange(T_min, T_max, dt) * 4

        T1 = 20e3
        reset_time = 5 * T1
        Tpi=230

        fmin = 180e6
        fmax = 220e6
        df = 0.2e6
        fvec = np.arange(fmin, fmax, df)
        avgs =1000
        with program() as prog1:
            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)  # Averaging
            t = declare(int)  # array of time delays
            fr = declare(fixed)

            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()
            Id = declare(fixed)
            Qd = declare(fixed)

            I_disc = declare_stream()
            Q_disc = declare_stream()

            ###############
            # the sequence:
            ###############
            update_frequency("resonator", 31e6)

            update_frequency("qubit", 200.5e6)

            # update_frequency("qubit_multi", 448e6)
            with for_(n, 0, n < avgs, n + 1):
                with for_(fr, fmin, fr < fmax, fr + df):
                    wait(int(reset_time // 4), "qubit")
                    update_frequency("resonator", 31e6)
                    update_frequency("qubit", fr)
                    #       update_frequency("qubit_multi", fr+9e6)
                    wait(int(reset_time // 4), "qubit")  # wait for the qubit to relax, several T1s
                    play("saturation", "qubit", duration=Tpi // 4)
                    #  play("saturation", "qubit_multi", duration=640)
                    align("qubit", "resonator")
                    measure("readout", 'resonator', None,
                            demod.full('integW1', Id, "out1"),
                            demod.full('integW2', Qd, "out1")
                            )
                    save(Id, I_disc)
                    save(Qd, Q_disc)

                # f_res=fvec[np.min(Id**2+Qd**2)]
                # update_frequency("qubit", f_res)


                with for_(t, T_min, t < T_max, t + dt):
                    wait(int(reset_time // 4), "qubit")

                    align("qubit", "qubit_multi")
                    play("saturation", "qubit", duration=t)

                    align("qubit","qubit_multi", "resonator")
                    measure("readout" , 'resonator', None,
                            demod.full('integW1', I, "out1"),
                            demod.full('integW2', Q, "out1")
                            )
                    save(I, I_st)
                    save(Q, Q_st)

                """Play a ge pi pulse and then readout"""
                wait(int(reset_time // 4), "qubit")
                align("qubit", "qubit_multi")
                play("saturation", "qubit", duration=Tpi//4)
                play("saturation_multi", "qubit_multi", duration=Tpi//4)
                align('qubit','resonator')
            #    reset_phase('resonator')
                measure("readout",'resonator',None,
                        demod.full('integW1',I,"out1"),
                        demod.full('integW2',Q,"out1"))
                save(I,I_st)
                save(Q,Q_st)

                """Just readout without playing anything"""
                wait(int(reset_time // 4), "qubit")
                align('qubit','resonator')
             #   reset_phase('resonator')
                measure("readout",'resonator',None,
                        demod.full('integW1',I,"out1"),
                        demod.full('integW2',Q,"out1"))
                save(I,I_st)
                save(Q,Q_st)

            with stream_processing():
                I_disc.buffer(len(fvec)).save('Id')
                Q_disc.buffer(len(fvec)).save('Qd')
                I_st.buffer(len(times)+2).average().save('I')
                Q_st.buffer(len(times)+2).average().save('Q')

        return prog1, times

    def rabi1d_dt(self,dt,T1, *args):
        T_min =4
        T_max = 2000 // 4
        times = np.arange(T_min, T_max, dt) * 4

        reset_time = 5 * T1
        Tpi=246
        avgs =2000
        with program() as prog1:
            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)  # Averaging
            t = declare(int)  # array of time delays
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()

            ###############
            # the sequence:
            ###############
            update_frequency("resonator", 29.5e6)
            update_frequency("qubit", 253e6)

            with for_(n, 0, n < avgs, n + 1):
                with for_(t, T_min, t < T_max, t + dt):
                    wait(int(reset_time // 4), "qubit")
                    play("saturation", "qubit", duration=t)
                    align("qubit", "resonator")
                    measure("readout" , 'resonator', None,
                            demod.full('integW1', I, "out1"),
                            demod.full('integW2', Q, "out1")
                            )
                    save(I, I_st)
                    save(Q, Q_st)

                """Play a ge pi pulse and then readout"""
                wait(int(reset_time // 4), "qubit")
                play("saturation", "qubit",duration=Tpi//4)
                align('qubit','resonator')
                measure("readout",'resonator',None,
                        demod.full('integW1',I,"out1"),
                        demod.full('integW2',Q,"out1"))
                save(I,I_st)
                save(Q,Q_st)

                """Just readout without playing anything"""
                wait(int(reset_time // 4), "qubit")
                align('qubit','resonator')
             #   reset_phase('resonator')
                measure("readout",'resonator',None,
                        demod.full('integW1',I,"out1"),
                        demod.full('integW2',Q,"out1"))
                save(I,I_st)
                save(Q,Q_st)

            with stream_processing():
                I_st.buffer(len(times)+2).average().save('I')
                Q_st.buffer(len(times)+2).average().save('Q')

        return prog1, times

    def plot2d(self,I,Q,x1,x2):
        fig = plt.figure(figsize = (10,5))
        ax = fig.add_subplot(111)
        
        mags = np.sqrt(I**2+Q**2)

        ax.pcolormesh(x1,x2,mags)
        return ax
    def T1_f(self,*args):
        dt = 5000//4
        T_min = 4
        T_max = 250000//4
        times = np.arange(T_min, T_max, dt)*4
        T1=35e3
        reset_time = 5*T1

        fmin = 385e6
        fmax = 405e6
        df = 1e6
        # reset_time = 2e3
        fvec = np.arange(fmin, fmax, df)

        Tpi=250

        avgs = 5000
        with program() as prog:

            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)      # Averaging
            i = declare(int)      # Amplitudes
            t = declare(int) #array of time delays
            f = declare(int)
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()

            ###############
            # the sequence:
            ###############

            update_frequency("resonator", 29.4e6)
            with for_(n, 0, n < avgs, n + 1):
                with for_(f,fmin,f<fmax,f+df):
                    update_frequency("qubit", f)
                    with for_(t, T_min, t < T_max, t + dt):
                        wait(int(reset_time//4), "qubit")
                        play("saturation", "qubit",duration=Tpi//4) # pi pulse with saturation
                        wait(t)
                        align("qubit", "resonator")
                      #  reset_phase('resonator')
                        measure("readout",'resonator',None,
                            demod.full('integW1',I,"out1"),
                            demod.full('integW2',Q,"out1"))
                        save(I,I_st)
                        save(Q,Q_st)


            with stream_processing():
                I_st.buffer(len(times)).buffer(len(fvec)).average().save('I')
                Q_st.buffer(len(times)).buffer(len(fvec)).average().save('Q')

        return prog,times,fvec
    
    def T1(self, params):
        dt = params["time interval (ns)"]
        T_min = params["time range (ns)"][0]
        T_max = params["time range (ns)"][1]
        times = np.arange(T_min, T_max, dt)
        T1 = params["T1 time (ns)"]
        reset_time = params["reset time (units of T1 time)"]*T1
        Tpi = params["Tpi (ns)"]//4
        avgs = params["averages"]



        with program() as prog:

            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)      # Averaging
            i = declare(int)      # Amplitudes
            t = declare(int) #array of time delays
            f = declare(int)
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()

            ###############
            # the sequence:
            ###############
            with for_(n, 0, n < avgs, n + 1):
                wait(int(reset_time//4), "qubit")
                with for_(t, T_min//4, t < T_max//4, t + dt//4):
                    wait(int(reset_time//4), "qubit")
                    # align("qubit", "qubit_multi")
                    # 
                    play("pi", "qubit", duration= Tpi)
                    # play("saturation_multi", "qubit_multi", duration=Tpi//4)
                    wait(t)
                    align("qubit", "resonator")
                    measure("readout",'resonator',None,
                        demod.full('integW1',I,"out1"),
                        demod.full('integW2',Q,"out1"))
                    save(I,I_st)
                    save(Q,Q_st)

                # wait(int(reset_time//4), "qubit")
                # align("qubit", "qubit_multi")
                # play("saturation", "qubit", duration=Tpi // 4)
                # # play("saturation_multi", "qubit_multi", duration=Tpi // 4)
                # align("qubit", "resonator")
                # measure("readout",'resonator',None, #amp=0.45, f=10.2 looks best for low power (resonator punched out) readout
                #         demod.full('integW1',I,"out1"),
                #         demod.full('integW2',Q,"out1")
                #                )
                # save(I,I_st)
                # save(Q,Q_st)

                # wait(int(reset_time//4), "qubit")
                # align("qubit", "resonator")
                # measure("readout",'resonator',None,
                #         demod.full('integW1',I,"out1"),
                #         demod.full('integW2',Q,"out1"))

                # save(I,I_st)
                # save(Q,Q_st)

            with stream_processing():
                I_st.buffer(len(times)+2).average().save('I')
                Q_st.buffer(len(times)+2).average().save('Q')

        return {"prog": prog, "times": times}
    
    def ramsey2d(self, *args):
        dt = 5
        T_min = 1
        T_max = 500 // 4
        times = np.arange(T_min, T_max, dt) * 4

        T1 = 1e3

        fmin = 400e6 - 50e6
        fmax = 400e6 - 10e6
        df = 1e6
        # reset_time = 2e3
        fvec = np.arange(fmin, fmax, df)
        reset_time = 5 * T1

        dphi_min = -0.05
        dphi_max = 0.05
        ddphi = 0.001

        avgs = 100000
        with program() as prog:
            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)  # Averaging
            i = declare(int)  # Amplitudes
            t = declare(int)  # array of time delays
            f = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            phi=declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()

            ###############
            # the sequence:
            ###############
            update_frequency("resonator", 29.6e6)
            with for_(n, 0, n < avgs, n + 1):
                with for_(f, fmin, f < fmax, f + df):
                    update_frequency("qubit", f)
                    assign(phi, 0)
                    with for_(t, T_min, t < T_max, t + dt):
                        wait(int(reset_time // 4), "qubit")
                        play("saturation", "qubit", duration=60//8)
                        wait(t)

                        frame_rotation_2pi(phi, 'qubit')
                        play("saturation", "qubit", duration=60 // 8)
                        align("qubit", "resonator")
                 #       reset_phase('resonator')
                        measure("readout" * amp(0.1), 'resonator', None,
                                demod.full('integW1', I, "out1"),
                                demod.full('integW2', Q, "out1"))
                        assign(phi, phi + 0.01 * dt)
                        save(I, I_st)
                        save(Q, Q_st)
                        reset_frame('qubit')
            with stream_processing():
                I_st.buffer(len(times)).buffer(len(fvec)).average().save('I')
                Q_st.buffer(len(times)).buffer(len(fvec)).average().save('Q')

        return prog, fvec + 3.7e9, times

    def ramsey(self, params):
        dt = params["time interval (ns)"]//4
        T_min = params["time range (ns)"][0]//4
        T_max = params["time range (ns)"][1]//4
        times = np.arange(T_min, T_max, dt)*4
        T1 = params["T1 time (ns)"]//4
        reset_time = params["reset time (units of T1 time)"]*T1
        ramsey_frequency = params["ramsey frequency (Hz)"]

        # dphi_min = -0.05
        # dphi_max = 0.05
        # ddphi = 0.001
        # dphi_vec = np.arange(dphi_min, dphi_max + ddphi/2, ddphi)
        
        dphi = dt*4*(10**-9)*ramsey_frequency
        Tpi= params["Tpi (ns)"]//4
        Tpi2=Tpi/2
        avgs = params["averages"]
        with program() as prog:

            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)      # Averaging
            i = declare(int)      # Amplitudes
            t = declare(int) #array of time delays
            f = declare(int)
            phi = declare(fixed)
            # dphi = declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()
            
            # assign(dphi, 2*np.pi*ramsey_frequency)
            # update_frequency("resonator",29.4e6)
            # update_frequency("qubit",401e6)
            # update_frequency("qubit_multi",200.5e6)
            with for_(n, 0, n < avgs, n + 1):
             #   with for_(dphi, dphi_min, dphi < dphi_max + ddphi/2, dphi + ddphi):
                # assign(phi,0)
                with for_(t, T_min, t < T_max, t + dt):
                    wait(int(reset_time), "qubit")
                    # align("qubit","qubit_multi")
                    play("pi/2","qubit") #pi/2
                    # play("saturation_multi","qubit_multi",duration=Tpi2//4) #pi/2
                    wait(t)
                    frame_rotation_2pi(dphi, 'qubit')
                    play("pi/2","qubit") #pi/2
                    # play("saturation_multi","qubit_multi",duration=Tpi2//4) #pi/2

                    align("qubit", "resonator")
                    measure("readout",'resonator',None,
                            demod.full('integW1',I,"out1"),
                            demod.full('integW2',Q,"out1"))
                    assign(phi,phi+dphi*dt)
                    save(I,I_st)
                    save(Q,Q_st)
                reset_frame('qubit')

                # wait(int(reset_time // 4), "qubit")
                # align('qubit', 'qubit_multi')

                # play("saturation", "qubit", duration=Tpi // 4)  # pi
                # play("saturation_multi", "qubit_multi", duration=Tpi // 4)  # pi
                # align("qubit", 'qubit_multi', "resonator")

                # measure("readout",'resonator',None,
                #         demod.full('integW1',I,"out1"),
                #         demod.full('integW2',Q,"out1")
                #                )
                # save(I,I_st)
                # save(Q,Q_st)

                # wait(int(reset_time //4), "qubit")
                # align("qubit", "resonator")
                # measure("readout",'resonator',None,
                #         demod.full('integW1',I,"out1"),
                #         demod.full('integW2',Q,"out1"))

                # save(I,I_st)
                # save(Q,Q_st)

            with stream_processing():
                I_st.buffer(len(times)+2).average().save('I')
                Q_st.buffer(len(times)+2).average().save('Q')
        return {'prog': prog, 'times': times}

    def ramsey_ng(self, *args):
        dt = 100//4
        T_min = 4
        T_max = 10000//4
        times = np.arange(T_min, T_max, dt)*4
        T1=15.6e3
        reset_time = 5*T1

        dphi_min = -0.05
        dphi_max = 0.05
        ddphi = 0.001
        dphi_vec = np.arange(dphi_min, dphi_max + ddphi/2, ddphi)
        Tpi=220
        avgs = 1000
        with program() as prog:

            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)      # Averaging
            i = declare(int)      # Amplitudes
            t = declare(int) #array of time delays
            f = declare(int)
            phi = declare(fixed)
            dphi = declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()

            update_frequency("resonator",29.5e6)
            update_frequency("qubit",444.5e6)
            with for_(n, 0, n < avgs, n + 1):
                with for_(t, T_min, t < T_max, t + dt):
                    wait(int(reset_time//4), "qubit")
                    align('qubit','qubit_multi')
                    play("saturation","qubit",duration=Tpi//8) #pi/2

                    wait(int(32//4))
                    frame_rotation_2pi(-0.25, 'qubit')

                    play("saturation","qubit",duration=Tpi//8) #pi/2
                    align("qubit", "resonator")
                    measure("readout",'resonator',None,
                            demod.full('integW1',I,"out1"),
                            demod.full('integW2',Q,"out1"))
                    assign(phi,phi+0.005*dt)
                    save(I,I_st)
                    save(Q,Q_st)
                    reset_frame('qubit')

                wait(int(reset_time // 4), "qubit")
                align('qubit', 'qubit_multi')

                play("saturation", "qubit", duration=Tpi // 4)  # pi/2
         #       play("saturation" * amp(0.5), "qubit_multi", duration=Tpi // 4)  # pi/2

                measure("readout",'resonator',None,
                        demod.full('integW1',I,"out1"),
                        demod.full('integW2',Q,"out1")
                               )
                save(I,I_st)
                save(Q,Q_st)

                wait(int(reset_time //4), "qubit")
                align("qubit", "resonator")
                measure("readout",'resonator',None,
                        demod.full('integW1',I,"out1"),
                        demod.full('integW2',Q,"out1"))

                save(I,I_st)
                save(Q,Q_st)

            with stream_processing():
                I_st.buffer(len(times)+2).average().save('I')
                Q_st.buffer(len(times)+2).average().save('Q')
        return prog,times

    def spin_echo(self,*args):
        dt = 40//4
        T_min = 16
        T_max = 3000//4
        times = np.arange(T_min, T_max, dt)*4
        T1=25e3
        reset_time = 5*T1
        phase=0.0
        Tpi=224

        avgs = 5000
        with program() as prog:

            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)      # Averaging
            i = declare(int)      # Amplitudes
            t = declare(int) #array of time delays
            f = declare(int)
            m = declare(int)
            phi = declare(fixed)
            dphi = declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()

            ###############
            # the sequence:
            ###############
            update_frequency("resonator",29.4e6)
            update_frequency("qubit", 401e6)
            # update_frequency("qubit_multi", 381e6)


            with for_(n, 0, n < avgs, n + 1):
                assign(phi,0)
                with for_(t, T_min, t < T_max, t + dt):
                    wait(int(reset_time//4), "qubit")

                    play("saturation", "qubit",duration=Tpi//8) #pi/2
          #          play("saturation_multi", "qubit_multi",duration=Tpi//8) #pi/2
          #
                    wait(t/2)
                    frame_rotation_2pi(phase, 'qubit')
                    play("saturation", "qubit", duration=Tpi // 4)  # pi


                    wait(t/2)

                    frame_rotation_2pi(phi,'qubit')
                    play("saturation", "qubit",duration=Tpi//8) #pi/2 with phase rotation
             #       play("saturation_multi", "qubit_multi",duration=Tpi//8) #pi/2


                    align("qubit", "resonator")
                    # reset_phase('resonator')
                    measure("readout",'resonator',None,
                            demod.full('integW1',I,"out1"),
                            demod.full('integW2',Q,"out1"))
                    assign(phi,phi+0.001*dt//4)
                    save(I,I_st)
                    save(Q,Q_st)
                    reset_frame('qubit')

                wait(int(reset_time // 4), "qubit")
                play("saturation", "qubit",duration=Tpi//4)
       #         play("saturation_multi", "qubit_multi", duration=Tpi // 4)  # pi/2
                align("qubit", "resonator")
                # reset_phase('resonator')
                measure("readout",'resonator',None,
                        demod.full('integW1',I,"out1"),
                        demod.full('integW2',Q,"out1")
                               )
                save(I,I_st)
                save(Q,Q_st)

                wait(int(reset_time //4), "qubit")
                align("qubit", "resonator")
                # reset_phase('resonator')
                measure("readout",'resonator',None,
                        demod.full('integW1',I,"out1"),
                        demod.full('integW2',Q,"out1"))

                save(I,I_st)
                save(Q,Q_st)

            with stream_processing():
                I_st.buffer(len(times)+2).average().save('I')
                Q_st.buffer(len(times)+2).average().save('Q')
            
        return prog,times

    def multiple_spin_echo(self, phase=0, *args):

        dN=1
        N_min=1
        N_max=10
        Nvec=np.arange(N_min,N_max,dN)
        avgs = 5000

        dt = 5000 // 4
        T_min = 2* 4 *(N_max-1)
        T_max = 500000// 4
        times = np.arange(T_min, T_max, dt) * 4
        T1 = 20e3
        reset_time = 5 * T1
        phase = 0.25
        Tpi = 176



        with program() as prog:
            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)  # Averaging
            N= declare(int)
            t = declare(int)  # array of time delays
            el = declare(int)
            phi = declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)
            half_t = declare(int)
            final_t = declare(int)

            invN = declare(fixed, value=(
                        1 / Nvec).tolist())  # this is a QUA array with the values calculated at compilation time

            i = declare(int)

            I_st = declare_stream()
            Q_st = declare_stream()

            ###############
            # the sequence:
            ###############
            update_frequency("resonator", 31e6)
            update_frequency("qubit", 194e6)
            update_frequency("qubit_multi", 200e6)

            with for_(n, 0, n < avgs, n + 1):
                assign(phi, 0)
                with for_(t, T_min, t < T_max, t + dt):
                    assign(i, 0)
                    with for_(N,N_min,N<N_max,N+dN):
                        wait(int(reset_time // 4), "qubit")
                        align('qubit','qubit_multi')
                        assign(half_t, Cast.mul_int_by_fixed(t, 0.5))
                        assign(final_t, Cast.mul_int_by_fixed(half_t, invN[i]))


                        play("saturation", "qubit", duration=Tpi // 8)  # pi/2
                        play("saturation_multi", "qubit_multi", duration=Tpi // 8)  # pi

                        frame_rotation_2pi(phase, 'qubit')
                        frame_rotation_2pi(phase, 'qubit_multi')

                        wait(final_t)


                        with for_(el,0,el<N,el+1):
                            play("saturation", "qubit", duration=Tpi // 4)  # pi
                            play("saturation_multi", "qubit_multi", duration=Tpi // 4)  # pi

                            wait(final_t)
                            wait(final_t)


                        align('qubit','qubit_multi')

                        play("saturation", "qubit", duration=Tpi // 4)  # pi
                        play("saturation_multi", "qubit_multi", duration=Tpi // 4)  # pi

                        reset_frame('qubit')
                        reset_frame('qubit_multi')

                        wait(final_t)

                        frame_rotation_2pi(phi, 'qubit')
                        frame_rotation_2pi(phi, 'qubit_multi')

                        align('qubit','qubit_multi')
                        play("saturation", "qubit", duration=Tpi // 8)  # pi/2 with phase rotation
                        play("saturation_multi", "qubit_multi", duration=Tpi // 8)  # pi

                        align("qubit", "resonator")
                        measure("readout", 'resonator', None,
                                demod.full('integW1', I, "out1"),
                                demod.full('integW2', Q, "out1"))
                        assign(phi, phi + 0.0001 * dt)
                        save(I, I_st)
                        save(Q, Q_st)
                        reset_frame('qubit')
                        assign(i, i + 1)

                        # reset_frame('qubit_multi')

            with stream_processing():
                I_st.buffer(len(Nvec)).buffer(len(times)).average().save('I')
                Q_st.buffer(len(Nvec)).buffer(len(times)).average().save('Q')
                # I_st.save_all('I')
        return prog, Nvec, times
        
    def ef_spec(self,pi=True,*args):
        fmin = -200e6
        fmax = -1e6
        df = 2e6
        reset_time=5*35e3
        fvec=np.arange(fmin,fmax,df)
        Tpi=250
        avgs=100
        with program() as prog:
            fr = declare(int)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n,0,n<avgs,n+1):
                with for_(fr,fmin,fr<fmax,fr+df):
                    update_frequency("resonator",29.4e6)
                    update_frequency("qubit",401e6)
                    # update_frequency("qubit_multi",448e6)
                    wait(int(reset_time//4), "qubit")# wait for the qubit to relax, several T1s
                    if pi==True:
                        align("qubit","qubit_multi")
                        play("saturation","qubit",duration=Tpi//4)
                        # play("saturation_multi","qubit_multi",duration=Tpi//4)

                    elif pi==False:
                        pass
                    update_frequency("qubit",fr)
                    play("saturation", "qubit",duration = 15e3//4)
                    align("qubit", "resonator")
                  #  reset_phase('resonator')
                    measure("readout",'resonator',None,
                            demod.full('integW1',I,"out1"),
                            demod.full('integW2',Q,"out1")
                           )
                    save(I,I_stream)
                    save(Q,Q_stream)

            with stream_processing():
                I_stream.buffer(len(fvec)).average().save("I")
                Q_stream.buffer(len(fvec)).average().save("Q")
                
        return prog,fvec+4.0e9

    def ef_spec_temp(self, pi=True, *args):

        reset_time = 16e3 * 5
        Tpi = 220
        avgs = 1e3
        with program() as prog:
            fr = declare(int)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < avgs, n + 1):
                update_frequency("resonator", 29.5e6)
                update_frequency("qubit", 441e6)
                update_frequency("qubit_multi", 448e6)
                wait(int(reset_time // 4), "qubit")  # wait for the qubit to relax, several T1s

                align("qubit", "qubit_multi")
                play("saturation", "qubit", duration=Tpi // 4)
                play("saturation_multi", "qubit_multi", duration=Tpi // 4)

                update_frequency("qubit", 2e6)
                play("saturation", "qubit", duration=15e3 // 4)
                align("qubit", "resonator")
                #  reset_phase('resonator')
                measure("readout", 'resonator', None,
                        demod.full('integW1', I, "out1"),
                        demod.full('integW2', Q, "out1")
                        )
                save(I, I_stream)
                save(Q, Q_stream)

                wait(int(reset_time // 4), "qubit")  # wait for the qubit to relax, several T1s
                update_frequency("qubit", 441e6)
                update_frequency("qubit_multi", 448e6)
                align("qubit", "qubit_multi")
                play("saturation", "qubit", duration=15e3 // 4)
                play("saturation_multi", "qubit_multi", duration=15e3 // 4)
                update_frequency("qubit", 2e6)
                play("saturation", "qubit", duration=15e3 // 4)
                align("qubit", "resonator")
                #  reset_phase('resonator')
                measure("readout", 'resonator', None,
                        demod.full('integW1', I, "out1"),
                        demod.full('integW2', Q, "out1")
                        )
                save(I, I_stream)
                save(Q, Q_stream)

                wait(int(reset_time // 4), "qubit")  # wait for the qubit to relax, several T1s
                align("qubit", "qubit_multi")
                update_frequency("qubit", 2e6)
                play("saturation", "qubit", duration=15e3 // 4)
                align("qubit", "resonator")
                #  reset_phase('resonator')
                measure("readout", 'resonator', None,
                        demod.full('integW1', I, "out1"),
                        demod.full('integW2', Q, "out1")
                        )
                save(I, I_stream)
                save(Q, Q_stream)

            with stream_processing():
                I_stream.buffer(3).average().save("I")
                Q_stream.buffer(3).average().save("Q")

        return prog, np.int(avgs)

    def fh_spec(self, pi=True, *args):
        fmin = -500e6
        fmax = -300e6
        df = 1e6
        reset_time = 7e3 * 5
        fvec = np.arange(fmin, fmax, df)
        Tpi = 375
        efTpi= 335
        avgs = 1e3
        with program() as prog:
            fr = declare(int)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < avgs, n + 1):
                with for_(fr, fmin, fr < fmax, fr + df):
                    update_frequency("resonator", 28e6)
                    update_frequency("qubit", -145e6)
                    wait(int(reset_time // 4), "qubit")  # wait for the qubit to relax, several T1s
                    play("saturation", "qubit", duration=Tpi // 4)
                    update_frequency("qubit", -302e6)
                    wait(int(reset_time // 4), "qubit")  # wait for the qubit to relax, several T1s
                    play("saturation", "qubit", duration=efTpi // 4)
                    update_frequency("qubit", fr)
                    play("saturation", "qubit", duration=1e3 // 4)
                    align("qubit", "resonator")
                    #  reset_phase('resonator')
                    measure("readout" * amp(0.2), 'resonator', None,
                            demod.full('integW1', I, "out1"),
                            demod.full('integW2', Q, "out1")
                            )
                    save(I, I_stream)
                    save(Q, Q_stream)

            with stream_processing():
                I_stream.buffer(len(fvec)).average().save("I")
                Q_stream.buffer(len(fvec)).average().save("Q")

        return prog, fvec

    def ef_spec_V(self, *args):
        fmin = -473e6
        fmax = -468e6
        df = 0.05e6
        reset_time = 10e3
        fvec = np.arange(fmin, fmax, df)

        amin = -1.0
        amax = 1.0
        da = 0.05
        avec = np.arange(amin, amax + da / 2, da)
        reset_time = 15e3

        avgs = 5e3
        with program() as prog:
            fr = declare(int)
            n = declare(int)
            a=declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < avgs, n + 1):
                with for_(fr, fmin, fr < fmax, fr + df):
                    with for_(a, amin, a < amax+da/2, a + da):

                        update_frequency("resonator", 59.8e6)
                        update_frequency("qubit", -24.5e6)
                        wait(int(reset_time // 4), "qubit")  # wait for the qubit to relax, several T1s
                      #  play("saturation", "qubit", duration=40 // 4)
                        update_frequency("qubit", fr)
                        play("saturation", "qubit", duration=1e3 // 4)
                        align("Vsource", "resonator","qubit")
                        play("CW" * amp(a), "Vsource", duration=int(5e3 // 4))  # 0.8 V for amp=1
                        reset_phase('resonator')
                        measure("readout" * amp(0.6), 'resonator', None,
                                demod.full('integW1', I, "out1"),
                                demod.full('integW2', Q, "out1")
                                )
                        save(I, I_stream)
                        save(Q, Q_stream)

            with stream_processing():
                I_stream.buffer(len(avec)).buffer(len(fvec)).average().save("I")
                Q_stream.buffer(len(avec)).buffer(len(fvec)).average().save("Q")

        return prog, fvec, avec

    def histogram(self,*args):
        T1=40e3
        reset_time = 5 * T1
        avgs = 1000
        nvec = np.arange(0, avgs, 1)

        amin = 0.1
        amax = 1.2
        da = 0.2
        avec = np.arange(amin, amax+da/2, da)

        Tpi=5e3

        with program() as prog:
            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)  #
            m = declare(int)  # Averaging
            a = declare(fixed)
            f = declare(int)

            I = declare(fixed)
            Q = declare(fixed)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()

            update_frequency('resonator',29.4e6)
            update_frequency('qubit', 400e6)

            with for_(n, 0, n < avgs, n + 1):
                with for_(a, amin, a < amax + da / 2, a + da):
                    #  """Just readout without playing anything"""
                    wait(int(reset_time // 4), "qubit")
                    align("qubit", "resonator")
                    measure("readout" * amp(a), 'resonator', None,
                            demod.full('integW1', I1, "out1"),
                            demod.full('integW2', Q1, "out1"))
                    save(I1, I_st)
                    save(Q1, Q_st)

                    #   """Play a ge pi pulse and then readout"""
                    wait(int(reset_time // 4), "qubit")
                    play("saturation", "qubit", duration=Tpi//4)
                    align("qubit", "resonator")
                    measure("readout" * amp(a), 'resonator', None,
                            demod.full('integW1', I1, "out1"),
                            demod.full('integW2', Q1, "out1"))

                    save(I1, I_st)
                    save(Q1, Q_st)

            with stream_processing():
                I_st.buffer(2 * len(avec)).buffer(avgs).save('I')
                Q_st.buffer(2 * len(avec)).buffer(avgs).save('Q')
        return prog,avec, avgs

    def histogram_f(self,*args):
        T1=35e3
        reset_time = 5 * T1
        avgs = 10000
        nvec = np.arange(0, avgs, 1)

        amin = 0.5
        amax = 1.0
        da = 0.1
        avec = np.arange(amin, amax + da / 2, da)

        fmin = 29e6
        fmax = 31e6
        df = 0.25e6
        fvec = np.arange(fmin, fmax, df)

        Tpi=250

        with program() as prog:
            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)  # Averaging
            a = declare(fixed)
            f = declare(int)

            I = declare(fixed)
            Q = declare(fixed)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()

            with for_(n, 0, n < avgs, n + 1):
                with for_(f, fmin, f < fmax, f + df):
                    update_frequency('qubit', 401e6)
                    update_frequency('resonator', f)
                    wait(int(reset_time // 4), "qubit")
                    align("qubit", "resonator")
                    measure("readout" , 'resonator', None,
                            demod.full('integW1', I1, "out1"),
                            demod.full('integW2', Q1, "out1"))
                    save(I1, I_st)
                    save(Q1, Q_st)

                    #   """Play a ge pi pulse and then readout"""
                    wait(int(reset_time // 4), "qubit")
                    play("saturation", "qubit", duration=Tpi//4)
                    align("qubit", "resonator")
                    measure("readout", 'resonator', None,
                            demod.full('integW1', I1, "out1"),
                            demod.full('integW2', Q1, "out1"))

                    save(I1, I_st)
                    save(Q1, Q_st)

            with stream_processing():
                I_st.buffer(2 * len(fvec)).buffer(len(nvec)).save('I')
                Q_st.buffer(2 * len(fvec)).buffer(len(nvec)).save('Q')
        return prog,fvec,avgs

    def histogram_fid(self,Is, Qs, numbins=100):

        a_num = len(Is[0])

        colors = ['r', 'b', 'g', 'orange']
        labels = ['g', 'e', 'f', 'h']
        titles = ['-I', '-Q']

        fig = plt.figure(figsize=(12, 16))

        ax = fig.add_subplot(421)
        x0g, y0g = np.mean(Is[0]), np.mean(Qs[0])
        x0e, y0e = np.mean(Is[1]), np.mean(Qs[1])
        phi = np.arctan((y0e - y0g) / (x0e - x0g))
        for ii in range(2):
            ax.plot(Is[ii], Qs[ii], '.', color=colors[ii], alpha=0.85)
        ax.plot(x0g, y0g, 'v', color='black')
        ax.plot(x0e, y0e, '^', color='black')

        ax.set_xlabel('I (V)')
        ax.set_ylabel('Q (V)')

        ax = fig.add_subplot(422)

        ax.plot(x0g, y0g, 'v', color='black')
        ax.plot(x0e, y0e, '^', color='black')
        ax.set_xlabel('I')
        ax.set_ylabel('Q')

        Isrot = [Is[ii] * np.cos(phi) + Qs[ii] * np.sin(phi) for ii in range(2)]
        Qsrot = [-Is[ii] * np.sin(phi) + Qs[ii] * np.cos(phi) for ii in range(2)]

        ax = fig.add_subplot(423, title='rotated')
        Is, Qs = Isrot, Qsrot

        x0g, y0g = np.mean(Is[0]), np.mean(Qs[0])
        x0e, y0e = np.mean(Is[1]), np.mean(Qs[1])
        phi = np.arctan((y0e - y0g) / (x0e - x0g))
        for ii in range(2):
            ax.plot(Is[ii], Qs[ii], '.', color=colors[ii], alpha=0.85)
        ax.plot(x0g, y0g, 'v', color='black')
        ax.plot(x0e, y0e, '^', color='black')

        ax.set_xlabel('I (V)')
        ax.set_ylabel('Q (V)')

        ax = fig.add_subplot(424)

        ax.plot(x0g, y0g, 'v', color='black')
        ax.plot(x0e, y0e, '^', color='black')
        ax.set_xlabel('I')
        ax.set_ylabel('Q')

        ax = fig.add_subplot(4, 2, 5, title='I')
        ax.hist(Is[0], bins=numbins, alpha=0.75, color=colors[0])
        ax.hist(Is[1], bins=numbins, alpha=0.75, color=colors[1])
        ax.set_xlabel('I' + '(V)')
        ax.set_ylabel('Number')
        ax.legend()

        ax = fig.add_subplot(4, 2, 6, title='Q')
        ax.hist(Qs[0], bins=numbins, alpha=0.75, color=colors[0])
        ax.hist(Qs[1], bins=numbins, alpha=0.75, color=colors[1])

        ax.set_xlabel('Q' + '(V)')
        ax.set_ylabel('Number')
        ax.legend()

        sshg, ssbinsg = np.histogram(Is[0], bins=numbins, range=[min(Is[0]), max(Is[0])])
        sshe, ssbinse = np.histogram(Is[1], bins=numbins, range=[min(Is[0]), max(Is[0])])
        fid = np.abs(((np.cumsum(sshg) - np.cumsum(sshe)) / sshg.sum())).max()

        returnfid = fid
        print("Single shot readout fidility from channel I", " = ", fid)
        print('---------------------------')

        ax = fig.add_subplot(4, 2, 7)
        ax.plot(ssbinse[:-1], np.cumsum(sshg) / sshg.sum(), color=colors[0])
        ax.plot(ssbinse[:-1], np.cumsum(sshe) / sshg.sum(), color=colors[1])
        ax.plot(ssbinse[:-1], np.abs(np.cumsum(sshe) - np.cumsum(sshg)) / sshg.sum(), color='k')

        sshg, ssbinsg = np.histogram(Qs[0], bins=numbins, range=[min(Qs[0]), max(Qs[0])])
        sshe, ssbinse = np.histogram(Qs[1], bins=numbins, range=[min(Qs[0]), max(Qs[0])])
        fid = np.abs(((np.cumsum(sshg) - np.cumsum(sshe)) / sshg.sum())).max()

        print("Single shot readout fidility from channel Q", i, " = ", fid)
        print('---------------------------')

        ax = fig.add_subplot(4, 2, 8)
        ax.plot(ssbinse[:-1], np.cumsum(sshg) / sshg.sum(), color=colors[0])
        ax.plot(ssbinse[:-1], np.cumsum(sshe) / sshg.sum(), color=colors[1])
        ax.plot(ssbinse[:-1], np.abs(np.cumsum(sshe) - np.cumsum(sshg)) / sshg.sum(), color='k')

        fig.tight_layout()
        plt.show()

        return (returnfid)

    def histogram_simp_fid(self, Is, Qs, ax, numbins=100):

        a_num = len(Is[0])

        colors = ['r', 'b', 'g', 'orange']
        labels = ['g', 'e', 'f', 'h']
        titles = ['-I', '-Q']

        x0g, y0g = np.mean(Is[0]), np.mean(Qs[0])
        x0e, y0e = np.mean(Is[1]), np.mean(Qs[1])
        phi = np.arctan((y0e - y0g) / (x0e - x0g))

        Isrot = [Is[ii] * np.cos(phi) + Qs[ii] * np.sin(phi) for ii in range(2)]
        Qsrot = [-Is[ii] * np.sin(phi) + Qs[ii] * np.cos(phi) for ii in range(2)]

        Is, Qs = Isrot, Qsrot

        x0g, y0g = np.mean(Is[0]), np.mean(Qs[0])
        x0e, y0e = np.mean(Is[1]), np.mean(Qs[1])
        phi = np.arctan((y0e - y0g) / (x0e - x0g))

        sshg, ssbinsg = np.histogram(Is[0], bins=numbins, range=[min(Is[0]), max(Is[0])])
        sshe, ssbinse = np.histogram(Is[1], bins=numbins, range=[min(Is[0]), max(Is[0])])
        fid = np.abs(((np.cumsum(sshg) - np.cumsum(sshe)) / sshg.sum())).max()

        ax.plot(ssbinse[:-1], np.cumsum(sshg) / sshg.sum(), color=colors[0])
        ax.plot(ssbinse[:-1], np.cumsum(sshe) / sshg.sum(), color=colors[1])
        ax.plot(ssbinse[:-1], np.abs(np.cumsum(sshe) - np.cumsum(sshg)) / sshg.sum(), color='k')

        return (fid)

    def ef_rabi2d(self,*args):
        dt = 20//4
        T_min = 4
        T_max = 2000//4
        times = np.arange(T_min, T_max, dt)*4

        T1=35e3

        fmin=-140e6
        fmax=-50e6
        df=1e6
        fvec=np.arange(fmin,fmax,df)
        reset_time = 5*T1
        Tpi=224
        avgs = 2000
        with program() as prog:

            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)      # Averaging
            i = declare(int)      # Amplitudes
            t = declare(int)     #array of time delays
            f = declare(int)
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()


            ###############
            # the sequence:
            ###############
            update_frequency("resonator",29.4e6)
            with for_(n, 0, n < avgs, n + 1):
                with for_(f, fmin,f<fmax, f+df):
                    with for_(t, T_min, t < T_max, t + dt):
                        wait(int(reset_time//4), "qubit")
                        update_frequency("qubit", 401e6)
                        update_frequency("qubit_multi", 395e6)
                        align("qubit","qubit_multi")
                        play("saturation", "qubit",duration = Tpi//4)
                        play("saturation_multi", "qubit_multi",duration = Tpi//4)
                        update_frequency("qubit", f)
              #          align("qubit","qubit_multi")
                        play("saturation"*amp(0.2), "qubit",duration = t)
                #        play("saturation"*amp(0.5), "qubit_multi",duration = t)
                        align("qubit", "qubit_multi","resonator")
                        measure("readout",'resonator',None,
                                demod.full('integW1',I,"out1"),
                                demod.full('integW2',Q,"out1")
                               )
                        save(I,I_st)
                        save(Q,Q_st)

            with stream_processing():
                I_st.buffer(len(times)).buffer(len(fvec)).average().save('I')
                Q_st.buffer(len(times)).buffer(len(fvec)).average().save('Q')

        return prog,fvec +4.0e9,times

    def ef_rabi1d(self,dt,*args):
        T_min = 4
        T_max = 3000//4
        times = np.arange(T_min, T_max, dt)*4

        T1=15e3
        reset_time = 5*T1
        Tpi=220
        Tpief=120

        avgs = 5000
        with program() as prog:

            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)      # Averaging
            i = declare(int)      # Amplitudes
            t = declare(int)     #array of time delays
            t = declare(int)
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()


            ###############
            # the sequence:
            ###############
            update_frequency("resonator",29.5e6)
            with for_(n, 0, n < avgs, n + 1):
                with for_(t, T_min, t < T_max, t + dt):
                    wait(int(reset_time//4), "qubit")
                    update_frequency("qubit", 441e6)
                    update_frequency("qubit_multi", 448e6)
                    align("qubit", "qubit_multi")
                    play("saturation", "qubit", duration=Tpi // 4)
                    play("saturation_multi", "qubit_multi", duration=Tpi // 4)
                    update_frequency("qubit", -128e6)
                    play("saturation", "qubit", duration=t)
                    align("qubit", "qubit_multi", "resonator")
                    measure("readout",'resonator',None,
                            demod.full('integW1',I,"out1"),
                            demod.full('integW2',Q,"out1")
                           )
                    save(I,I_st)
                    save(Q,Q_st)

                """Play a ge pi pulse and then readout"""
                wait(int(reset_time // 4), "qubit")
                update_frequency("qubit", 441e6)
                update_frequency("qubit_multi", 448e6)
                align("qubit", "qubit_multi")
                play("saturation", "qubit", duration=Tpi // 4)
                play("saturation_multi", "qubit_multi", duration=Tpi // 4)
                update_frequency("qubit", -128e6)
                play("saturation", "qubit", duration=Tpief//4)
                align("qubit", "qubit_multi", "resonator")
                measure("readout", 'resonator', None,
                        demod.full('integW1', I, "out1"),
                        demod.full('integW2', Q, "out1"))
                save(I, I_st)
                save(Q, Q_st)

                """Just readout without playing anything"""
                wait(int(reset_time // 4), "qubit")
                update_frequency("qubit", -128e6)
                play("saturation", "qubit", duration=Tpief // 4)
                align("qubit", "resonator")
                measure("readout", 'resonator', None,
                        demod.full('integW1', I, "out1"),
                        demod.full('integW2', Q, "out1"))
                save(I, I_st)
                save(Q, Q_st)

            with stream_processing():
                I_st.buffer(len(times)+2).average().save('I')
                Q_st.buffer(len(times)+2).average().save('Q')

        return prog,times

    def ef_T1(self,*args):

        dt = 200//4
        T_min = 0
        T_max = 10000//4
        times = np.arange(T_min, T_max, dt)*4
        T1=13.5e3
        reset_time = 5*T1
        Tpi=246
        efTpi=1e3
        avgs = 20000
        with program() as prog:

            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)      # Averaging
            i = declare(int)      # Amplitudes
            t = declare(int) #array of time delays
            f = declare(int)
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()

            ###############
            # the sequence:
            ###############
            update_frequency("resonator",29.5e6)
            with for_(n, 0, n < avgs, n + 1):
                with for_(t, T_min, t < T_max, t + dt):
                    wait(int(reset_time//4), "qubit")
                    update_frequency("qubit",253e6)
                    play("saturation", "qubit",duration=Tpi//4) # pi pulse with saturation
                    update_frequency("qubit",-262e6)
                    play("saturation","qubit",duration=efTpi//4)
                    wait(t)
                    update_frequency("qubit",253e6)
                    play("saturation", "qubit",duration=Tpi//4)
                    align("qubit", "resonator")
                    # reset_phase('resonator')
                    measure("readout"*amp(0.5),'resonator',None,
                        demod.full('integW1',I,"out1"),
                        demod.full('integW2',Q,"out1"))
                    save(I,I_st)
                    save(Q,Q_st)

                wait(int(reset_time//4), "qubit")
                update_frequency("qubit", 253e6)
                play("saturation", "qubit", duration=Tpi // 4)  # pi pulse with saturation
                align("qubit", "resonator")
                # reset_phase('resonator')
                measure("readout"*amp(0.5),'resonator',None, #amp=0.45, f=10.2 looks best for low power (resonator punched out) readout
                        demod.full('integW1',I,"out1"),
                        demod.full('integW2',Q,"out1")
                               )
                save(I,I_st)
                save(Q,Q_st)

                wait(int(reset_time//4), "qubit")
                align("qubit", "resonator")
                # reset_phase('resonator')
                measure("readout"*amp(0.5),'resonator',None,
                        demod.full('integW1',I,"out1"),
                        demod.full('integW2',Q,"out1"))

                save(I,I_st)
                save(Q,Q_st)

            with stream_processing():
                I_st.buffer(len(times)+2).average().save('I')
                Q_st.buffer(len(times)+2).average().save('Q')

        return prog,times

    def T1rho(self,*args):
        dt = 20//4
        T_min = 1
        T_max = 3000//4
        times = np.arange(T_min, T_max, dt)*4

        T1=25e3

        amin = 0.0
        amax = 1.0
        da = 0.2
        avec=np.arange(amin,amax+da/2,da)
        reset_time = 5*T1
        Tpi=156
        avgs = 10000
        with program() as prog:

            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)      # Averaging
            i = declare(int)      # Amplitudes
            t = declare(int)     #array of time delays
            a = declare(fixed)
            phi = declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()


            ###############
            # the sequence:
            ###############
            update_frequency("resonator",30.5e6)
            update_frequency("qubit",386e6)
            update_frequency("qubit_multi",381e6)


            with for_(n, 0, n < avgs, n + 1):
                with for_(a, amin,a<amax+da/2, a+da):
                    with for_(t, T_min, t < T_max, t + dt):
                        wait(int(reset_time//4), "qubit")

                        play("saturation","qubit",duration=Tpi//8)
                  #      play("saturation_multi","qubit_multi",duration=Tpi//8)

                        reset_frame('qubit')
#                        frame_rotation_2pi(-0.08,'qubit')
                        frame_rotation_2pi(0.25-0.05, 'qubit')
                        play("saturation"*amp(a), "qubit",duration = t)
                      #  play("saturation_multi"*amp(a), "qubit_multi",duration = t)

                        reset_frame('qubit')
                        reset_frame('qubit_multi')


                        frame_rotation_2pi(phi, 'qubit')
                    #    frame_rotation_2pi(phi, 'qubit_multi')

                        play("saturation", "qubit", duration=Tpi// 8)
                   #     play("saturation", "qubit_multi", duration=Tpi// 8)

                        align("qubit", "resonator")
                        measure("readout", 'resonator', None,
                                demod.full('integW1', I, "out1"),
                                demod.full('integW2', Q, "out1"))
                        assign(phi, phi + 0.01 * dt )
                        save(I, I_st)
                        save(Q, Q_st)
                        reset_frame('qubit')

                        ## DO a T1 experiment concurrently
                        wait(int(reset_time // 4), "qubit")
                        play("saturation", "qubit", duration=Tpi // 4)  # pi pulse with saturation
                      #  play("saturation", "qubit_multi", duration=Tpi// 4)
                        wait(t)
                        align("qubit", "resonator")
                        measure("readout" , 'resonator', None,
                                demod.full('integW1', I, "out1"),
                                demod.full('integW2', Q, "out1"))
                        save(I, I_st)
                        save(Q, Q_st)

            with stream_processing():
                I_st.buffer(2*len(times)).buffer(len(avec)).average().save('I')
                Q_st.buffer(2*len(times)).buffer(len(avec)).average().save('Q')

        return prog,avec,times

    def T1rho_test(self,*args):
        dt = 40//4
        T_min = 1
        T_max = 5000//4
        times = np.arange(T_min, T_max, dt)*4

        T1=8e3

        amin = -0.15
        amax = -0.05
        da = 0.005
        avec=np.arange(amin,amax+da/2,da)
        reset_time = 5*T1
        Tpi=375
        avgs = 1000
        with program() as prog:

            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)      # Averaging
            i = declare(int)      # Amplitudes
            t = declare(int)     #array of time delays
            a = declare(fixed)
            phi = declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()


            ###############
            # the sequence:
            ###############
            update_frequency("resonator",30.3e6)
            update_frequency("qubit",-144.5e6)

            with for_(n, 0, n < avgs, n + 1):
                with for_(a, amin,a<amax+da/2, a+da):
                    with for_(t, T_min, t < T_max, t + dt):
                        wait(int(reset_time//4), "qubit")

                        play("saturation","qubit",duration=Tpi//8)
                        reset_frame('qubit')

                        frame_rotation_2pi(0.25+a, 'qubit')
                        play("saturation"*amp(0.5), "qubit",duration = t)
                        reset_frame('qubit')

                        frame_rotation_2pi(phi, 'qubit')
                     #   play("saturation", "qubit", duration=Tpi// 8)
                        align("qubit", "resonator")
                        measure("readout" * amp(0.1), 'resonator', None,
                                demod.full('integW1', I, "out1"),
                                demod.full('integW2', Q, "out1"))
                        assign(phi, phi + 0.01 * dt )
                        save(I, I_st)
                        save(Q, Q_st)
                        reset_frame('qubit')

                        ## DO a T1 experiment concurrently
                        wait(int(reset_time // 4), "qubit")
                        play("saturation", "qubit", duration=Tpi // 4)  # pi pulse with saturation
                        wait(t)
                        align("qubit", "resonator")
                        measure("readout" * amp(0.2), 'resonator', None,
                                demod.full('integW1', I, "out1"),
                                demod.full('integW2', Q, "out1"))
                        save(I, I_st)
                        save(Q, Q_st)

            with stream_processing():
                I_st.buffer(2*len(times)).buffer(len(avec)).average().save('I')
                Q_st.buffer(2*len(times)).buffer(len(avec)).average().save('Q')

        return prog,avec,times


    def ef_spec_diff(self, pi=True, *args):
        fmin = -10e6
        fmax = 10e6
        df = 2e6
        reset_time = 5*15e3
        fvec = np.arange(fmin, fmax, df)
        Tpi=220
        avgs = 1e3
        with program() as prog:
            fr = declare(int)
            n = declare(int)
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < avgs, n + 1):
                with for_(fr, fmin, fr < fmax, fr + df):
                    update_frequency("resonator", 29.5e6)
                    update_frequency("qubit", 441e6)
                    update_frequency("qubit_multi", 448e6)
                    wait(int(reset_time // 4), "qubit")  # wait for the qubit to relax, several T1s
                    align("qubit", "qubit_multi")
                    play("saturation", "qubit", duration=Tpi // 4)
                    play("saturation_multi", "qubit_multi", duration=Tpi // 4)
                    update_frequency("qubit", fr)
                    play("saturation", "qubit", duration=15e3 // 4)
                    align("qubit", "resonator")
                    #  reset_phase('resonator')
                    measure("readout", 'resonator', None,
                            demod.full('integW1', I1, "out1"),
                            demod.full('integW2', Q1, "out1")
                            )
                    save(I1, I_stream)
                    save(Q1, Q_stream)

                    wait(int(reset_time // 4), "qubit")  # wait for the qubit to relax, several T1s
                    update_frequency("qubit", fr)
                    play("saturation", "qubit", duration=15e3 // 4)
                    align("qubit", "resonator")
                    #  reset_phase('resonator')
                    measure("readout", 'resonator', None,
                            demod.full('integW1', I2, "out1"),
                            demod.full('integW2', Q2, "out1")
                            )

                    save(I2, I_stream)
                    save(Q2, Q_stream)

            with stream_processing():
                I_stream.buffer(2*len(fvec)).average().save("I")
                Q_stream.buffer(2*len(fvec)).average().save("Q")

        return prog, fvec

    def IQ_blobs(self,*args):
        T1=16e3
        reset_time = T1 * 5
        fr = 373.8e6
        avgs=200
        nvec=np.arange(0,2*avgs,1)
        Tpi=220
        reset_time=T1*5

        with program() as prog:
            fr = declare(int)
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            update_frequency("resonator", 29.5e6)

            with for_(n, 0, n < avgs, n + 1):
                wait(int(reset_time // 4), "qubit")  # wait for the qubit to relax, several T1s
                update_frequency("qubit", 441e6)
                update_frequency("qubit_multi", 448e6)
                align("qubit", "qubit_multi")
                play("saturation", "qubit", duration=Tpi // 4)
                play("saturation_multi", "qubit_multi", duration=Tpi // 4)
                align("qubit", "qubit_multi", "resonator")
                measure("readout", 'resonator', None,
                        demod.full('integW1', I, "out1"),
                        demod.full('integW2', Q, "out1")
                        )
                save(I, I_stream)
                save(Q, Q_stream)

                wait(int(reset_time // 4), "qubit")  # wait for the qubit to relax, several T1s
                align("qubit", "resonator")
                measure("readout", 'resonator', None,
                        demod.full('integW1', I, "out1"),
                        demod.full('integW2', Q, "out1")
                        )
                save(I, I_stream)
                save(Q, Q_stream)

            with stream_processing():
                I_stream.buffer(2*avgs).save("I")
                Q_stream.buffer(2*avgs).save("Q")

        return prog,nvec


    def IQ_blobs_f(self,*args):
        T1=1e3
        reset_time = T1 * 5
        fr = 373.8e6
        avgs=2000
        nvec=np.arange(0,2*avgs,1)

        fmin=fr-10e6
        fmax=fr+10e6
        df=0.1e6
        fvec=np.arange(fmin,fmax,df)

        with program() as prog:
            fr = declare(int)
            n = declare(int)
            f = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            update_frequency("resonator", 59.8e6)
            update_frequency("qubit", fr)

            with for_(f,fmin,f<fmax,f+df):
                with for_(n, 0, n < avgs, n + 1):

                    wait(int(reset_time // 4), "qubit")  # wait for the qubit to relax, several T1s
                    play('saturation', 'qubit', duration=52// 4)
                    align("qubit", "resonator")
                   # reset_phase('resonator')
                    measure("readout"*amp(0.01), 'resonator', None,
                            demod.full('integW1', I, "out1"),
                            demod.full('integW2', Q, "out1")
                            )
                    save(I, I_stream)
                    save(Q, Q_stream)
        #        with for_(n, 0, n < avgs, n + 1):
                    wait(int(reset_time*10 // 4), "qubit")  # wait for the qubit to relax, several T1s
                    align("qubit", "resonator")
                #    reset_phase('resonator')
                    measure("readout"*amp(0.01), 'resonator', None,
                            demod.full('integW1', I, "out1"),
                            demod.full('integW2', Q, "out1")
                            )
                    save(I, I_stream)
                    save(Q, Q_stream)

            with stream_processing():
                I_stream.buffer(2*avgs).save("I")
                Q_stream.buffer(2*avgs).save("Q")

        return prog,fvec,nvec

    def IQ_blobs_wRabi(self, *args):

        dt = 60//4
        T_min = 1
        T_max = 1000 // 4
        times = np.arange(T_min, T_max, dt) * 4

        T1 = 2e3
        reset_time = T1 * 5
        fr = 373.8e6
        avgs = 10000
        nvec = np.arange(0, 2 * avgs, 1)
        a = 0.1
        with program() as prog:
            fr = declare(int)
            n = declare(int)
            t=declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            update_frequency("resonator", 28.6e6)
            update_frequency("qubit", 374.5e6)

            with for_(n, 0, n < avgs, n + 1):
                with for_(t, T_min, t < T_max, t + dt):
                    wait(int(reset_time // 4), "qubit")
                    play("saturation", "qubit", duration=t)
                    align("qubit", "resonator")
               #     reset_phase('resonator')
                    measure("readout" * amp(a), 'resonator', None,
                            demod.full('integW1', I, "out1"),
                            demod.full('integW2', Q, "out1")
                            )
                    save(I, I_stream)
                    save(Q, Q_stream)
                """Play a ge pi pulse and then readout"""
                wait(int(reset_time // 4), "qubit")
                play("saturation", "qubit",duration=60//4)
                align('qubit','resonator')
              #  reset_phase('resonator')
                measure("readout"*amp(a),'resonator',None,
                        demod.full('integW1',I,"out1"),
                        demod.full('integW2',Q,"out1"))
                save(I,I_stream)
                save(Q,Q_stream)

                """Just readout without playing anything"""
                wait(int(reset_time // 4), "qubit")
                align('qubit','resonator')
               # reset_phase('resonator')
                measure("readout"*amp(a),'resonator',None,
                        demod.full('integW1',I,"out1"),
                        demod.full('integW2',Q,"out1"))
                save(I,I_stream)
                save(Q,Q_stream)


            with stream_processing():
                I_stream.buffer(len(times)+2).buffer(avgs).save("I")
                Q_stream.buffer(len(times)+2).buffer(avgs).save("Q")

        return prog, nvec,times

    def qubit_temp(self, *args):
        dt = 20//4
        T_min = 4
        T_max = 2000//4
        times = np.arange(T_min, T_max, dt) * 4

        T1 = 35e3
        reset_time = 5 * T1
        Tpi=250
        avgs = 10000
        with program() as prog:
            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)  # Averaging
            i = declare(int)  # Amplitudes
            t = declare(int)  # array of time delays
            f = declare(int)
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()

            ###############
            # the sequence:
            ###############
            update_frequency("resonator", 29.4e6)

            with for_(n, 0, n < avgs, n + 1):
                with for_(t, T_min, t < T_max, t + dt):
                    # reference
                    wait(int(reset_time // 4), "qubit")
                    update_frequency("qubit", 401e6)
                    # update_frequency("qubit_multi", 448e6)
                    align("qubit", "qubit_multi")
                    play("saturation", "qubit", duration=Tpi // 4)
                    # play("saturation_multi", "qubit_multi", duration=Tpi // 4)
                    update_frequency("qubit", -75e6)
                    update_frequency("qubit_multi", -150e6)

                    play("saturation", "qubit", duration=t
                     #    ,chirp=(-30,'GHz/sec')
                         )
                    play("saturation_multi", "qubit_multi", duration=t
                         #    ,chirp=(-30,'GHz/sec')
                         )
                    align("qubit", "qubit_multi", "resonator")
                    measure("readout", 'resonator', None,
                            demod.full('integW1', I, "out1"),
                            demod.full('integW2', Q, "out1")
                            )
                    save(I, I_st)
                    save(Q, Q_st)


                    # signal
                    wait(int(reset_time // 4), "qubit")
                    update_frequency("qubit", -75e6)
                    update_frequency("qubit_multi", -150e6)

                    play("saturation", "qubit", duration=t
                     #    ,chirp=(-30,'GHz/sec')
                         )
                    play("saturation_multi", "qubit_multi", duration=t)
                         #    ,chirp=(-30
                    align("qubit","qubit_multi", "resonator")
                    measure("readout", 'resonator', None,
                            demod.full('integW1', I, "out1"),
                            demod.full('integW2', Q, "out1")
                            )
                    save(I, I_st)
                    save(Q, Q_st)

            with stream_processing():
                I_st.buffer(2 * len(times)).average().save('I')
                Q_st.buffer(2 * len(times)).average().save('Q')
        return prog, times, avgs

    def pumped_qubit_temp(self, *args):
        dt = 335//4//8
        T_min = 1
        T_max = 3000//4
        times = np.arange(T_min, T_max, dt) * 4

        T1 = 8e3
        reset_time = 5 * T1
        Tpi=375
        avgs = 20000

        tpulsemin=8//4
        tpulsemax=40//4
        dtpulse=4//4
        tpulsevec=np.arange(tpulsemin,tpulsemax,dtpulse) *4
        with program() as prog:
            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)  # Averaging
            i = declare(int)  # Amplitudes
            t = declare(int)  # array of time delays
            f = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            tpulse = declare(int)

            I_st = declare_stream()
            Q_st = declare_stream()

            ###############
            # the sequence:
            ###############
            update_frequency("resonator", 29e6)

            with for_(n, 0, n < avgs, n + 1):
                with for_(tpulse,tpulsemin,tpulse<tpulsemax,tpulse+dtpulse):
                    with for_(t, T_min, t < T_max, t + dt):
                        # reference
                        wait(int(reset_time // 4), "qubit")
                        update_frequency("qubit", -145e6)
                        play("saturation", "qubit", duration=Tpi // 4)
                        update_frequency("qubit", -302e6)
                        play("saturation", "qubit", duration=t)
                        align("qubit", "resonator")
                        measure("readout" * amp(0.2), 'resonator', None,
                                demod.full('integW1', I, "out1"),
                                demod.full('integW2', Q, "out1")
                                )
                        save(I, I_st)
                        save(Q, Q_st)

                        # signal
                        wait(int(reset_time // 4), "qubit")
                        update_frequency("qubit", -145e6)
                        play("saturation", "qubit", duration=tpulse)
                        update_frequency("qubit", -302e6)
                        play("saturation", "qubit", duration=t)
                        align("qubit", "resonator")
                        measure("readout" * amp(0.2), 'resonator', None,
                                demod.full('integW1', I, "out1"),
                                demod.full('integW2', Q, "out1")
                                )
                        save(I, I_st)
                        save(Q, Q_st)

            with stream_processing():
                I_st.buffer(2*len(times)).buffer(len(tpulsevec)).average().save('I')
                Q_st.buffer(2*len(times)).buffer(len(tpulsevec)).average().save('Q')
        return prog, times,tpulsevec,avgs

    def SNR(self, avgs, *args):
        T1 = 2e3
        reset_time = 5 * T1

        Tpi = 60
        with program() as prog:
            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)  # Averaging
            i = declare(int)  # Amplitudes
            t = declare(int)  # array of time delays
            f = declare(int)
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()

            ###############
            # the sequence:
            ###############
            update_frequency("resonator", 28.6e6)
            update_frequency("qubit", 374.5e6)
            with for_(n, 0, n < avgs, n + 1):

                wait(int(reset_time // 4), "qubit")
                play("saturation", "qubit", duration=Tpi // 4)  # pi pulse with saturation
                align("qubit", "resonator")
                measure("readout" * amp(1), 'resonator', None,
                        demod.full('integW1', I, "out1"),
                        demod.full('integW2', Q, "out1")
                        )
                save(I, I_st)
                save(Q, Q_st)

                wait(int(reset_time // 4), "qubit")
                align("qubit", "resonator")
                measure("readout" * amp(1), 'resonator', None,
                        demod.full('integW1', I, "out1"),
                        demod.full('integW2', Q, "out1"))

                save(I, I_st)
                save(Q, Q_st)

            with stream_processing():
                I_st.buffer(2*avgs).save('I')
                Q_st.buffer(2*avgs).save('Q')

        return prog, avgs
    def ef_ramsey(self, *args):
        dt = 12//4
        T_min = 0
        T_max = 1000//4
        times = np.arange(T_min, T_max, dt)*4
        T1=15.6e3
        reset_time = 5*T1

        dphi_min = -0.05
        dphi_max = 0.05
        ddphi = 0.001
        dphi_vec = np.arange(dphi_min, dphi_max + ddphi/2, ddphi)
        Tpi=220
        Tefpi=200
        avgs = 2000
        with program() as prog:

            ##############################
            # declare real-time variables:
            ##############################

            n = declare(int)      # Averaging
            i = declare(int)      # Amplitudes
            t = declare(int) #array of time delays
            f = declare(int)
            phi = declare(fixed)
            dphi = declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)

            I_st = declare_stream()
            Q_st = declare_stream()

            update_frequency("resonator",29.5e6)
            update_frequency("qubit",441e6)
            update_frequency("qubit_multi",448e6)
            with for_(n, 0, n < avgs, n + 1):
             #   with for_(dphi, dphi_min, dphi < dphi_max + ddphi/2, dphi + ddphi):
          #      assign(phi,0)
                with for_(t, T_min, t < T_max, t + dt):
                    wait(int(reset_time//4), "qubit")
                    align('qubit','qubit_multi')
                    play("saturation","qubit",duration=Tpi//4) #pi/2
                    play("saturation_multi","qubit_multi",duration=Tpi//4) #pi/2
                    update_frequency("qubit", -128e6)

                    play("saturation"*amp(0.2),"qubit",duration=Tefpi//8) #pi/2

                    wait(t)
                    frame_rotation_2pi(phi, 'qubit')

                    play("saturation"*amp(0.2),"qubit",duration=Tefpi//8) #pi/2
                    align("qubit", "resonator")
                    measure("readout",'resonator',None,
                            demod.full('integW1',I,"out1"),
                            demod.full('integW2',Q,"out1"))
                  #  assign(phi,phi+0.01*dt)
                    save(I,I_st)
                    save(Q,Q_st)
                    reset_frame('qubit')

                wait(int(reset_time // 4), "qubit")
            align('qubit', 'qubit_multi')
            play("saturation", "qubit", duration=Tpi // 4)  # pi/2
            play("saturation_multi", "qubit_multi", duration=Tpi // 4)  # pi/2
            update_frequency("qubit", -128e6)
            play("saturation"*amp(0.2), "qubit", duration=Tefpi // 4)  # pi/2
     #       play("saturation" * amp(0.5), "qubit_multi", duration=Tpi // 4)  # pi/2

            measure("readout",'resonator',None,
                    demod.full('integW1',I,"out1"),
                    demod.full('integW2',Q,"out1")
                           )
            save(I,I_st)
            save(Q,Q_st)

            wait(int(reset_time //4), "qubit")
            play("saturation", "qubit", duration=Tpi // 4)  # pi/2
            play("saturation", "qubit_multi", duration=Tpi // 4)  # pi/2
            align("qubit", "resonator")
            measure("readout",'resonator',None,
                    demod.full('integW1',I,"out1"),
                    demod.full('integW2',Q,"out1"))

            save(I,I_st)
            save(Q,Q_st)

            with stream_processing():
                I_st.buffer(len(times)+2).average().save('I')
                Q_st.buffer(len(times)+2).average().save('Q')
        return prog,times
