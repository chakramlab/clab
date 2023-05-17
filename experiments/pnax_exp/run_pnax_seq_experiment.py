__author__ = 'Alex Ma'

from pnaxpulseexperiment import PnaxPulseExperiment
from slab import *
from numpy import *
import os
import json
import collections

def get_data_filename(prefix):
    datapath = os.getcwd() + '\data'
    return  os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))

def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = update_dict(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d

def run_pnax_experiment_and_append(expt_name, data_file, config_file, vary_dict={}, vary_pt=0):

    datapath = os.getcwd() + '\data'
    # config_file = os.path.join(datapath, config_file)
    config_file = 'D:\\_data\\20230222_T162_blueforssnowy_coffin cavity test\\config_pnax_rabi.json'
    with open(config_file, 'r') as fid:
            cfg_str = fid.read()
    cfg_dict = json.loads(cfg_str)

    data_file_temp = os.path.join(datapath, 'data_pnax_temp.h5')
    # important - so that data don't pile up..
    if os.path.isfile(data_file_temp):
        os.remove(data_file_temp)
    with SlabFile(data_file) as f:
        f.attrs['config'] = json.dumps(cfg_dict)

    cfg_dict_temp = update_dict(cfg_dict, vary_dict)
    with open('config_pnax_temp.json', 'w') as fp:
        json.dump(cfg_dict_temp, fp)

    PnaxPulseExperiment(expt_name, config_file='..\\config_pnax_temp.json', data_file= data_file_temp)

    with SlabFile(data_file) as f:
        with SlabFile(data_file_temp, 'a') as ftemp:
            for key in list(ftemp.keys()):
                if key != 'expt_2d':
                    f.append_line(key,array(ftemp[key]).flatten())
        f.append_pt('vary_pts', vary_pt)

# -------------------------------------------
# here on are user defined sweep experiments:
# -------------------------------------------
def res_power_sweep(config_file = "..\\config_pnax.json"):

    prefix = 'Res_PowerSweep'
    expt_name ='vacuum_rabi_focus_scan'
    data_file = get_data_filename(prefix)

    vary_pts = np.arange(-50, 20, 5)
    for ii, pt in enumerate(vary_pts):

        print("taking seq data with vary_pt =", pt)
        vary_dict = {"readout": {"power": float(pt)},
                     }
        run_pnax_experiment_and_append(expt_name, data_file, config_file, vary_dict, pt)

def vacuum_rabi_sweep(data_path, filename, config_file = "..\\config_pnax.json"):

    prefix = 'VacuumRabi_PowerSweep'
    expt_name ='vacuum_rabi'
    data_file = data_path + filename
    print(data_file)

    vary_pts = linspace(-40, 0, 5)
    for ii, pt in enumerate(vary_pts):

        print("taking seq data with vary_pt =", pt)
        vary_dict = {"readout": {"power": pt},
                     "vacuum_rabi": {"pi_pulse": True}
                     }
        run_pnax_experiment_and_append(expt_name, data_file, config_file, vary_dict, pt)

def pulse_probe_sweep(config_file = "..\\config_pnax.json"):

    prefix = 'PulseProbe_FluxSweep'
    expt_name ='pulse_probe'
    data_file = get_data_filename(prefix)
    print(data_file)

    vary_pts = linspace(0.0e-3,-1.000e-3,15)

    # read_freqs = [4.9096666666666673, 4.9096666666666673, 4.9108333333333327, 4.9108333333333327, 4.9108333333333327, 4.9114166666666668, 4.9114166666666668, 4.9102499999999996, 4.9102499999999996, 4.9102499999999996, 4.9090833333333332, 4.9090833333333332, 4.9096666666666673, 4.9085000000000001, 4.9085000000000001, 4.9085000000000001, 4.9090833333333332, 4.9090833333333332, 4.9085000000000001, 4.9085000000000001]






    for ii, pt in enumerate(vary_pts):
        print("taking seq data with vary_pt =", pt)
        vary_dict = {"flux": {"current": pt}}
        run_pnax_experiment_and_append(expt_name, data_file, config_file, vary_dict, pt)

def pulse_probe_powsweep(config_file = "..\\config_pnax.json"):

    prefix = 'PulseProbe_PowerSweep'
    expt_name ='pulse_probe'
    data_file = get_data_filename(prefix)
    print(data_file)

    vary_pts = linspace(-20.0,18.0,30)

    # read_freqs = [4.9096666666666673, 4.9096666666666673, 4.9108333333333327, 4.9108333333333327, 4.9108333333333327, 4.9114166666666668, 4.9114166666666668, 4.9102499999999996, 4.9102499999999996, 4.9102499999999996, 4.9090833333333332, 4.9090833333333332, 4.9096666666666673, 4.9085000000000001, 4.9085000000000001, 4.9085000000000001, 4.9090833333333332, 4.9090833333333332, 4.9085000000000001, 4.9085000000000001]

    for ii, pt in enumerate(vary_pts):
        print("taking seq data with vary_pt =", pt)
        vary_dict = {"drive": {"power": pt}}
        run_pnax_experiment_and_append(expt_name, data_file, config_file, vary_dict, pt)

def raman_powsweep(config_file = "..\\config_pnax.json"):

    prefix = 'raman_PowerSweep'
    expt_name ='raman'
    data_file = get_data_filename(prefix)
    print(data_file)

    vary_pts = linspace(-5.0,5.0,21)

    for ii, pt in enumerate(vary_pts):
        print("taking seq data with vary_pt =", pt)
        vary_dict = {"raman": {"pump_pow": pt}}
        run_pnax_experiment_and_append(expt_name, data_file, config_file, vary_dict, pt)

def raman_freqsweep(config_file = "..\\config_pnax.json"):

    prefix = 'raman_PumpFreqSweep'
    expt_name ='raman'
    data_file = get_data_filename(prefix)
    print(data_file)

    vary_pts = linspace(4.66e9,4.78e9,60)

    for ii, pt in enumerate(vary_pts):
        print("taking seq data with vary_pt =", pt)
        vary_dict = {"raman": {"pump_freq": pt}}
        run_pnax_experiment_and_append(expt_name, data_file, config_file, vary_dict, pt)

def T1_sweep(config_file = "..\\config_pnax.json"):

    prefix = 'T1_FluxSweep'
    expt_name ='t1'
    data_file = get_data_filename(prefix)
    print(data_file)

    res_freqs = [5.0644999999999998, 5.0625, 5.0594999999999999, 5.0570000000000004, 5.0555000000000003]

    read_freqs = [4.9131666666666671, 4.9131666666666671, 4.9125833333333331, 4.9096666666666673, 4.9102499999999996]
    flux_pts = [0.001, 0.0, -0.001, -0.002, -0.0030000000000000001]

    for ii, pt in enumerate(flux_pts):
        print("Taking T1 data at %.4f mA =" %pt*1000)
        print("Read Frequency is now %.5f GHz" %read_freqs[ii])

        vary_dict = {"flux": {"current": pt},
                     "qubit":{"freq": res_freqs[ii]*1e9},
                     "readout": {"freq": read_freqs[ii]*1e9}
                     }
        for k in range(3):
            run_pnax_experiment_and_append(expt_name, data_file, config_file, vary_dict, pt)

def rabi_freq_sweep(config_file = "..\\config_pnax.json"):

    prefix = 'Rabi_FreqSweep'
    expt_name = 'rabi'
    data_file = get_data_filename(prefix)
    print(data_file)

    vary_pts = linspace(5.6374e9, 5.6414e9, 30)
    for ii, pt in enumerate(vary_pts):
        print("taking seq data with vary_pt =", pt)
        vary_dict = {"qubit": {"freq": pt}}
        run_pnax_experiment_and_append(expt_name, data_file, config_file, vary_dict, pt)

def rabi_power_sweep(config_file = "..\\config_pnax.json"):

    prefix = 'Rabi_PowerSweep'
    expt_name = 'rabi'
    data_file = get_data_filename(prefix)
    print(data_file)

    vary_pts = linspace(5,15,8)
    for ii, pt in enumerate(vary_pts):
        print("taking seq data with vary_pt =", pt)
        vary_dict = {"drive": {"power": pt}}
        run_pnax_experiment_and_append(expt_name, data_file, config_file, vary_dict, pt)


# config_file = '..\\config_pnax_vacuum_rabi.json'
# res_power_sweep(config_file)
    # rabi_power_sweep(config_file)
