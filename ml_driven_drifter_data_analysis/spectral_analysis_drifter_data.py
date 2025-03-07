#%%


import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean 
import pycwt as wavelet
from scipy.interpolate import interp1d
import pandas as pd

#%%
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"

datatree=xr.open_datatree(f'{DATA_DIR}/interim/processed_drifter_data.nc')
# %%

def lower_temporal_resolution(datatree, period='3h'):
    d={}

    for node in datatree.leaves:
        ds=node.ds
        ds_resampled=ds.resample(time=period).nearest()
        d[node.name]=ds_resampled
    
    shortest_length=min([len(d[key]['time']) for key in d.keys()])
    for key in d.keys():
        d[key]=d[key].isel(time=slice(0, shortest_length))
    
    return xr.DataTree.from_dict(d)

def Wavelet_transform(data_norm, dt):
    mother=wavelet.Morlet(6)
    s0 =  dt/2  # Starting scale, in this case 2 * 0.25 years = 6 months
    dj = 1 / 12  # Twelve sub-octaves per octaves
    J = 20 / dj  # Dynamically adjust to frequency range
    alpha, _, _ = wavelet.ar1(data_norm)  # Lag-1 autocorrelation for red noise

    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(data_norm, dt, dj, s0, J,
                                                        mother)
    # Normalized wavelet and Fourier power spectra
    power = (np.abs(wave)) ** 2
    period = 1/freqs
    return period, coi, power, freqs,

def power_spectrum(datatree, vars):

    df=[]

    for node in datatree.leaves:
        ds=node.ds
        deltat=ds['time'].diff("time").dt.total_seconds().mean().values
        normalised=(ds[vars].values-np.mean(ds[vars].values))/np.std(ds[vars].values)
        period, coi, power, freqs, =Wavelet_transform(normalised, deltat)
        fft_power = np.abs(np.fft.fft(ds[vars].values))**2
        fft_freqs = np.fft.fftfreq(len(ds[vars].values), d=deltat)
        # Append the results to the dataframe
        
        node_data = {
        'drifter': node.name, 'coi': coi,
        'period': period,
        'freq': freqs,
        'power': power,
        'fft_power': fft_power,  # Add FFT power calculation
        'fft_freqs' : fft_freqs,
        'time': ds['time'].values
        }

        # Append the current node data to the list
        df.append(node_data)
    
    return pd.DataFrame(df)

def plot_single_Wavelet_period(pw, dt, drifter_id):
    
    power=pw['power'].values[drifter_id]
    period=pw['period'].values[drifter_id]
    N=len(pw['time'].values[drifter_id])
    t=np.arange(0, N)*dt
    coi=pw['coi'].values[drifter_id]

    plt.figure(figsize=(10, 8))

    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 20]
    # Plot the wavelet power spectrum as a function of frequency
    plt.contourf(t/24/3600, period/3600, np.log2(power), np.log2(levels),
                extend='both', cmap=plt.cm.viridis)


    # Update the extent for frequency
    extent = [t.min()/24/3600, t.max()/24/3600, np.min(period)/3600, np.max(period)/3600]
    # Plot cone of influence
    plt.fill(np.concatenate([t/24/3600, (t[-1:] + dt)/24/3600, (t[-1:] + dt)/24/3600, (t[:1] - dt)/24/3600, (t[:1] - dt)/24/3600]),
            np.concatenate([coi/3600, [1e-9/3600], period[-1:]/3600, period[-1:]/3600, [1e-9/3600]]),
            'k', alpha=0.3, hatch='x')

    plt.plot(t/24/3600, np.full(len(t), 12.4206), label='M2', linestyle='dashdot', color='red')
    plt.plot(t/24/3600, np.full(len(t), 12.0), label='S2', linestyle='dashdot', color='blue')
    plt.plot(t/24/3600, np.full(len(t), 6.2103), label='M4', linestyle='dashdot', color='darkorange')
    plt.plot(t/24/3600, np.full(len(t), 14.79), label='Inertial period', linestyle='dashdot', color='black')

    plt.legend()
    plt.ylim(period.min()/3600, 24)
    plt.xlim(t[0]/3600/24, t[-1]/3600/24)

    plt.ylabel('Period (hrs)')
    plt.xlabel('Time since release [days]')

def plot_Fourier_Wavelet(pw, dt, name_fig='Wavelet_Fourier', output_path=None):

    t= (pw['time'][0] - pw['time'][0][0]) / np.timedelta64(1, 's')
    power=pw['power'].mean(axis=0)
    period=pw['period'].mean(axis=0)
    freqs=pw['freq'].mean(axis=0)
    coi=pw['coi'].values.mean(axis=0)
    fft_power=pw['fft_power'].mean(axis=0)
    fft_freqs=pw['fft_freqs'].mean(axis=0)*3600*24
  
    # Create a figure with two subplots, adjusting width ratio to make left subplot narrower\n",
    fig, ax = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [1, 4]})
    
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16,  24, 32, 40]
    c=ax[1].contourf(t, freqs*3600*24, np.log2(power), np.log2(levels), shading='gouraud',
                    extend='both', cmap=cmocean.cm.balance)
    plt.colorbar(c, ax=ax[1], label='PSD [$log_{10}m^2 s^{-2} cpd^{-1}]$')
    
    extent = [t.min(), t.max(), np.min(freqs)*3600*24, np.max(freqs)*3600*24]
    
    # Plot cone of influence
    plt.fill(np.concatenate([t/24/3600, (t[-1:] + dt)/24/3600, (t[-1:] + dt)/24/3600, (t[:1] - dt)/24/3600, (t[:1] - dt)/24/3600]),
            np.concatenate([1/(coi*24*3600), [1e-9*24*3600], freqs[-1:]*3600*24, freqs[-1:]*3600*24, [1e-9*24*3600]]),
            'k', alpha=0.3, hatch='x')
    
    ax[1].plot(t , np.full(len(t),  1.9323), label='M2', linestyle='dashdot', color='cyan', linewidth=2)#d62728\n",
    ax[1].plot(t , np.full(len(t), 2.0), label='S2', linestyle='dashdot', color='#39FF14', linewidth=2)#ff7f0e\n",
    ax[1].plot(t, np.full(len(t),  3.8645), label='M4', linestyle='dashdot', color='gold', linewidth=2)#2ca02c\n",
    ax[1].plot(t , np.full(len(t),  1.622), label='Inertial Frequency', linestyle='dashdot', color='#FF00FF', linewidth=2)#1f77b4\n",
    ax[1].set_ylim(0, 5)
    ax[1].set_xlim(0, np.max(t))

   # ax[1].set_ylabel('Period (hrs)')\n",
    ax[1].set_xlabel('Time since release [days]')
    #ax[1].set_title('Wavelet Power Spectrum')
    dy=0.05
    
    
    ax[0].fill_betweenx(fft_freqs, 
                        fft_power.min(), fft_power.max(),  # Horizontal span across the plot
                       where=(fft_freqs >= 1.9323 - dy) & (fft_freqs <= 1.9323 + dy),
                        color='cyan', alpha=0.3, label='M2')
    ax[0].fill_betweenx(fft_freqs,
                fft_power.min(), fft_power.max(),
               where=(fft_freqs >= 2.0 - dy) & (fft_freqs <= 2.0 + dy),
                color='#39FF14', alpha=0.3, label='S2')
    
    ax[0].fill_betweenx(fft_freqs, 
                    fft_power.min(), fft_power.max(),
                    where=(fft_freqs >=3.8645 - dy) & (fft_freqs <=3.8645 + dy),
                    color='gold', alpha=0.3, label='M4')
    ax[0].fill_betweenx(fft_freqs,
                    fft_power.min(), fft_power.max(),
                    where=(fft_freqs >= 1.622 - dy) & (fft_freqs <=1.622 + dy),
                    color='#FF00FF', alpha=0.3, label='Inertial period')
    ax[0].plot(fft_power, fft_freqs, color='black', zorder=2, linewidth=1)
    ax[0].set_ylabel('Frequency [cpd]')
    ax[0].set_xlabel('PSD [$m^2 s^{-2} cpd^{-1}$]')
    ax[0].set_ylim(0, 5)
    "    # Reverse the x-axis limits for the mirrored effect\n",
    ax[0].set_xlim(300, 0)  # From max power to 0\n",
    #ax[0].set_title('FFT')
    ax[0].legend()
    plt.tight_layout()  # Adjust layout to prevent overlap\n",

    if output_path is None:
        PROJ_ROOT = Path(__file__).resolve().parents[1]
        FIGURES_DIR = PROJ_ROOT / "reports" / "figures"
        output_path = FIGURES_DIR / f'{name_fig}.svg'
    plt.savefig(output_path, dpi=300)

    plt.show()

# %%
dt_lowres=lower_temporal_resolution(datatree)

pw_x=power_spectrum(dt_lowres, 'vx')
pw_y=power_spectrum(dt_lowres, 'vy')
pw_total=power_spectrum(dt_lowres, 'v')
# %%
plot_single_Wavelet_period(pw_x, 3*3600, drifter_id=5)
plot_Fourier_Wavelet(pw_x, 3*3600 )

# %%
plot_Fourier_Wavelet(pw_y, 3*3600 )
plot_single_Wavelet_period(pw_y, 3*3600, drifter_id=5)
#%%
#plot_single_Wavelet_period(pw_total, 3*3600, drifter_id=5)
plot_Fourier_Wavelet(pw_total, 3*3600 )

# %%


# %%
