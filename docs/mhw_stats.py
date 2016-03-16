'''

  Software which uses the MHW definition
  of Hobday et al. (2016) applied to
  a select SST time series and outputs
  summary plots and statistics

'''

# Load required modules

import numpy as np
from scipy import io
from datetime import date
from Scientific.IO import NetCDF

from matplotlib import pyplot as plt

import marineHeatWaves as mhw

# Some basic parameters

coldSpells = False # If true detect coldspells instead of heatwaves
col_clim = '0.25'
col_thresh = 'g-'
if coldSpells:
    mhwname = 'MCS'
    mhwfullname = 'coldspell'
    col_evMax = (0, 102./255, 204./255)
    col_ev = (153./255, 204./255, 1)
    col_bar = (0.5, 0.5, 1)
else:
    mhwname = 'MHW'
    mhwfullname = 'heatwave'
    col_evMax = 'r'
    col_ev = (1, 0.6, 0.5)
    col_bar = (1, 0.5, 0.5)

#
# Generate synthetic temperature time series
# (Replace this with loading in of real temperature data)
#

# Generate time vector using datetime format (January 1 of year 1 is day 1)
t = np.arange(date(1982,1,1).toordinal(),date(2014,12,31).toordinal()+1)
dates = [date.fromordinal(tt.astype(int)) for tt in t]
# Generate synthetic temperature time series
sst = np.zeros(len(t))
sst[0] = 0 # Initial condition
a = 0.85 # autoregressive parameter
for i in range(1,len(t)):
    sst[i] = a*sst[i-1] + 0.75*np.random.randn() + 0.5*np.cos(t[i]*2*np.pi/365.25)
sst = sst - sst.min() + 5.

#
# Apply Marine Heat Wave definition
#

n = 0
mhws, clim = mhw.detect(t, sst, coldSpells=coldSpells)
mhwBlock = mhw.blockAverage(t, mhws, temp=sst)
mean, trend, dtrend = mhw.meanTrend(mhwBlock)

# Plot various summary things

plt.figure(figsize=(15,7))
plt.subplot(2,2,1)
evMax = np.argmax(mhws['duration'])
plt.bar(range(mhws['n_events']), mhws['duration'], width=0.6, color=(0.7,0.7,0.7))
plt.bar(evMax, mhws['duration'][evMax], width=0.6, color=col_bar)
plt.xlim(0, mhws['n_events'])
plt.ylabel('[days]')
plt.title('Duration')
plt.subplot(2,2,2)
evMax = np.argmax(np.abs(mhws['intensity_max']))
plt.bar(range(mhws['n_events']), mhws['intensity_max'], width=0.6, color=(0.7,0.7,0.7))
plt.bar(evMax, mhws['intensity_max'][evMax], width=0.6, color=col_bar)
plt.xlim(0, mhws['n_events'])
plt.ylabel(r'[$^\circ$C]')
plt.title('Maximum Intensity')
plt.subplot(2,2,4)
evMax = np.argmax(np.abs(mhws['intensity_mean']))
plt.bar(range(mhws['n_events']), mhws['intensity_mean'], width=0.6, color=(0.7,0.7,0.7))
plt.bar(evMax, mhws['intensity_mean'][evMax], width=0.6, color=col_bar)
plt.xlim(0, mhws['n_events'])
plt.title('Mean Intensity')
plt.ylabel(r'[$^\circ$C]')
plt.xlabel(mhwname + ' event number')
plt.subplot(2,2,3)
evMax = np.argmax(np.abs(mhws['intensity_cumulative']))
plt.bar(range(mhws['n_events']), mhws['intensity_cumulative'], width=0.6, color=(0.7,0.7,0.7))
plt.bar(evMax, mhws['intensity_cumulative'][evMax], width=0.6, color=col_bar)
plt.xlim(0, mhws['n_events'])
plt.title(r'Cumulative Intensity')
plt.ylabel(r'[$^\circ$C$\times$days]')
plt.xlabel(mhwname + ' event number')
plt.savefig('mhw_stats/' + mhwname + '_list_byNumber.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

ts = date(1982,1,1).toordinal()
te = date(2015,3,1).toordinal()
plt.figure(figsize=(15,7))
plt.subplot(2,2,1)
evMax = np.argmax(mhws['duration'])
plt.bar(mhws['date_peak'], mhws['duration'], width=150, color=(0.7,0.7,0.7))
plt.bar(mhws['date_peak'][evMax], mhws['duration'][evMax], width=150, color=col_bar)
plt.xlim(ts, te)
plt.ylabel('[days]')
plt.title('Duration')
plt.subplot(2,2,2)
evMax = np.argmax(np.abs(mhws['intensity_max']))
plt.bar(mhws['date_peak'], mhws['intensity_max'], width=150, color=(0.7,0.7,0.7))
plt.bar(mhws['date_peak'][evMax], mhws['intensity_max'][evMax], width=150, color=col_bar)
plt.xlim(ts, te)
plt.ylabel(r'[$^\circ$C]')
plt.title('Maximum Intensity')
plt.subplot(2,2,4)
evMax = np.argmax(np.abs(mhws['intensity_mean']))
plt.bar(mhws['date_peak'], mhws['intensity_mean'], width=150, color=(0.7,0.7,0.7))
plt.bar(mhws['date_peak'][evMax], mhws['intensity_mean'][evMax], width=150, color=col_bar)
plt.xlim(ts, te)
plt.title('Mean Intensity')
plt.ylabel(r'[$^\circ$C]')
plt.subplot(2,2,3)
evMax = np.argmax(np.abs(mhws['intensity_cumulative']))
plt.bar(mhws['date_peak'], mhws['intensity_cumulative'], width=150, color=(0.7,0.7,0.7))
plt.bar(mhws['date_peak'][evMax], mhws['intensity_cumulative'][evMax], width=150, color=col_bar)
plt.xlim(ts, te)
plt.title(r'Cumulative Intensity')
plt.ylabel(r'[$^\circ$C$\times$days]')
plt.savefig('mhw_stats/' + mhwname + '_list_byDate.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

# Plot top 10 events

# Maximum intensity
outfile = open('mhw_stats/' + mhwname + '_topTen_iMax.txt', 'w')
evs = np.argsort(np.abs(mhws['intensity_max']))[-10:]
plt.figure(figsize=(23,16))
for i in range(10):
    ev = evs[-(i+1)]
    plt.subplot(5,2,i+1)
    # Find indices for all ten MHWs before and after event of interest and shade accordingly
    for ev0 in np.arange(max(ev-10,0), min(ev+11,mhws['n_events']-1), 1):
        t1 = np.where(t==mhws['time_start'][ev0])[0][0]
        t2 = np.where(t==mhws['time_end'][ev0])[0][0]
        plt.fill_between(dates[t1:t2+1], sst[t1:t2+1], clim['thresh'][t1:t2+1], color=col_ev)
    # Find indices for MHW of interest (2011 WA event) and shade accordingly
    t1 = np.where(t==mhws['time_start'][ev])[0][0]
    t2 = np.where(t==mhws['time_end'][ev])[0][0]
    plt.fill_between(dates[t1:t2+1], sst[t1:t2+1], clim['thresh'][t1:t2+1], color=col_evMax)
    # Plot SST, seasonal cycle, threshold, shade MHWs with main event in red
    plt.plot(dates, sst, 'k-', linewidth=2)
    plt.plot(dates, clim['thresh'], col_thresh, linewidth=2)
    plt.plot(dates, clim['seas'], col_clim, linewidth=2)
    plt.title('Number ' + str(i+1))
    plt.xlim(mhws['time_start'][ev]-150, mhws['time_end'][ev]+150)
    if coldSpells:
        plt.ylim(clim['seas'].min() + mhws['intensity_max'][ev] - 0.5, clim['seas'].max() + 1)
    else:
        plt.ylim(clim['seas'].min() - 1, clim['seas'].max() + mhws['intensity_max'][ev] + 0.5)
    plt.ylabel(r'SST [$^\circ$C]')
    # Save stats
    outfile.write('Number ' + str(i+1) + '\n')
    outfile.write('Maximum intensity: ' + str(mhws['intensity_max'][ev]) + ' deg. C\n')
    outfile.write('Average intensity: '+ str( mhws['intensity_mean'][ev]) + ' deg. C\n')
    outfile.write('Cumulative intensity: ' + str(mhws['intensity_cumulative'][ev]) + ' deg. C-days\n')
    outfile.write('Duration: ' + str(mhws['duration'][ev]) + ' days\n')
    outfile.write('Start date: ' + str(mhws['date_start'][ev].strftime("%d %B %Y")) + '\n')
    outfile.write('End date: ' + str(mhws['date_end'][ev].strftime("%d %B %Y")) + '\n')
    outfile.write('\n')

plt.legend(['SST', 'threshold', 'seasonal climatology'], loc=4)
outfile.close()
plt.savefig('mhw_stats/' + mhwname + '_topTen_iMax.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

# Mean intensity
outfile = open('mhw_stats/' + mhwname + '_topTen_iMean.txt', 'w')
evs = np.argsort(np.abs(mhws['intensity_mean']))[-10:]
plt.clf()
for i in range(10):
    ev = evs[-(i+1)]
    plt.subplot(5,2,i+1)
    # Find indices for all ten MHWs before and after event of interest and shade accordingly
    for ev0 in np.arange(max(ev-10,0), min(ev+11,mhws['n_events']-1), 1):
        t1 = np.where(t==mhws['time_start'][ev0])[0][0]
        t2 = np.where(t==mhws['time_end'][ev0])[0][0]
        plt.fill_between(dates[t1:t2+1], sst[t1:t2+1], clim['thresh'][t1:t2+1], color=col_ev)
    # Find indices for MHW of interest (2011 WA event) and shade accordingly
    t1 = np.where(t==mhws['time_start'][ev])[0][0]
    t2 = np.where(t==mhws['time_end'][ev])[0][0]
    plt.fill_between(dates[t1:t2+1], sst[t1:t2+1], clim['thresh'][t1:t2+1], color=col_evMax)
    # Plot SST, seasonal cycle, threshold, shade MHWs with main event in red
    plt.plot(dates, sst, 'k-', linewidth=2)
    plt.plot(dates, clim['thresh'], col_thresh, linewidth=2)
    plt.plot(dates, clim['seas'], col_clim, linewidth=2)
    plt.title('Number ' + str(i+1))
    plt.xlim(mhws['time_start'][ev]-150, mhws['time_end'][ev]+150)
    if coldSpells:
        plt.ylim(clim['seas'].min() + mhws['intensity_max'][ev] - 0.5, clim['seas'].max() + 1)
    else:
        plt.ylim(clim['seas'].min() - 1, clim['seas'].max() + mhws['intensity_max'][ev] + 0.5)
    plt.ylabel(r'SST [$^\circ$C]')
    # Save stats
    outfile.write('Number ' + str(i+1) + '\n')
    outfile.write('Maximum intensity: ' + str(mhws['intensity_max'][ev]) + ' deg. C\n')
    outfile.write('Average intensity: '+ str( mhws['intensity_mean'][ev]) + ' deg. C\n')
    outfile.write('Cumulative intensity: ' + str(mhws['intensity_cumulative'][ev]) + ' deg. C-days\n')
    outfile.write('Duration: ' + str(mhws['duration'][ev]) + ' days\n')
    outfile.write('Start date: ' + str(mhws['date_start'][ev].strftime("%d %B %Y")) + '\n')
    outfile.write('End date: ' + str(mhws['date_end'][ev].strftime("%d %B %Y")) + '\n')
    outfile.write('\n')

plt.legend(['SST', 'threshold', 'seasonal climatology'], loc=4)
outfile.close()
plt.savefig('mhw_stats/' + mhwname + '_topTen_iMean.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

# Cumulative intensity
outfile = open('mhw_stats/' + mhwname + '_topTen_iCum.txt', 'w')
evs = np.argsort(np.abs(mhws['intensity_cumulative']))[-10:]
plt.clf()
for i in range(10):
    ev = evs[-(i+1)]
    plt.subplot(5,2,i+1)
    # Find indices for all ten MHWs before and after event of interest and shade accordingly
    for ev0 in np.arange(max(ev-10,0), min(ev+11,mhws['n_events']-1), 1):
        t1 = np.where(t==mhws['time_start'][ev0])[0][0]
        t2 = np.where(t==mhws['time_end'][ev0])[0][0]
        plt.fill_between(dates[t1:t2+1], sst[t1:t2+1], clim['thresh'][t1:t2+1], color=col_ev)
    # Find indices for MHW of interest (2011 WA event) and shade accordingly
    t1 = np.where(t==mhws['time_start'][ev])[0][0]
    t2 = np.where(t==mhws['time_end'][ev])[0][0]
    plt.fill_between(dates[t1:t2+1], sst[t1:t2+1], clim['thresh'][t1:t2+1], color=col_evMax)
    # Plot SST, seasonal cycle, threshold, shade MHWs with main event in red
    plt.plot(dates, sst, 'k-', linewidth=2)
    plt.plot(dates, clim['thresh'], col_thresh, linewidth=2)
    plt.plot(dates, clim['seas'], col_clim, linewidth=2)
    plt.title('Number ' + str(i+1))
    plt.xlim(mhws['time_start'][ev]-150, mhws['time_end'][ev]+150)
    if coldSpells:
        plt.ylim(clim['seas'].min() + mhws['intensity_max'][ev] - 0.5, clim['seas'].max() + 1)
    else:
        plt.ylim(clim['seas'].min() - 1, clim['seas'].max() + mhws['intensity_max'][ev] + 0.5)
    plt.ylabel(r'SST [$^\circ$C]')
    # Save stats
    outfile.write('Number ' + str(i+1) + '\n')
    outfile.write('Maximum intensity: ' + str(mhws['intensity_max'][ev]) + ' deg. C\n')
    outfile.write('Average intensity: '+ str( mhws['intensity_mean'][ev]) + ' deg. C\n')
    outfile.write('Cumulative intensity: ' + str(mhws['intensity_cumulative'][ev]) + ' deg. C-days\n')
    outfile.write('Duration: ' + str(mhws['duration'][ev]) + ' days\n')
    outfile.write('Start date: ' + str(mhws['date_start'][ev].strftime("%d %B %Y")) + '\n')
    outfile.write('End date: ' + str(mhws['date_end'][ev].strftime("%d %B %Y")) + '\n')
    outfile.write('\n')

plt.legend(['SST', 'threshold', 'seasonal climatology'], loc=4)
outfile.close()
plt.savefig('mhw_stats/' + mhwname + '_topTen_iCum.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

# Duration
outfile = open('mhw_stats/' + mhwname + '_topTen_Dur.txt', 'w')
evs = np.argsort(mhws['duration'])[-10:]
plt.clf()
for i in range(10):
    ev = evs[-(i+1)]
    plt.subplot(5,2,i+1)
    # Find indices for all ten MHWs before and after event of interest and shade accordingly
    for ev0 in np.arange(max(ev-10,0), min(ev+11,mhws['n_events']-1), 1):
        t1 = np.where(t==mhws['time_start'][ev0])[0][0]
        t2 = np.where(t==mhws['time_end'][ev0])[0][0]
        plt.fill_between(dates[t1:t2+1], sst[t1:t2+1], clim['thresh'][t1:t2+1], color=col_ev)
    # Find indices for MHW of interest (2011 WA event) and shade accordingly
    t1 = np.where(t==mhws['time_start'][ev])[0][0]
    t2 = np.where(t==mhws['time_end'][ev])[0][0]
    plt.fill_between(dates[t1:t2+1], sst[t1:t2+1], clim['thresh'][t1:t2+1], color=col_evMax)
    # Plot SST, seasonal cycle, threshold, shade MHWs with main event in red
    plt.plot(dates, sst, 'k-', linewidth=2)
    plt.plot(dates, clim['thresh'], col_thresh, linewidth=2)
    plt.plot(dates, clim['seas'], col_clim, linewidth=2)
    plt.title('Number ' + str(i+1))
    plt.xlim(mhws['time_start'][ev]-150, mhws['time_end'][ev]+150)
    if coldSpells:
        plt.ylim(clim['seas'].min() + mhws['intensity_max'][ev] - 0.5, clim['seas'].max() + 1)
    else:
        plt.ylim(clim['seas'].min() - 1, clim['seas'].max() + mhws['intensity_max'][ev] + 0.5)
    plt.ylabel(r'SST [$^\circ$C]')
    # Save stats
    outfile.write('Number ' + str(i+1) + '\n')
    outfile.write('Maximum intensity: ' + str(mhws['intensity_max'][ev]) + ' deg. C\n')
    outfile.write('Average intensity: '+ str( mhws['intensity_mean'][ev]) + ' deg. C\n')
    outfile.write('Cumulative intensity: ' + str(mhws['intensity_cumulative'][ev]) + ' deg. C-days\n')
    outfile.write('Duration: ' + str(mhws['duration'][ev]) + ' days\n')
    outfile.write('Start date: ' + str(mhws['date_start'][ev].strftime("%d %B %Y")) + '\n')
    outfile.write('End date: ' + str(mhws['date_end'][ev].strftime("%d %B %Y")) + '\n')
    outfile.write('\n')

plt.legend(['SST', 'threshold', 'seasonal climatology'], loc=4)
outfile.close()
plt.savefig('mhw_stats/' + mhwname + '_topTen_Dur.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

# Annual averages
years = mhwBlock['years_centre']
plt.figure(figsize=(13,7))
plt.subplot(2,2,2)
plt.plot(years, mhwBlock['count'], 'k-')
plt.plot(years, mhwBlock['count'], 'ko')
if np.abs(trend['count']) - dtrend['count'] > 0:
     plt.title('Frequency (trend = ' + '{:.2}'.format(10*trend['count']) + '* per decade)')
else:
     plt.title('Frequency (trend = ' + '{:.2}'.format(10*trend['count']) + ' per decade)')
plt.ylabel('[count per year]')
plt.grid()
plt.subplot(2,2,1)
plt.plot(years, mhwBlock['duration'], 'k-')
plt.plot(years, mhwBlock['duration'], 'ko')
if np.abs(trend['duration']) - dtrend['duration'] > 0:
    plt.title('Duration (trend = ' + '{:.2}'.format(10*trend['duration']) + '* per decade)')
else:
    plt.title('Duration (trend = ' + '{:.2}'.format(10*trend['duration']) + ' per decade)')
plt.ylabel('[days]')
plt.grid()
plt.subplot(2,2,4)
plt.plot(years, mhwBlock['intensity_max'], '-', color=col_evMax)
plt.plot(years, mhwBlock['intensity_mean'], 'k-')
plt.plot(years, mhwBlock['intensity_max'], 'o', color=col_evMax)
plt.plot(years, mhwBlock['intensity_mean'], 'ko')
plt.legend(['Max', 'mean'], loc=2)
if (np.abs(trend['intensity_max']) - dtrend['intensity_max'] > 0) * (np.abs(trend['intensity_mean']) - dtrend['intensity_mean'] > 0):
    plt.title('Intensity (trend = ' + '{:.2}'.format(10*trend['intensity_max']) + '* (max), ' + '{:.2}'.format(10*trend['intensity_mean'])  + '* (mean) per decade)')
elif (np.abs(trend['intensity_max']) - dtrend['intensity_max'] > 0):
    plt.title('Intensity (trend = ' + '{:.2}'.format(10*trend['intensity_max']) + '* (max), ' + '{:.2}'.format(10*trend['intensity_mean'])  + ' (mean) per decade)')
elif (np.abs(trend['intensity_mean']) - dtrend['intensity_mean'] > 0):
    plt.title('Intensity (trend = ' + '{:.2}'.format(10*trend['intensity_max']) + ' (max), ' + '{:.2}'.format(10*trend['intensity_mean'])  + '* (mean) per decade)')
else:
    plt.title('Intensity (trend = ' + '{:.2}'.format(10*trend['intensity_max']) + ' (max), ' + '{:.2}'.format(10*trend['intensity_mean'])  + ' (mean) per decade)')
plt.ylabel(r'[$^\circ$C]')
plt.grid()
plt.subplot(2,2,3)
plt.plot(years, mhwBlock['intensity_cumulative'], 'k-')
plt.plot(years, mhwBlock['intensity_cumulative'], 'ko')
if np.abs(trend['intensity_cumulative']) - dtrend['intensity_cumulative'] > 0:
    plt.title('Cumulative intensity (trend = ' + '{:.2}'.format(10*trend['intensity_cumulative']) + '* per decade)')
else:
    plt.title('Cumulative intensity (trend = ' + '{:.2}'.format(10*trend['intensity_cumulative']) + ' per decade)')
plt.ylabel(r'[$^\circ$C$\times$days]')
plt.grid()
plt.savefig('mhw_stats/' + mhwname + '_annualAverages_meanTrend.png', bbox_inches='tight', pad_inches=0.5, dpi=150)

# Save results as text data
outfile = 'mhw_stats/' + mhwname + '_data'

# Event data
csvfile = open(outfile +'.events.csv', 'w')
#csvfile.write('# Marine heatwave statistics for individual detected events at [' + str(lon[i]) + ' E ' + str(lat[j]) + ' N] from NOAA OI AVHRR V2 SST data (1982-2014)\n')
csvfile.write('# Marine ' + mhwfullname + ' statistics for individual detected events at ' + locations['name'][0] + ' from NOAA OI AVHRR V2 SST data (1982-2014)\n')
csvfile.write('Event number, Start year, Start month, Start day, Peak year, Peak month, Peak day, End year, End month, End day, Duration [days], Maximum intensity [deg C], Mean intensity [deg C], Cumulative intensity [deg C x days], Intensity variability [deg C], Rate of onset [deg C / days], Rate of decline [deg C / days], Maximum intensity (rel. thresh.) [deg C], Mean intensity (rel. thresh.) [deg C], Cumulative intensity (rel. thresh.) [deg C x days], Intensity variability (rel. thresh.) [deg C], Maximum intensity (absolute) [deg C], Mean intensity (absolute) [deg C], Cumulative intensity (absolute) [deg C x days], Intensity variability (absolute) [deg C], Maximum intensity (normalized) [unitless], Mean intensity (normalized) [unitless]\n')
for ev in range(mhws['n_events']):
    csvfile.write(str(ev+1) + ', ' + str(mhws['date_start'][ev].year) + ', ' + str(mhws['date_start'][ev].month) + ', ' + str(mhws['date_start'][ev].day) + ', ' + str(mhws['date_peak'][ev].year) + ', ' + str(mhws['date_peak'][ev].month) + ', ' + str(mhws['date_peak'][ev].day) + ', ' + str(mhws['date_end'][ev].year) + ', ' + str(mhws['date_end'][ev].month) + ', ' + str(mhws['date_end'][ev].day) + ', ' + str(mhws['duration'][ev]) + ', ' + '{:.4}'.format(mhws['intensity_max'][ev]) + ', ' + '{:.4}'.format(mhws['intensity_mean'][ev]) + ', ' + '{:.4}'.format(mhws['intensity_cumulative'][ev]) + ', ' + '{:.4}'.format(mhws['intensity_var'][ev]) + ', ' + '{:.4}'.format(mhws['rate_onset'][ev]) + ', ' + '{:.4}'.format(mhws['rate_decline'][ev]) + ', ' + '{:.4}'.format(mhws['intensity_max_relThresh'][ev]) + ', ' + '{:.4}'.format(mhws['intensity_mean_relThresh'][ev]) + ', ' + '{:.4}'.format(mhws['intensity_cumulative_relThresh'][ev]) + ', ' + '{:.4}'.format(mhws['intensity_var_relThresh'][ev]) + ', ' + '{:.4}'.format(mhws['intensity_max_abs'][ev]) + ', ' + '{:.4}'.format(mhws['intensity_mean_abs'][ev]) + ', ' + '{:.4}'.format(mhws['intensity_cumulative_abs'][ev]) + ', ' + '{:.4}'.format(mhws['intensity_var_abs'][ev]) + ', ' + '{:.4}'.format(mhws['intensity_max_norm'][ev]) + ', ' + '{:.4}'.format(mhws['intensity_mean_norm'][ev]) + '\n')
csvfile.close()

# Annual average data
csvfile = open(outfile +'.annual.csv', 'w')
#csvfile.write('# Annual average marine heatwave statistics at [' + str(lon[i]) + ' E ' + str(lat[j]) + ' N] from NOAA OI AVHRR V2 SST data (1982-2014)\n')
csvfile.write('# Annual average marine ' + mhwfullname + ' statistics at ' + locations['name'][0] + ' from NOAA OI AVHRR V2 SST data (1982-2014)\n')
csvfile.write('# A value of nan indicates missing data. This should correspond to a year with no MHW events (count = 0)\n')
csvfile.write('Year, ' + mhwname + ' event count [number], Duration [days], Maximum intensity [deg C], Mean intensity [deg C], Cumulative intensity [deg C x days], Total ' + mhwname + ' days [days], Total cumulative intensity [deg C x days], Intensity variability [deg C], Rate of onset [deg C / days], Rate of decline [deg C / days], Maximum intensity (rel. thresh.) [deg C], Mean intensity (rel. thresh.) [deg C], Cumulative intensity (rel. thresh.) [deg C x days], Intensity variability (rel. thresh.) [deg C], Maximum intensity (absolute) [deg C], Mean intensity (absolute) [deg C], Cumulative intensity (absolute) [deg C x days], Intensity variability (absolute) [deg C], Maximum intensity (normalized) [unitless], Mean intensity (normalized) [unitless], Mean temperature [deg C], Max temperature [deg C], Min temperature [deg C]\n')
for yr in range(len(mhwBlock['years_centre'])):
    csvfile.write(str(mhwBlock['years_centre'][yr].astype(int)) + ', ' + str(mhwBlock['count'][yr].astype(int)) + ', ' + '{:.4}'.format(mhwBlock['duration'][yr]) + ', ' + '{:.4}'.format(mhwBlock['intensity_max'][yr]) + ', ' + '{:.4}'.format(mhwBlock['intensity_mean'][yr]) + ', ' + '{:.4}'.format(mhwBlock['intensity_cumulative'][yr]) + ', ' + '{:.4}'.format(mhwBlock['total_days'][yr]) + ', ' + '{:.4}'.format(mhwBlock['total_icum'][yr]) + ', ' + '{:.4}'.format(mhwBlock['intensity_var'][yr]) + ', ' + '{:.4}'.format(mhwBlock['rate_onset'][yr]) + ', ' + '{:.4}'.format(mhwBlock['rate_decline'][yr]) + ', ' + '{:.4}'.format(mhwBlock['intensity_max_relThresh'][yr]) + ', ' + '{:.4}'.format(mhwBlock['intensity_mean_relThresh'][yr]) + ', ' + '{:.4}'.format(mhwBlock['intensity_cumulative_relThresh'][yr]) + ', ' + '{:.4}'.format(mhwBlock['intensity_var_relThresh'][yr]) + ', ' + '{:.4}'.format(mhwBlock['intensity_max_abs'][yr]) + ', ' + '{:.4}'.format(mhwBlock['intensity_mean_abs'][yr]) + ', ' + '{:.4}'.format(mhwBlock['intensity_cumulative_abs'][yr]) + ', ' + '{:.4}'.format(mhwBlock['intensity_var_abs'][yr]) + ', ' + '{:.4}'.format(mhwBlock['intensity_max_norm'][yr]) + ', ' + '{:.4}'.format(mhwBlock['intensity_mean_norm'][yr]) + ', ' + '{:.4}'.format(mhwBlock['temp_mean'][yr]) + ', ' + '{:.4}'.format(mhwBlock['temp_max'][yr]) + ', ' + '{:.4}'.format(mhwBlock['temp_min'][yr]) + '\n')
csvfile.close()

