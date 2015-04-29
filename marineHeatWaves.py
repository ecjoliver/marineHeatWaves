'''

    A set of functions which implement the Marine Heat Wave (MHW)
    definition of Hobday et al. (in preparation)

'''


import numpy as np
import scipy as sp
from scipy import linalg
import scipy.ndimage as ndimage
from datetime import date


def detect(t, sst, climatologyPeriod=[1983,2012], pctile=90, windowHalfWidth=5, smoothPercentile=True, smoothPercentileWidth=31, minDuration=5, joinAcrossGaps=True, maxGap=2):
    '''

    Applies the Hobday et al. (in preparation) marine heat wave definition to an input time
    series of sst ('sst') along with a time vector ('t'). Outputs properties of
    all detected marine heat waves.

    Inputs:

      t       Time vector, in datetime format (e.g., date(1982,1,1).toordinal())
              [1D numpy array of length T]
      sst     Temperature vector [1D numpy array of length T]

    Outputs:

      mhw     Detected marine heat waves (MHWs). Each key (following list) is a
              list of length N where N is the number of detected MHWs:
 
        'time_start'           Start time of MHW [datetime format]
        'time_end'             End time of MHW [datetime format]
        'time_peak'            Time of MHW peak [datetime format]
        'date_start'           Start date of MHW [datetime format]
        'date_end'             End date of MHW [datetime format]
        'date_peak'            Date of MHW peak [datetime format]
        'index_start'          Start index of MHW
        'index_end'            End index of MHW
        'index_peak'           Index of MHW peak
        'duration'             Duration of MHW [days]
        'intensity_max'        Maximum (peak) intensity [deg. C]
        'intensity_mean'       Mean intensity [deg. C]
        'intensity_var'        Intensity variability [deg. C]
        'intensity_cumulative' Cumulative intensity [deg. C x days]
        'rate_onset'           Onset rate of MHW [deg. C / days]
        'rate_decline'         Decline rate of MHW [deg. C / days]

        'intensity_max_relThresh', 'intensity_mean_relThresh', 'intensity_var_relThresh', 
        and 'intensity_cumulative_relThresh' are as above except relative to the
        threshold (e.g., 90th percentile) rather than the seasonal climatology

        'n_events'             A scalar integer (not a list) indicating the total
                               number of detected MHW events

      clim    Climatology of SST. Each key (following list) is a seasonally-varying
              time series [1D numpy array of length T] of a particular measure:

        'thresh'               Seasonally varying threshold (e.g., 90th percentile)
        'seas'                 Climatological seasonal cycle

    Options:

      climatologyPeriod      Period over which climatology is calculated, specified
                             as list of start and end years (DEFAULT = [1983,2012])
      pctile                 Threshold percentile (%) for detection of extreme values
                             (DEFAULT = 90)
      windowHalfWidth        Width of window (one sided) about day-of-year used for
                             the pooling of values and calculation of threshold percentile
                             (DEFAULT = 5 [days])
      smoothPercentile       Boolean switch indicating whether to smooth the threshold
                             percentile timeseries with a moving average (DEFAULT = True)
      smoothPercentileWidth  Width of moving average window for smoothing threshold
                             (DEFAULT = 31 [days])
      minDuration            Minimum duration for acceptance detected MHWs
                             (DEFAULT = 5 [days])
      joinAcrossGaps         Boolean switch indicating whether to join MHWs
                             which occur before/after a short gap (DEFAULT = True)
      maxGap                 Maximum length of gap allowed for the joining of MHWs
                             (DEFAULT = 2 [days])

    Notes:

      1. This function assumes that the input time series consist of continuous daily values
         with no missing values.

      2. This function supports leap years. This is done by ignoring Feb 29s for the initial
         calculation of the climatology and threshold. The value of these for Feb 29 is then
         linearly interpolated from the values for Feb 28 and Mar 1.

      3. The calculation of onset and decline rates assumes that the heat wave started a half-day
         before the start day and ended a half-day after the end-day. (This is consistent with the
         duration definition as implemented, which assumes duration = end day - start day + 1.)

    Written by Eric Oliver, Institue for Marine and Antarctic Studies, University of Tasmania, Feb 2015

    '''

    #
    # Initialize MHW output variable
    #

    mhw = {}
    mhw['time_start'] = [] # datetime format
    mhw['time_end'] = [] # datetime format
    mhw['time_peak'] = [] # datetime format
    mhw['date_start'] = [] # datetime format
    mhw['date_end'] = [] # datetime format
    mhw['date_peak'] = [] # datetime format
    mhw['index_start'] = []
    mhw['index_end'] = []
    mhw['index_peak'] = []
    mhw['duration'] = [] # [days]
    mhw['intensity_max'] = [] # [deg C]
    mhw['intensity_mean'] = [] # [deg C]
    mhw['intensity_var'] = [] # [deg C]
    mhw['intensity_cumulative'] = [] # [deg C]
    mhw['intensity_max_relThresh'] = [] # [deg C]
    mhw['intensity_mean_relThresh'] = [] # [deg C]
    mhw['intensity_var_relThresh'] = [] # [deg C]
    mhw['intensity_cumulative_relThresh'] = [] # [deg C]
    mhw['rate_onset'] = [] # [deg C / day]
    mhw['rate_decline'] = [] # [deg C / day]

    #
    # Time and dates vectors
    #

    # Generate vectors for year, month, day-of-month, and day-of-year
    T = len(t)
    year = np.zeros((T))
    month = np.zeros((T))
    day = np.zeros((T))
    doy = np.zeros((T))
    for i in range(T):
        year[i] = date.fromordinal(t[i]).year
        month[i] = date.fromordinal(t[i]).month
        day[i] = date.fromordinal(t[i]).day
        doy[i] = t[i] - date(year[i].astype(int),1,1).toordinal() + 1

    # Modify day-of-year vector so that non-leap-years run 1...59 then 61...366
    feb28 = 59
    feb29 = 60
    for y in np.unique(year):
        # Check for non-leap year
        if ~np.sum(doy[year==y] == 366).astype(bool):
            doy[(year==y) * (doy>feb28)] += 1

    #
    # Calculate threshold and seasonal climatology (varying with day-of-year)
    #

    # Length of climatological year
    lenClimYear = 366
    # Start and end indices
    clim_start = np.where(year == climatologyPeriod[0])[0][0]
    clim_end = np.where(year == climatologyPeriod[1])[0][-1]
    # Inialize arrays
    thresh_climYear = np.NaN*np.zeros(lenClimYear)
    seas_climYear = np.NaN*np.zeros(lenClimYear)
    clim = {}
    clim['thresh'] = np.NaN*np.zeros(T)
    clim['seas'] = np.NaN*np.zeros(T)
    # Loop over all day-of-year values, and calculate threshold and seasonal climatology across years
    for d in range(1,lenClimYear+1):
        # Special case for Feb 29
        if d == feb29:
            continue
        # find all indices for each day of the year +/- windowHalfWidth and from them calculate the threshold
        tt0 = np.where(doy[clim_start:clim_end] == d)[0] 
        tt = np.array([])
        for w in range(-windowHalfWidth, windowHalfWidth+1):
            tt = np.append(tt, tt0 + w)
        thresh_climYear[d-1] = np.percentile(sst[tt.astype(int)], pctile)
        seas_climYear[d-1] = np.mean(sst[tt.astype(int)])
    # Special case for Feb 29
    thresh_climYear[feb29-1] = 0.5*thresh_climYear[feb29-2] + 0.5*thresh_climYear[feb29]
    seas_climYear[feb29-1] = 0.5*seas_climYear[feb29-2] + 0.5*seas_climYear[feb29]

    # Smooth if desired
    if smoothPercentile:
        thresh_climYear = runavg(thresh_climYear, smoothPercentileWidth)
        seas_climYear = runavg(seas_climYear, smoothPercentileWidth)

    # Generate threshold for full time series
    clim['thresh'] = thresh_climYear[doy.astype(int)-1]
    clim['seas'] = seas_climYear[doy.astype(int)-1]

    #
    # Find MHWs as exceedances above the threshold
    #

    # Time series of "True" when threshold is exceeded, "False" otherwise
    exceed_bool = sst - clim['thresh']
    exceed_bool[exceed_bool<=0] = False
    exceed_bool[exceed_bool>0] = True
    # Find contiguous regions of exceed_bool = True
    events, n_events = ndimage.label(exceed_bool)

    # Find all MHW events of duration >= minDuration
    for ev in range(1,n_events+1):
        event_duration = (events == ev).sum()
        if event_duration < minDuration:
            continue
        mhw['time_start'].append(t[np.where(events == ev)[0][0]])
        mhw['time_end'].append(t[np.where(events == ev)[0][-1]])

    # Link heat waves that occur before and after a short gap (gap must be no longer than maxGap)
    if joinAcrossGaps:
        # Calculate gap length for each consecutive pair of events
        gaps = np.array(mhw['time_start'][1:]) - np.array(mhw['time_end'][0:-1]) - 1
        if len(gaps) > 0:
            while gaps.min() <= maxGap:
                # Find first short gap
                ev = np.where(gaps <= maxGap)[0][0]
                # Extend first MHW to encompass second MHW (including gap)
                mhw['time_end'][ev] = mhw['time_end'][ev+1]
                # Remove second event from record
                del mhw['time_start'][ev+1]
                del mhw['time_end'][ev+1]
                # Calculate gap length for each consecutive pair of events
                gaps = np.array(mhw['time_start'][1:]) - np.array(mhw['time_end'][0:-1]) - 1
                if len(gaps) == 0:
                    break

    # Calculate marine heat wave properties
    mhw['n_events'] = len(mhw['time_start'])
    for ev in range(mhw['n_events']):
        mhw['date_start'].append(date.fromordinal(mhw['time_start'][ev]))
        mhw['date_end'].append(date.fromordinal(mhw['time_end'][ev]))
        # Get SST series during MHW event, relative to both threshold and to seasonal climatology
        tt_start = np.where(t==mhw['time_start'][ev])[0][0]
        tt_end = np.where(t==mhw['time_end'][ev])[0][0]
        mhw['index_start'].append(tt_start)
        mhw['index_end'].append(tt_end)
        sst_mhw = sst[tt_start:tt_end+1]
        thresh_mhw = clim['thresh'][tt_start:tt_end+1]
        seas_mhw = clim['seas'][tt_start:tt_end+1]
        mhw_relSeas = sst_mhw - seas_mhw
        mhw_relThresh = sst_mhw - thresh_mhw
        # Find peak
        tt_peak = np.argmax(mhw_relSeas)
        mhw['time_peak'].append(mhw['time_start'][ev] + tt_peak)
        mhw['date_peak'].append(date.fromordinal(mhw['time_start'][ev] + tt_peak))
        mhw['index_peak'].append(tt_start + tt_peak)
        # MHW Duration
        mhw['duration'].append(len(mhw_relSeas))
        # MHW Intensity metrics
        mhw['intensity_max'].append(mhw_relSeas[tt_peak])
        mhw['intensity_mean'].append(mhw_relSeas.mean())
        mhw['intensity_var'].append(np.sqrt(mhw_relSeas.var()))
        mhw['intensity_cumulative'].append(mhw_relSeas.sum())
        mhw['intensity_max_relThresh'].append(mhw_relThresh[tt_peak])
        mhw['intensity_mean_relThresh'].append(mhw_relThresh.mean())
        mhw['intensity_var_relThresh'].append(np.sqrt(mhw_relThresh.var()))
        mhw['intensity_cumulative_relThresh'].append(mhw_relThresh.sum())
        # Rates of onset and decline
        # Requires getting MHW strength at "start" and "end" of event (continuous: assume start/end half-day before/after first/last point)
        if tt_start > 0:
            mhw_relSeas_start = 0.5*(mhw_relSeas[0] + sst[tt_start-1] - clim['seas'][tt_start-1])
            mhw['rate_onset'].append((mhw_relSeas[tt_peak] - mhw_relSeas_start) / (tt_peak+0.5))
        else: # MHW starts at beginning of time series
            if tt_peak == 0: # Peak is also at begining of time series, assume onset time = 1 day
                mhw['rate_onset'].append((mhw_relSeas[tt_peak] - mhw_relSeas[0]) / 1.)
            else:
                mhw['rate_onset'].append((mhw_relSeas[tt_peak] - mhw_relSeas[0]) / tt_peak)
        if tt_end < T-1:
            mhw_relSeas_end = 0.5*(mhw_relSeas[-1] + sst[tt_end+1] - clim['seas'][tt_end+1])
            mhw['rate_decline'].append((mhw_relSeas[tt_peak] - mhw_relSeas_end) / (tt_end-tt_peak+0.5))
        else: # MHW finishes at end of time series
            if tt_peak == T-1: # Peak is also at end of time series, assume decline time = 1 day
                mhw['rate_decline'].append((mhw_relSeas[tt_peak] - mhw_relSeas[-1]) / 1.)
            else:
                mhw['rate_decline'].append((mhw_relSeas[tt_peak] - mhw_relSeas[-1]) / (tt_end-tt_peak))

    return mhw, clim


def blockAverage(t, mhw, blockLength=1):
    '''

    Calculate statistics of marine heatwave (MHW) properties averaged over blocks of
    a specified length of time. Takes as input a collection of detected MHWs
    (using the marineHeatWaves.detect function) and a time vector for the source
    SST series.

    Inputs:

      t       Time vector, in datetime format (e.g., date(1982,1,1).toordinal())
      mhw     Marine heat waves (MHWs) detected using marineHeatWaves.detect

    Outputs:

      mhwBlock   Time series of block-averaged MHW properties. Each key (following list)
                 is a list of length N where N is the number of blocks:
 
        'years_start'          Start year blocks (inclusive)
        'years_end'            End year of blocks (inclusive)
        'years_centre'         Decimal year at centre of blocks
        'count'                Total MHW count in each block
        'duration'             Average MHW duration in each block [days]
        'intensity_max'        Average MHW "maximum (peak) intensity" in each block [deg. C]
        'intensity_mean'       Average MHW "mean intensity" in each block [deg. C]
        'intensity_var'        Average MHW "intensity variability" in each block [deg. C]
        'intensity_cumulative' Average MHW "cumulative intensity" in each block [deg. C x days]
        'rate_onset'           Average MHW onset rate in each block [deg. C / days]
        'rate_decline'         Average MHW decline rate in each block [deg. C / days]

        'intensity_max_relThresh', 'intensity_mean_relThresh', 'intensity_var_relThresh', 
        and 'intensity_cumulative_relThresh' are as above except relative to the
        threshold (e.g., 90th percentile) rather than the seasonal climatology

    Options:

      blockLength            Size of block (in years) over which to calculate the
                             averaged MHW properties. Must be an integer greater than
                             or equal to 1 (DEFAULT = 1 [year])

    Notes:

      This function assumes that the input time vector consists of continuous daily values.

    Written by Eric Oliver, Institue for Marine and Antarctic Studies, University of Tasmania, Feb-Mar 2015

    '''

    #
    # Time and dates vectors, and calculate block timing
    #

    # Generate vectors for year, month, day-of-month, and day-of-year
    T = len(t)
    year = np.zeros((T))
    month = np.zeros((T))
    day = np.zeros((T))
    for i in range(T):
        year[i] = date.fromordinal(t[i]).year
        month[i] = date.fromordinal(t[i]).month
        day[i] = date.fromordinal(t[i]).day

    # Number of blocks, round up to include partial blocks at end
    years = np.unique(year)
    nBlocks = np.ceil((years.max() - years.min() + 1) / blockLength)

    #
    # Initialize MHW output variable
    #

    mhwBlock = {}
    mhwBlock['count'] = np.zeros(nBlocks)
    mhwBlock['count'] = np.zeros(nBlocks)
    mhwBlock['duration'] = np.zeros(nBlocks)
    mhwBlock['intensity_max'] = np.zeros(nBlocks)
    mhwBlock['intensity_mean'] = np.zeros(nBlocks)
    mhwBlock['intensity_cumulative'] = np.zeros(nBlocks)
    mhwBlock['intensity_var'] = np.zeros(nBlocks)
    mhwBlock['intensity_max_relThresh'] = np.zeros(nBlocks)
    mhwBlock['intensity_mean_relThresh'] = np.zeros(nBlocks)
    mhwBlock['intensity_cumulative_relThresh'] = np.zeros(nBlocks)
    mhwBlock['intensity_var_relThresh'] = np.zeros(nBlocks)
    mhwBlock['rate_onset'] = np.zeros(nBlocks)
    mhwBlock['rate_decline'] = np.zeros(nBlocks)

    # Start, end, and centre years for all blocks
    mhwBlock['years_start'] = years[range(0, len(years), blockLength)]
    mhwBlock['years_end'] = mhwBlock['years_start'] + blockLength - 1
    mhwBlock['years_centre'] = 0.5*(mhwBlock['years_start'] + mhwBlock['years_end'])

    #
    # Calculate block averages
    #

    for i in range(mhw['n_events']):
        # Block index for year of each MHW (MHW year defined by start year)
        iBlock = np.where((mhwBlock['years_start'] <= mhw['date_start'][i].year) * (mhwBlock['years_end'] >= mhw['date_start'][i].year))[0][0]
        # Add MHW properties to block count
        mhwBlock['count'][iBlock] += 1
        mhwBlock['duration'][iBlock] += mhw['duration'][i]
        mhwBlock['intensity_max'][iBlock] += mhw['intensity_max'][i]
        mhwBlock['intensity_mean'][iBlock] += mhw['intensity_mean'][i]
        mhwBlock['intensity_cumulative'][iBlock] += mhw['intensity_cumulative'][i]
        mhwBlock['intensity_var'][iBlock] += mhw['intensity_var'][i]
        mhwBlock['intensity_max_relThresh'][iBlock] += mhw['intensity_max_relThresh'][i]
        mhwBlock['intensity_mean_relThresh'][iBlock] += mhw['intensity_mean_relThresh'][i]
        mhwBlock['intensity_cumulative_relThresh'][iBlock] += mhw['intensity_cumulative_relThresh'][i]
        mhwBlock['intensity_var_relThresh'][iBlock] += mhw['intensity_var_relThresh'][i]
        mhwBlock['rate_onset'][iBlock] += mhw['rate_onset'][i]
        mhwBlock['rate_decline'][iBlock] += mhw['rate_decline'][i]

    # Calculate averages
    count = 1.*mhwBlock['count']
    count[count==0] = np.nan
    mhwBlock['duration'] = mhwBlock['duration'] / count
    mhwBlock['intensity_max'] = mhwBlock['intensity_max'] / count
    mhwBlock['intensity_mean'] = mhwBlock['intensity_mean'] / count
    mhwBlock['intensity_cumulative'] = mhwBlock['intensity_cumulative'] / count
    mhwBlock['intensity_var'] = mhwBlock['intensity_var'] / count
    mhwBlock['intensity_max_relThresh'] = mhwBlock['intensity_max_relThresh'] / count
    mhwBlock['intensity_mean_relThresh'] = mhwBlock['intensity_mean_relThresh'] / count
    mhwBlock['intensity_cumulative_relThresh'] = mhwBlock['intensity_cumulative_relThresh'] / count
    mhwBlock['intensity_var_relThresh'] = mhwBlock['intensity_var_relThresh'] / count
    mhwBlock['rate_onset'] = mhwBlock['rate_onset'] / count
    mhwBlock['rate_decline'] = mhwBlock['rate_decline'] / count

    return mhwBlock


def meanTrend(mhwBlock):
    '''

    Calculates the mean and trend of marine heatwave (MHW) properties. Takes as input a
    collection of block-averaged MHW properties (using the marineHeatWaves.blockAverage
    function). Handles missing values (which should be specified by NaNs).

    Inputs:

      mhwBlock      Time series of block-averaged MHW statistics calculated using the
                    marineHeatWaves.blockAverage function

    Outputs:

      mean          Mean of all MHW properties over all block-averaged values
      trend         Linear trend of all MHW properties over all block-averaged values

                    Both mean and trend have the following keys, the units the trend
                    are the units of the property of interest per year:

        'duration'             Duration of MHW [days]
        'intensity_max'        Maximum (peak) intensity [deg. C]
        'intensity_mean'       Mean intensity [deg. C]
        'intensity_var'        Intensity variability [deg. C]
        'intensity_cumulative' Cumulative intensity [deg. C x days]
        'rate_onset'           Onset rate of MHW [deg. C / days]
        'rate_decline'         Decline rate of MHW [deg. C / days]

        'intensity_max_relThresh', 'intensity_mean_relThresh', 'intensity_var_relThresh', 
        and 'intensity_cumulative_relThresh' are as above except relative to the
        threshold (e.g., 90th percentile) rather than the seasonal climatology
 
    Notes:

      This calculation performs a multiple linear regression of the form
        y ~ beta * X + eps
      where y is the MHW property of interest and X is a matrix of predictors. The first
      column of X is all ones to estimate the mean, the second column is the time vector
      which is taken as mhwBlock['years_centre'] and offset to be equal to zero at its
      mid-point.

    Written by Eric Oliver, Institue for Marine and Antarctic Studies, University of Tasmania, Feb-Mar 2015

    '''

    # Initialize mean and trend dictionaries
    mean = {}
    trend = {}

    # Construct matrix of predictors, first column is all ones to estimate the mean,
    # second column is the time vector, equal to zero at mid-point.
    t = mhwBlock['years_centre']
    X = np.array([np.ones(t.shape), t-t.mean()]).T

    # Loop over all keys in mhwBlock
    for key in mhwBlock.keys():
        # Skip time-vector keys of mhwBlock
        if (key == 'years_centre') + (key == 'years_end') + (key == 'years_start'):
            continue

        # Predictand (MHW property of interest)
        y = mhwBlock[key]
        valid = ~np.isnan(y) # non-NaN indices

        # Perform linear regression over valid indices
        if np.sum(~np.isnan(y)) > 0: # If at least one non-NaN value
            beta = linalg.lstsq(X[valid,:], y[valid])[0]
        else:
            beta = [np.nan, np.nan]

        # Insert regression coefficients into mean and trend dictionaries
        mean[key] = beta[0]
        trend[key] = beta[1]

    # Return mean, trend
    return mean, trend


def runavg(ts, w):
    '''

    Performs a running average of an input time series using uniform window
    of width w. This function assumes that the input time series is periodic.

    Inputs:

      ts            Time series [1D numpy array]
      w             Integer length (must be odd) of running average window

    Outputs:

      ts_smooth     Smoothed time series

    Written by Eric Oliver, Institue for Marine and Antarctic Studies, University of Tasmania, Feb-Mar 2015

    '''
    # Original length of ts
    N = len(ts)
    # make ts three-fold periodic
    ts = np.append(ts, np.append(ts, ts))
    # smooth by convolution with a window of equal weights
    ts_smooth = np.convolve(ts, np.ones(w)/w, mode='same')
    # Only output central section, of length equal to the original length of ts
    ts = ts_smooth[N:2*N]

    return ts
