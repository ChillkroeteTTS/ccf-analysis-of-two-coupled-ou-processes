## Good parameter configurations
ensemble = 500 rtakes about 30s but results in very smooth curves.
Correlation plot of e with max(ccf) suggests a linear correlation

## Ideas for analysis
Collection of ideas to mention in the report

### Identical ACF of noise and ou process!?

**What I saw in the measurements:**

> With IC(initial condition) set to 0 (the mean of the process), the ACF of the OU process seems to be same as the one of the noise (should be checked in a graph),
this is probably, because the "relaxation force" is too weak as long as the process is close to it's mean.
>The resolution could also play a role, since the time window where relaxation is happening is very small/the relaxation is very fast.
>So if the resolution is small, deviations of the process are immediatly compensated for.

**Finer Resolution**
> AS can be seen in the pictures acf_10000_ic0/15 (very high resolution), the resolution plays no role in the ACF of the ou.
> With an ic different from 0, there is a clear autocorrelation which diminishes with increasing lag (see acf_5000_ic15.png).
> Without, the AC reduces to zero when lag !== 0. This indicates, that the relaxing force is too strong to allow for a memory to build up.

>> However! how does this explain, that the ccf is not 1 when e = 1 and  

**Todos**

- Plot the ACF of the OU and of the powering noise in one graph to see similarity
- I

### Correlation between factor 'e' and max(ccf)
**What I saw in the measurements** 

>AS suggested by the correlation graph between e and max(ccf) and the ccf plots, the peak value of the ccf rises linear with e.
This is intuitively explainable by 'e' controlling the mixin of the noises powering both ou processes where 1 means they are 
powered by the same noise. However, the ccf never reaches 1 (total identity), since both ou processes have different memories.
s

**Todos**
 
 
### Correlation between the width of the ccf peak and the AC of the driving noise
**What I saw in the measurements**
> Looking at the CCF and the ACF of the noise powering the ou-process, a correlation between the width of the ccf peak and the
> ACF of the noise which powers the first OU-process is indicated.

**Todos**
> A graph showing the ccf in range[500, inf] as well as the ACF of noise1 should clarify if there is any correlation between
> the width of the ccf and the noises ACF. This graph should be plotted with different taus for the red noise.
