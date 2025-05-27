from dataclasses import dataclass
from functools import partial
from typing import Tuple

import dill
import jax
import jax.numpy as np
from numpyro import deterministic, factor, sample
from numpyro.distributions import Uniform, Normal, TruncatedNormal
from tqdm.auto import tqdm

from utils import (
    Parameters,
    apply_filter_multiple,
    event_ln_likelihood,
    infft,
    kl_divergence_from_signals,
    nfft,
    segment_times,
)

FMIN = 0
FMAX = 10


@dataclass
class BurstParameters(Parameters):
    """
    The parameters of a sine-Gaussian burst.

    Parameters
    ==========
    frequency: np.ndarray
        The peak frequency of the signal [Hz].
    amplitude: np.ndarray
        The amplitude of the signal.
    bandwidth: np.ndarray
        The bandwidth of the source [Hz].
    phase: np.ndarray
        The phase of the signal at the peak time [rad].
    delta_t: np.ndarray
        The peak time of the signal relative to the midpoint of the
        time series [s].
    """

    frequency: np.ndarray
    amplitude: np.ndarray
    bandwidth: np.ndarray
    phase: np.ndarray
    delta_t: np.ndarray


@jax.jit
def gausspulse_complex(
    times: np.ndarray, parameters: BurstParameters, sample_rate: float = 256
) -> np.ndarray:
    r"""
    Generate a complex time-domain sine-gaussian pulse.

    ... math::

        h(t; f, \Delta, \varphi, \delta_{t}) =

    Parameters
    ==========
    times: np.ndarray
        The times to evaluate the signal at.
    frequency: [float, np.ndarray]
        The peak frequency of the signal [Hz].
    bandwidth: [float, np.ndarray]
        The bandwidth of the source [Hz].
    phase: [float, np.ndarray]
        The phase of the signal at the peak time [rad].
    delta_t: [float, np.ndarray]
        The peak time of the signal relative to the midpoint of the
        time series [s].
    """
    ref = np.power(10.0, -0.3)
    a = -((np.pi * parameters.frequency * parameters.bandwidth) ** 2) / (
        4.0 * np.log(ref)
    )

    if parameters.delta_t.ndim == 1:
        times = times[:, np.newaxis] - parameters.delta_t
    else:
        times = times + parameters.delta_t
    ft = parameters.frequency * times
    yenv = (
        np.exp(-a * times**2)
        * (parameters.bandwidth * parameters.frequency) ** 0.25
    )
    yenv *= (64 / sample_rate) ** 0.5
    exponent = 2 * (np.pi * ft + parameters.phase)
    return np.nan_to_num(yenv * np.exp(1j * exponent)).squeeze()


@jax.jit
def gausspulse(
    times: np.ndarray, parameters: BurstParameters, sample_rate: float = 256
) -> np.ndarray:
    __doc__ = gausspulse_complex.__doc__.replace("complex", "real")
    return gausspulse_complex(times, parameters, sample_rate).real


def generic_signal(
    parameters: Parameters,
    times: np.ndarray,
    signal_function: callable,
    *,
    sample_rate: float = 256,
) -> np.ndarray:
    return (
        np.atleast_1d(parameters.amplitude)
        * signal_function(times, parameters, sample_rate=sample_rate)
    ).T


signal = jax.jit(partial(generic_signal, signal_function=gausspulse))
signal_complex = jax.jit(partial(generic_signal, signal_function=gausspulse_complex))

BURST_DEFAULTS = dict(
    frequency=np.array(5.0),
    bandwidth=np.array(0.5),
    amplitude=np.array(50.0),
    phase=np.array(0.0),
    delta_t=np.array(0.0),
)


def draw_single(
    prng_key: jax.Array,
    mean_frequency: np.ndarray,
    sigma_frequency: np.ndarray,
    bounds: dict,
) -> BurstParameters:
    prng_key, *subkeys = jax.random.split(prng_key, 6)
    args = ((), float)  # shape, dtype
    params = BURST_DEFAULTS.copy()
    for ii, (key, value) in enumerate(bounds.items()):
        if key == "frequency":
            # value = (mean_frequency, sigma_frequency)
            low = (FMIN - mean_frequency) / sigma_frequency
            high = (FMAX - mean_frequency) / sigma_frequency
            params[key] = jax.random.truncated_normal(
                subkeys[ii], low, high, shape=(), dtype=float
            ) * sigma_frequency + mean_frequency
        elif key in ["frequency", "amplitude", "bandwidth", "delta_t"]:
            params[key] = jax.random.normal(subkeys[ii], shape=(), dtype=float) * value[1] + value[0]
        else:
            params[key] = jax.random.uniform(subkeys[ii], *args, *value)
    return BurstParameters(**params)


@jax.jit
def simulate_signal(
    prng_key: jax.Array,
    mean_frequency: np.ndarray,
    sigma_frequency: np.ndarray,
    bounds: dict,
    sample_rate: float,
    times: np.ndarray,
) -> np.ndarray:
    parameters = draw_single(prng_key, mean_frequency, sigma_frequency, bounds=bounds)
    sig = signal(parameters, sample_rate=sample_rate, times=times)
    return sig.squeeze()


@jax.jit
def simulate_complex(
    prng_key: jax.Array,
    mean_frequency: np.ndarray,
    sigma_frequency: np.ndarray,
    bounds: dict,
    sample_rate: float,
    times: np.ndarray,
) -> np.ndarray:
    parameters = draw_single(prng_key, mean_frequency, sigma_frequency, bounds=bounds)
    sig = signal_complex(parameters, sample_rate=sample_rate, times=times)
    return sig.squeeze()


@jax.jit
def simulate_event(
    prng_key: jax.Array,
    mean_frequency: np.ndarray,
    sigma_frequency: np.ndarray,
    bounds: dict,
    duration: float,
    sample_rate: float,
    times: np.ndarray,
) -> Tuple[np.ndarray, BurstParameters]:
    parameters = draw_single(prng_key, mean_frequency, sigma_frequency, bounds=bounds)
    sig = signal(parameters, times=times, sample_rate=sample_rate).squeeze()
    sig = nfft(sig, sample_rate)
    prng_key, subkey_1, subkey_2 = jax.random.split(prng_key, 3)
    noise = (
        (
            jax.random.normal(subkey_1, shape=sig.shape)
            + 1j * jax.random.normal(subkey_2, shape=sig.shape)
        )
        * duration**0.5
        / 2
    )
    sig += noise
    # sig = noise
    return sig, parameters


def construct_numpyro_model(duration, sample_rate, bounds, phase_marginalization=True):
    times = segment_times(duration, sample_rate)

    def numpyro_model(event: np.ndarray):
        amplitude = sample("amplitude", Normal(*bounds["amplitude"]))
        frequency = sample("frequency", TruncatedNormal(*bounds["frequency"], low=FMIN, high=FMAX))
        bandwidth = sample("bandwidth", Normal(*bounds["bandwidth"]))
        if phase_marginalization:
            phase = deterministic("phase", 0.0)
        else:
            phase = sample("phase", Uniform(*bounds["phase"]))
        if "delta_t" in bounds:
            delta_t = sample("delta_t", Normal(*bounds["delta_t"]))
        else:
            delta_t = deterministic("delta_t", 0.0)
        parameters = BurstParameters(
            amplitude=amplitude,
            frequency=frequency,
            bandwidth=bandwidth,
            phase=phase,
            delta_t=delta_t,
        )
        template = signal(parameters, times=times, sample_rate=sample_rate).squeeze()
        factor(
            "ln_l",
            event_ln_likelihood(
                event=event,
                signals=nfft(template, sample_rate),
                duration=duration,
                phase_marginalization=phase_marginalization,
            ).squeeze(),
        )

    return numpyro_model


def create_jaxted_arguments(duration, sample_rate, bounds, phase_marginalization=True):
    times = segment_times(duration, sample_rate)

    boundary_fn = None

    def jaxted_ln_likelihood(parameters, *, event, **kwargs):
        parameters = BurstParameters(
            amplitude=parameters["amplitude"],
            frequency=parameters["frequency"],
            bandwidth=parameters["bandwidth"],
            phase=parameters["phase"],
            delta_t=parameters["delta_t"],
        )
        # print(parameters["amplitude"].ndim, parameters["amplitude"].shape)
        template = signal(
            parameters, times=times, sample_rate=sample_rate
        )
        return event_ln_likelihood(
            event=event,
            signals=nfft(template, sample_rate),
            duration=duration,
            phase_marginalization=phase_marginalization,
        ).squeeze()
    
    def truncated_normal_ln_prob(value, mu, sigma, low, high):
        return -0.5 * ((value - mu) / sigma) ** 2  + np.log(value >= low) + np.log(value <= high)
    
    def jaxted_ln_prior(parameters, *args, **kwargs):
        ln_prob = (
            truncated_normal_ln_prob(parameters["amplitude"], *bounds["amplitude"], -np.inf, np.inf)
            + truncated_normal_ln_prob(parameters["frequency"], *bounds["frequency"], FMIN, FMAX)
            + truncated_normal_ln_prob(parameters["bandwidth"], *bounds["bandwidth"], -np.inf, np.inf)
        )
        # ln_prob += np.log(parameters["frequency"] >= FMIN) + np.log(parameters["frequency"] <= FMAX)
        if not phase_marginalization:
            ln_prob += np.log(parameters["phase"] >= bounds["phase"][0]) + np.log(parameters["phase"] <= bounds["phase"][1])
        if "delta_t" in bounds:
            ln_prob += truncated_normal_ln_prob(parameters["delta_t"], *bounds["delta_t"], -np.inf, np.inf)
        return ln_prob

    def jaxted_sample_prior(n_samples, rng_key, *args, **kwargs):
        # rng_key = jax.random.PRNGKey(10000)
        lowf = (FMIN - bounds["frequency"][0]) / bounds["frequency"][1]
        highf = (FMAX - bounds["frequency"][0]) / bounds["frequency"][1]
        keys = jax.random.split(rng_key, 5)
        return dict(
            amplitude=jax.random.normal(keys[0], (n_samples,)) * bounds["amplitude"][1] + bounds["amplitude"][0],
            frequency=jax.random.truncated_normal(keys[1], lowf, highf, (n_samples,)) * bounds["frequency"][1] + bounds["frequency"][0],
            # frequency=jax.random.uniform(keys[1], (n_samples,), minval=FMIN, maxval=FMAX),
            bandwidth=jax.random.normal(keys[2], (n_samples,)) * bounds["bandwidth"][1] + bounds["bandwidth"][0],
            phase=jax.random.uniform(keys[3], (n_samples,), *bounds["phase"]) if not phase_marginalization else np.zeros((n_samples,)),
            delta_t=jax.random.normal(keys[4], (n_samples,)) * bounds["delta_t"][1] + bounds["delta_t"][1] if "delta_t" in bounds else np.zeros((n_samples,)),
            # amplitude=sample("amplitude", Normal(*bounds["amplitude"], shape=(n_samples,))),
            # frequency=sample("frequency", TruncatedNormal(*bounds["frequency"], low=FMIN, high=FMAX, shape=(n_samples,))),
            # bandwidth=sample("bandwidth", Normal(*bounds["bandwidth"], shape=(n_samples,))),
            # phase=sample("phase", Uniform(*bounds["phase"], shape=(n_samples,))) if not phase_marginalization else 0.0,
            # delta_t=sample("delta_t", Normal(*bounds["delta_t"], shape=(n_samples,))) if "delta_t" in bounds else 0.0,
        )

    return jaxted_ln_likelihood, jaxted_ln_prior, jaxted_sample_prior, boundary_fn        


@partial(jax.vmap, in_axes=(0, None, None, None, None, None, None))
def simulate_many(rng, mean, sigma, sample_rate, duration, times, bounds):
    """
    A wrapper around :func:`simulate_event` that generates multiple events
    from a Gaussian distribution of sine-Gaussian bursts in white Gaussian
    noise.

    Parameters
    ----------
    rng: jax.random.PRNGKey
        The random number generator key.
    mean: float
        The mean frequency of the population.
    sigma: float
        The standard deviation of the population.
    sample_rate: float
        The sample rate of the time series.
    duration: float
        The duration of the time series in seconds.
    times: ndarray
        The time array corresponding to the provided duration and
        sample_rate (see :func:`segment_times`).
    bounds: dict
        The prior bounds for the sine-Gaussian burst parameters.
    """
    events = simulate_event(
        rng,
        mean,
        sigma,
        bounds=bounds,
        times=times,
        sample_rate=sample_rate,
        duration=duration,
    )
    return events


@partial(jax.jit, static_argnames=("n_events", "time_align"))
def simulate_population(
    rng_key,
    basis,
    projection,
    mean,
    sigma,
    bounds,
    offset=0.3,
    threshold=0,
    n_events=1000,
    sample_rate=256,
    duration=4,
    times=segment_times(4, 256),
    time_align=True,
):
    """
    Simulate a population of sine-Gaussian bursts in white Gaussian noise
    obeying a signal-to-noise ratio (SNR) threshold and also project against a
    template bank.

    This draws half of the population from a Gaussian distribution with
    a mean frequency of :code:`mean - offset` and the other half from
    :code:`mean + offset` and then applies the selection cut based on the
    matched filter SNR using the provided basis.

    Parameters
    ----------
    rng_key: jax.random.PRNGKey
        The random number generator key.
    basis: ndarray
        The basis functions, shape
        (:code:`n_filters`, :code:`duration * sample_rate`).
    mean: float
        The mean frequency of the population.
    sigma: float
        The standard deviation of the population.
    bounds: dict
        The prior bounds for the sine-Gaussian burst parameters.
    offset: float
        The offset from the mean frequency for the two populations,
        default is 0.3.
    threshold: float
        The signal-to-noise ratio threshold for the selection cut,
        default is 0.
    n_events: int
        The number of events to simulate, default is 1000.
        Note that the actual number of events may be less than this
        due to the selection cut.
    sample_rate: float
        The sample rate of the time series, default is 256 Hz.
    duration: float
        The duration of the time series in seconds, default is 4 s.
    times: ndarray
        The time array corresponding to the provided duration and
        sample_rate (see :func:`segment_times`). Note that no checks are
        performed to ensure consistency.
    time_align: bool
        Whether to time-align the data to the SNR peak, default is True.
    """
    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
    keys = jax.random.split(subkey1, n_events // 2)
    events1, truths = simulate_many(
        keys, mean - offset, sigma, sample_rate, duration, times, bounds
    )
    keys = jax.random.split(subkey2, n_events // 2)
    events2, truths2 = simulate_many(
        keys, mean + offset, sigma, sample_rate, duration, times, bounds
    )
    events = np.concatenate([events1, events2])
    truths.append(truths2)
    td_truths = infft(events, sample_rate=sample_rate).T
    filtered, snrs, _ = apply_filter_multiple(
        td_truths, basis, projection, time_align=time_align
    )
    # jax.debug.print("{} {}", np.mean(snrs), np.std(snrs))
    keep = snrs > threshold
    return events, truths, keep, rng_key, filtered.T


def kl_divergence_filter(
    rng_key: jax.Array,
    sigma_frequency: np.ndarray,
    population_size: int,
    ntrials: int,
    basis: np.ndarray,
    bounds: dict,
    sample_rate: float,
    duration: float,
    threshold: float = 0,
    fpeaks: np.ndarray = np.linspace(4.8, 5.2, 11),
    time_align: bool = True,
    weights: np.ndarray = 1,
) -> np.ndarray:
    """
    Run a population simulation to estimate the KL divergence between an observed population
    with the specific mean_frequency and standard deviation and a range of mean frequencies.

    Parameters
    ==========
    rng_key: jax.random.PRNGKey
        A random key to use for the simulations
    sigma_frequency: float
        The standard deviation of the all simulated and observed populations
    population_size: int
        The number of events to simulate and compare
    ntrials: int
        The number of trials to run for each mean frequency
    basis: np.ndarray
        The filter to apply to the data, this is obtained using an SVD of expected signals
    bounds: dict
        Dictionary of min/max bounds for the parameters being varies
    sample_rate: float
        The sample rate of the data
    duration: float
        The duration of the data
    threshold: float
        The threshold to use for the matched filter SNR
    time_align: bool
        Whether to time-align the data to the SNR peak, default is True.
    """

    times = segment_times(duration, sample_rate)

    label = f"{sigma_frequency}_{threshold}"

    print(f"Loading data from data_{label}.pkl")
    with open(f"data_{label}.pkl", "rb") as fobj:
        data = dill.load(fobj)

    # signals_1 = infft(data, sample_rate=sample_rate).T
    # signals_1 = apply_filter_multiple(
    #     signals_1, projection, basis, time_align=time_align
    # )[0].T
    signals_1 = data
    target_population_size = signals_1.shape[0]

    print(f"Signals loaded, with shape {signals_1.shape}")

    simulate_kwargs = dict(
        basis=basis,
        projection=weights,
        bounds=bounds,
        times=times,
        duration=duration,
        sample_rate=sample_rate,
        time_align=time_align,
    )

    def inner_func(fpeak, rng_key):
        _, _, keep, _, signals_2 = simulate_population(
            rng_key,
            mean=fpeak,
            sigma=sigma_frequency,
            offset=0.0,
            threshold=threshold,
            n_events=population_size,
            **simulate_kwargs,
        )
        signals_2 = signals_2[keep][:target_population_size]
        dimensions = signals_2.shape[-1]
        kl_div = kl_divergence_from_signals(signals_1, signals_2, dimensions=dimensions)
        pbar.update()
        return kl_div, None

    kl_divs = list()
    pbar = tqdm(total=len(fpeaks) * ntrials)
    for fpeak in fpeaks:
        pbar.set_description(f"Running for fpeak={fpeak:.2f}")
        rng_key, *subkeys = jax.random.split(rng_key, ntrials + 1)
        temp_1, _ = zip(*(inner_func(fpeak, key) for key in subkeys))
        kl_divs.append(np.array(temp_1))
    pbar.close()

    return np.array(kl_divs)


def load_basis(filename: str, truncation: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, weights, basis = np.load(filename, allow_pickle=True)
    basis = basis[:truncation].conjugate()
    projection = u[:, :truncation] * weights[:truncation]
    print(projection.shape, basis.shape)
    # return basis * weights[:, None], weights
    return basis, projection
