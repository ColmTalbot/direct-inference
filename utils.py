from dataclasses import dataclass
from functools import partial

import jax.numpy as np
from jax import jit, vmap
from jax.lax import cond
from jax.tree_util import register_pytree_node
from jax.scipy.linalg import toeplitz
from jax.scipy.special import i0e
from jax.scipy.stats import sem
from sklearn.neighbors import NearestNeighbors


@dataclass
class Parameters:
    """
    Base class for storing parameters in a JAX-friendly way.

    This supports PyTree flattening and unflattening, and provides
    methods fro slicing, appending, iteration, and accessing the individual
    parameters.

    This can be subclassed for a model with two variables (:code:`foo`
    and :code:`bar`) as follows:

    .. code-block:: python

        >>> @dataclass
        >>> class MyParameters(Parameters):
        >>>     foo: np.ndarray
        >>>     bar: np.ndarray

        >>> test = MyParameters(foo=np.array([1, 2, 3]), bar=np.array([4, 5, 6]))
        >>> print(test[0])
        MyParameters(foo=array(1), bar=array(4))

        >>> print(test[1:])
        MyParameters(foo=array([2, 3]), bar=array([5, 6]))

        >>> print(test.foo)
        array([1, 2, 3])

        >>> print(test["foo"])
        array([1, 2, 3])

        >>> for value in test:
        >>>     print(value)
        MyParameters(foo=array(1), bar=array(4))
        MyParameters(foo=array(2), bar=array(5))
        MyParameters(foo=array(3), bar=array(6))

        >>> other = MyParameters(foo=np.array([7, 8, 9]), bar=np.array([10, 11, 12]))
        >>> test.append(other)
        >>> print(test)
        MyParameters(foo=array([1, 2, 3, 7, 8, 9]), bar=array([4, 5, 6, 10, 11, 12]))
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls._tree_flatten, cls._tree_unflatten)

    def __getitem__(self, idx: str | slice) -> np.ndarray:
        if isinstance(idx, (int, slice)):
            kwargs = {key: value[idx] for key, value in self.__dict__.items()}
            return self.__class__(**kwargs)
        else:
            return getattr(self, idx)

    @property
    def _reference(self) -> np.ndarray:
        return list(self.__dict__.values())[0]

    def __len__(self) -> int:
        return len(self._reference)

    @property
    def shape(self) -> tuple:
        return self._reference.shape

    def append(self, other: "Parameters") -> None:
        this = self.__dict__
        that = other.__dict__
        for key in self.__dict__:
            setattr(self, key, np.concatenate([this[key], that[key]]))

    def __iter__(self) -> "Parameters":
        for idx in range(len(self)):
            yield self[idx]

    def _tree_flatten(self) -> tuple[tuple, dict]:
        children = tuple(self.__dict__.values())
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data: dict, children: tuple) -> "Parameters":
        return cls(*children, **aux_data)


def segment_times(duration: float, sample_rate: float) -> np.ndarray:
    """
    Compute the times for a segment of data centered on zero,
    :math:`[-T/2, -T/2 + \delta T, ..., T/2)`.

    Parameters
    ==========
    duration: float
        The duration of the segment in seconds (:math:`T`).
    sample_rate: float
        The sample rate of the data in Hz (:math:`1 / \delta T`).
    """
    return np.linspace(
        -duration // 2,
        duration // 2,
        sample_rate * duration,
        endpoint=False,
    )


def apply_filter_multiple(data, projection, basis, time_align=True):
    """
    Apply the matched filter bank to multiple time series.

    This function maximizes optimal signal-to-noise ratio (SNR) for
    each time series over time shifts within a narrow window.

    After realigning the time series so that SNR peak is aligned
    the time series is projected against the basis functions.

    Parameters
    ----------
    data: ndarray
        The time series data to filter, shape
        (:code:`n_events`, :code:`duration * sample_rate`).
    projection: ndarray
        The projection matrix, see :func:`projection_matrix`, shape
        (:code:`n_time_shift_filters`, :code:`time_window`,
        :code:`duration * sample_rate`).
    basis: ndarray
        The basis functions, shape
        (:code:`n_filters`, :code:`duration * sample_rate`).
    time_align: bool
        Whether to time-align the data to the SNR peak, default is True.

    Returns
    -------
    filtered: ndarray
        The filtered time series, shape (:code:`n_events`, :code:`n_filters`).
    snr: ndarray
        The optimal signal-to-noise ratio of the filtered time series, shape
        (:code:`n_events`,).
    delta: ndarray
        The time shift (in number of indices) that maximizes the signal-to-noise
        ratio, shape (:code:`n_events`,).
    """
    if time_align:
        U = projection @ data / len(data)
        snr_time_series = (abs(U) ** 2).sum(axis=0)
        delta = -np.argmax(snr_time_series, axis=0)
        data = vmap(np.roll, in_axes=(1, 0))(data, delta).T
    else:
        delta = 0
    filtered = 2 * basis.conjugate() @ data
    snr = (abs(filtered) ** 2).sum(axis=0) / len(data)
    filtered = filtered.real
    return filtered, snr**0.5, delta


@jit
def _population_weights(posteriors, mean_frequency, sigma_frequency):
    r"""
    Calculate the population weights for a set of posteriors assuming a
    Gaussian population model. The original prior is assumed to be uniform
    over :math:`[1, 9]`.

    .. math::

        w_i = \frac{\exp\left(-\frac{1}{2}\left(\frac{p_i - \mu}{\sigma}\right)^2\right)}
        {8\sqrt{2\pi}\sigma}

    Parameters
    ----------
    posteriors: ndarray
        The posteriors of the population, suggetsed shape
        (:code:`n_events`, :code:`n_samples`).
    mean_frequency: float
        The mean frequency of the population, :math:`\mu`.
    sigma_frequency: float
        The standard deviation of the population, :math:`\sigma`.
    """
    original_prior = 1 / (9 - 1)
    pop_weights = np.exp(-0.5 * (posteriors - mean_frequency) ** 2 / sigma_frequency**2)
    pop_weights /= (2 * np.pi) ** 0.5 * sigma_frequency
    return pop_weights / original_prior


@jit
def population_ln_l(posteriors, mean_frequency, sigma_frequency):
    r"""
    Calculate the population log-likelihood for a set of posteriors assuming a
    Gaussian population model. The original prior is assumed to be uniform
    over :math:`[1, 9]`.

    .. math::

        \ln {\cal L} = \sum_{i}^{N_{\rm events}}
        \ln \left( \sum_{j}^{N_{\rm samples}} w_{ij} \right)

    see :func:`_population_weights` for the definition of :math:`w_{ij}`.

    Parameters
    ----------
    posteriors: ndarray
        The posteriors of the population, shape
        (:code:`n_samples`, :code:`n_events`).
    mean_frequency: float
        The mean frequency of the population, :math:`\mu`.
    sigma_frequency: float
        The standard deviation of the population, :math:`\sigma`.
    """
    weights = _population_weights(posteriors, mean_frequency, sigma_frequency)
    return np.sum(np.log(np.mean(weights, axis=0)))


@jit
def population_sigma_ln_l(posteriors, mean_frequency, sigma_frequency):
    r"""
    Calculate the standard deviation in the population log-likelihood for a set
    of posteriors assuming a Gaussian population model. The original prior is
    assumed to be uniform over :math:`[1, 9]`.

    .. math::

        \sigma_{\ln {\cal L}} = \left( \sum_{i}^{N_{\rm events}}
        \frac{\sigma^{2}_{w_i}}{\mu_{w_i}^{2}} \right)^{1/2}

    see :func:`_population_weights` for the definition of :math:`w_{ij}`.

    Parameters
    ----------
    posteriors: ndarray
        The posteriors of the population, shape
        (:code:`n_samples`, :code:`n_events`).
    mean_frequency: float
        The mean frequency of the population, :math:`\mu`.
    sigma_frequency: float
        The standard deviation of the population, :math:`\sigma`.
    """
    weights = _population_weights(posteriors, mean_frequency, sigma_frequency)
    return np.sum(np.var(weights, axis=0) / np.mean(weights, axis=0) ** 2) ** 0.5


@jit
def approximate_divergence(posteriors, mean_frequency, sigma_frequency):
    r"""
    Calculate the approximate divergence for a set of posteriors assuming a
    Gaussian population model. The original prior is assumed to be uniform
    over :math:`[1, 9]`.

    .. math::

        D_{\rm KL} = -\frac{\ln {\cal L}}{N_{\rm events}}

    see :func:`population_ln_l` for the definition of :math:`\ln {\cal L}`.

    Parameters
    ----------
    posteriors: ndarray
        The posteriors of the population, shape
        (:code:`n_samples`, :code:`n_events`).
    mean_frequency: float
        The mean frequency of the population, :math:`\mu`.
    sigma_frequency: float
        The standard deviation of the population, :math:`\sigma`.
    """
    weights = _population_weights(posteriors, mean_frequency, sigma_frequency)
    return -np.mean(np.log(np.mean(weights, axis=0)))


@jit
def approximate_sigma_divergence(posteriors, mean_frequency, sigma_frequency):
    """
    Calculate the standard deviation in the approximate divergence for a set
    of posteriors assuming a Gaussian population model. The original prior is
    assumed to be uniform over :math:`[1, 9]`.

    .. math::

        \sigma_{D_{\rm KL}} = \frac{\sigma_{\ln {\cal L}}}{N_{\rm events}}

    see :func:`population_sigma_ln_l` for the definition of :math:`\sigma_{\ln {\cal L}}`.

    Parameters
    ----------
    posteriors: ndarray
        The posteriors of the population, shape
        (:code:`n_samples`, :code:`n_events`).
    mean_frequency: float
        The mean frequency of the population, :math:`\mu`.
    sigma_frequency: float
        The standard deviation of the population, :math:`\sigma`.
    """
    return population_sigma_ln_l(posteriors, mean_frequency, sigma_frequency) / len(
        posteriors
    )


@jit
def ln_i0(value: np.ndarray) -> np.ndarray:
    """
    A numerically stable method to evaluate ln(I_0) a modified Bessel function
    of order 0 used in the phase-marginalized likelihood.

    Parameters
    ==========
    value: array-like
        Value(s) at which to evaluate the function

    Returns
    =======
    array-like:
        The natural logarithm of the bessel function
    """
    value = np.abs(value)
    return np.log(i0e(value)) + value


@jit
def event_ln_likelihood(
    event: np.ndarray,
    signals: np.ndarray,
    duration: float = 4,
    phase_marginalization: bool = True,
) -> np.ndarray:
    r"""
    Calculate the standard gravitational-wave transient likelihood
    for the provided signals.

    .. math::

        \ln {\cal L} = - \frac{2}{T} \sum_{f_i}
        \left< h | h \right> - 2 {\cal R} \left< d | h \right>

    where :math:`T` is the duration of the signal, :math:`f_i` are the
    frequency bins, :math:`h` is the signal, :math:`d` is the data, and
    :math:`{\cal R}` is the real part.

    We also implement marginalization over the orbital phase using a Bessel
    function of the first kind.

    .. math::

        \ln {\cal L} = - \frac{2}{T} \sum_{f_i}
        \left< h | h \right> - 2 \ln I_{0}(|\left< d | h \right>|)

    Parameters
    ----------
    event: ndarray
        The event to analyze, shape (:code:`n_frequencies`,).
    signals: ndarray
        The signal templates, shape
        (:code:`n_templates`, :code:`n_frequencies`).
    duration: float
        The duration of the signal, default is 4.
    phase_marginalization: bool
        Whether to marginalize over the orbital phase, default is True.
    """
    correction = 4 / duration
    d_inner_h = np.sum(event * signals.conjugate(), axis=-1)
    d_inner_h = cond(phase_marginalization, ln_i0, np.real, d_inner_h)
    h_inner_h = np.sum(abs(signals) ** 2, axis=-1)

    ln_ls = correction * (d_inner_h - h_inner_h / 2)
    return ln_ls


@partial(vmap, in_axes=(0, 0, None))
def project_single(base, weight, width=100):
    r"""
    Create a single entry in the projection matrix for a given basis function.
    The entry is a Toeplitz matrix that will perform a narrow element of a
    convolution symmetrically around zero lag.

    .. math::

        P_{ij} = \frac{\sqrt{w}}{N} B_{\frac{T}{2} + i - j}

    where :math:`B` is the basis function, :math:`w` is the weight, :math:`N`
    is the length of the basis function, and :math:`T` is the width of the
    time window for maximization (:code:`time_window`).

    Parameters
    ----------
    base: ndarray
        The basis function, shape (:code:`n_time_shift_filters`,).
    weight: float
        The weight for the basis function
    width: int
        The width of the time window for maximization (:math:`T`),
        default is 100.
    """
    return weight**0.5 * toeplitz(np.roll(base, -width // 2))[:width] / len(base)


def projection_matrix(basis, weights, truncation=10, width=100):
    r"""
    Construct the truncated projection matrix for the template bank.
    This just uses the first :code:`truncation` basis functions.

    .. math::

        P_{ijk} = \frac{\sqrt{w_{i}}}{N} B_{i, \frac{T}{2} + j - k}

    where :math:`B_{i}` is the basis, :math:`w_{i}` are the per-element weights,
    :math:`N` is the length of an individual basis element, and :math:`T` is the
    width of the time window for maximization (:code:`time_window`).

    Parameters
    ----------
    basis: ndarray
        The basis functions, shape
        (:code:`n_filters`, :code:`duration * sample_rate`).
    weights: ndarray
        The weights for the basis functions, shape (:code:`n_filters`,).
    truncation: int
        The number of basis functions to use (:code:`n_time_shift_filters`),
        default is 10.
    width: int
        The width of the time window for maximization (:math:`T`),
        default is 100.
    """
    projection = project_single(basis[:truncation], weights[:truncation], width)
    return projection


@jit
def infft(input: np.ndarray, sample_rate: float = 256, axis: int = -1) -> np.ndarray:
    r"""
    Compute a normalized inverse real FFT of the input data.

    .. math::

        x = \mathcal{F}^{-1}_{\cal R}(\hat{x}) * \delta T

    Parameters
    ----------
    input: ndarray
        The input data, shape (:code:`...`, :code:`duration * sample_rate / 2 + 1`).
    sample_rate: float
        The sample rate of the data in Hz (:math:`1 / \delta T`), default is 256.
    axis: int
        The axis along which to perform the operation, default is -1.

    Returns
    -------
    ndarray:
        The inverse real FFT of the input data, shape
        (:code:`...`, :code:`duration * sample_rate`).
    """
    return np.fft.irfft(input, axis=axis) * sample_rate


@jit
def nfft(input: np.ndarray, sample_rate: float = 256, axis: int = -1) -> np.ndarray:
    r"""
    Compute a normalized real FFT of the input data.

    .. math::

        \hat{x} = \mathcal{F}_{\cal R}(x) / \delta T

    Parameters
    ----------
    input: ndarray
        The input data, shape (:code:`...`, :code:`duration * sample_rate`).
    sample_rate: float
        The sample rate of the data in Hz (:math:`1 / \delta T`), default is 256.
    axis: int
        The axis along which to perform the operation, default is -1.

    Returns
    -------
    ndarray:
        The real FFT of the input data, shape
        (:code:`...`, :code:`duration * sample_rate / 2 + 1`).
    """
    return np.fft.rfft(input, axis=axis) / sample_rate


@jit
def whiten(input: np.ndarray, psd: np.ndarray) -> np.ndarray:
    r"""
    Whiten the input timeseries via two FFTs.

    .. math::

        x = \mathcal{F}^{-1}_{\cal R}\left(\frac{\mathcal{F}_{\cal R}(x)}{\sqrt{P}}\right)

    Parameters
    ----------
    input: ndarray
        The input data, shape (:code:`...`, :code:`duration * sample_rate`).
    psd: ndarray
        The power spectral density of the data, :math:`P`, shape
        (:code:`...`, :code:`duration * sample_rate / 2 + 1`).

    Returns
    -------
    ndarray:
        The whitened timeseries, shape (:code:`...`, :code:`duration * sample_rate`).
    """
    return np.fft.irfft(np.fft.rfft(input) / psd**0.5)


def kl_divergence_from_signals(
    signals_1: np.ndarray, signals_2: np.ndarray, dimensions: int
) -> np.ndarray:
    n_neighbours = 30

    true = signals_1
    other = signals_2

    true = true.real
    other = other.real
    nbrs = NearestNeighbors(n_neighbors=n_neighbours).fit(true)
    r_k = nbrs.kneighbors()[0].T
    nbrs = NearestNeighbors(n_neighbors=n_neighbours).fit(other)
    s_k = nbrs.kneighbors(true)[0].T

    nn = len(true)
    mm = len(other)

    _divs = -dimensions * np.nanmean(
        np.nan_to_num(np.log(r_k / s_k), neginf=np.nan, posinf=np.nan), axis=-1
    ) + np.log(mm / (nn - 1))
    return np.mean(_divs)


def make_plots(
    fpeaks: np.ndarray,
    *divergences: np.ndarray,
    mode: str = "KL",
    xlabel: str = "Population mean frequency [Hz]",
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for ii, quantity in enumerate(divergences):
        if len(divergences) == 2:
            label = ["Approximation", "Standard"][ii]
        else:
            label = ["Approximation", None, "Traditional"][ii]
        if mode == "events":
            quantity = 1 / quantity
        plt.errorbar(
            fpeaks,
            np.nanmean(quantity, axis=-1),
            yerr=sem(quantity, axis=-1, nan_policy="omit"),
            label=label,
            color=f"C{ii}",
        )
        plt.fill_between(
            fpeaks,
            np.percentile(quantity, 5, axis=-1),
            np.percentile(quantity, 95, axis=-1),
            alpha=0.3,
            color=f"C{ii}",
        )
        plt.plot(
            fpeaks, quantity, alpha=min(0.2, 1 / quantity.shape[-1]), color=f"C{ii}"
        )
    if mode == "events":
        plt.ylabel("Number of events")
    else:
        plt.ylabel("KL Divergence")
    plt.xlabel(xlabel)
    plt.xlim(fpeaks[0], fpeaks[-1])
    if len(divergences) > 0:
        plt.legend(loc="best")
    plt.show()
    plt.close()
