"""The PMT's after-tracking components: time spread, delay, time walk, charge, threshold, pulse.

Each component is a frozen dataclass holding only structure (its form, knot positions, table lengths); every number
it uses is a value passed in at call time, a dict ``p`` of the component's fields (``fields()``: name -> unit, shape),
so a value never recompiles and always carries its derivative. Times are in ns, charges in photoelectrons (pe).

Time spread (TTS), per photoelectron:
  ``gaussian(sigma)``; ``emg(sigma, tau, shift)``: a Gaussian convolved with an exponential tail, shifted;
  ``table(pdf)``: a measured histogram on fixed bins, uniform within each bin.
  ``cdf(p, x, profile)`` / ``sf(p, x, profile)``: the distribution of the arrival-to-photoelectron delay, convolved
  with a scintillation emission profile ``(fractions, decay times, rise times)`` (``None``: prompt light); only the
  Gaussian form combines with a profile in closed form, the others refuse one. ``support(ranges)`` is the static
  time range the kernel can reach for any value inside its declared ranges.
Delay (``delay``): a fixed-position delay table (a wavelength shifter's emission time), probabilities as values.
Time walk (``time_walk``): a per-hit shift and extra Gaussian spread against the hit's charge, linear in fixed
  charge knots and constant beyond them.
Charge (SPE): ``gauss_exp(sigma, fraction, tau)``: a Gaussian main peak plus an exponential under-amplified part,
  mean 1 pe by construction; ``table(pdf)``: a measured spectrum on fixed bins. ``pmf(p, edges)`` the masses on a
  charge grid, ``moments(p)`` (mean, mean square), ``sample(rng, p, n)``.
Threshold: ``table(accept)`` acceptance against charge knots; ``poly(coefficients, offset, limit)``: acceptance
  min(1, poly(Q + offset)) below ``limit`` and 1 above (WCSim's SKI form), clipped to [0, 1].
Pulse (the single-photoelectron waveform, unit peak): ``bigauss(t_peak, sigma_rise, sigma_fall)``,
  ``exp2(t_start, tau_rise, tau_fall)``, ``table(shape)`` samples on a fixed time step, linear between them.

Notation: x a time lag (ns), q a charge (pe), F a CDF, S a survival function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import log_ndtr, ndtr

SPREAD = 8.0  # Gaussian widths a time kernel reaches: below 1e-15 of its mass lies beyond
FWHM_PER_SIGMA = 2.3548200450309493  # 2 sqrt(2 ln 2)


# ------------------------------------------------------------------------------- scintillation profile (exact)
def _tail(x, s, tau):
    """Return tau exp(-x/tau + s^2/2tau^2) Phi(x/s - s/tau), 0 for tau = 0 (formed with log_ndtr, no overflow)."""
    on = tau > 0.0
    tz = jnp.where(on, tau, 1.0)
    e = jnp.exp(-x / tz + 0.5 * (s / tz) ** 2 + log_ndtr(x / s - s / tz))
    return jnp.where(on, tz * e, 0.0)


def component_cdf(x, tau_d, tau_r, sigma):
    """Return the CDF at x of one emission component (density (e^-t/tau_d - e^-t/tau_r)/(tau_d - tau_r), t >= 0;
    tau_r = 0 a pure exponential, tau_d = 0 prompt) convolved with a Gaussian of width sigma (ns).
    """  # noqa: D205
    s = jnp.maximum(sigma, 1e-6)
    phi = ndtr(x / s)
    den = jnp.where(tau_d > 0.0, tau_d - tau_r, 1.0)
    return jnp.where(tau_d > 0.0, phi - (_tail(x, s, tau_d) - _tail(x, s, tau_r)) / den, phi)


def component_sf(x, tau_d, tau_r, sigma):
    """Return 1 - component_cdf without the cancellation (accurate deep in the right tail)."""
    s = jnp.maximum(sigma, 1e-6)
    phic = ndtr(-x / s)
    den = jnp.where(tau_d > 0.0, tau_d - tau_r, 1.0)
    return jnp.where(tau_d > 0.0, phic + (_tail(x, s, tau_d) - _tail(x, s, tau_r)) / den, phic)


PROMPT = ((1.0,), (0.0,), (0.0,))  # an emission profile of prompt light


def profile_cdf(x, profile, sigma):
    """Return the CDF of a profile mixture ``(fractions, tau_d, tau_r)`` convolved with a Gaussian of width sigma."""
    f, td, tr = (jnp.asarray(a, jnp.float32) for a in profile)
    return jnp.sum(f * component_cdf(x[..., None], td, tr, jnp.asarray(sigma)[..., None]), -1)


def profile_sf(x, profile, sigma):
    """Return 1 - profile_cdf, stable in the right tail."""
    f, td, tr = (jnp.asarray(a, jnp.float32) for a in profile)
    return jnp.sum(f * component_sf(x[..., None], td, tr, jnp.asarray(sigma)[..., None]), -1)


def _interp_cdf(x, edges, pdf):
    """Return the CDF at x of a histogram (masses ``pdf`` on fixed ``edges``, uniform within bins), normalised."""
    c = jnp.concatenate([jnp.zeros(1, pdf.dtype), jnp.cumsum(pdf)])
    return jnp.interp(x, jnp.asarray(edges, jnp.float32), c / c[-1])


def _refuse_profile(name, profile):
    if profile is not None:
        raise ValueError(f"time spread {name!r}: a scintillation emission profile combines in closed form only with "
                         "the 'gaussian' time spread; use form 'gaussian' for scintillation light")  # fmt: skip


# ------------------------------------------------------------------------------------------------ time spread
@dataclass(frozen=True)
class GaussianTTS:
    """A Gaussian transit-time spread of width ``sigma`` (ns)."""

    form: ClassVar[str] = "gaussian"

    def fields(self):
        """Return the field units and shapes."""
        return {"sigma": ("ns", ())}

    def cdf(self, p, x, profile=None):
        """Return the delay CDF at x (with the emission profile, when given)."""
        return profile_cdf(x, profile or PROMPT, p["sigma"])

    def sf(self, p, x, profile=None):
        """Return the delay survival function at x."""
        return profile_sf(x, profile or PROMPT, p["sigma"])

    def widened(self, p, extra):
        """Return the values with an extra Gaussian spread ``extra`` (ns) added in quadrature."""
        return {**p, "sigma": jnp.sqrt(p["sigma"] ** 2 + extra**2)}

    def moments(self, p):
        """Return (mean, variance) in ns, ns^2."""
        return 0.0, p["sigma"] ** 2

    def sample(self, rng, p, n):
        """Return n draws (numpy)."""
        return rng.normal(0.0, float(p["sigma"]), n)

    def support(self, ranges):
        """Return the static (lo, hi) lag range in ns over the declared ranges."""
        s = SPREAD * ranges["sigma"][1]
        return -s, s

    def min_width(self, ranges):
        """Return the narrowest width (ns) inside the declared ranges, which sets the time-node spacing."""
        return ranges["sigma"][0]


@dataclass(frozen=True)
class EMGTTS(GaussianTTS):
    """A Gaussian (``sigma``) convolved with an exponential tail (``tau``), shifted by ``shift`` (ns)."""

    form: ClassVar[str] = "emg"

    def fields(self):
        """Return the field units and shapes."""
        return {"sigma": ("ns", ()), "tau": ("ns", ()), "shift": ("ns", ())}

    def cdf(self, p, x, profile=None):
        """Return the delay CDF at x."""
        _refuse_profile(self.form, profile)
        return component_cdf(x - p["shift"], p["tau"], 0.0, p["sigma"])

    def sf(self, p, x, profile=None):
        """Return the delay survival function at x."""
        _refuse_profile(self.form, profile)
        return component_sf(x - p["shift"], p["tau"], 0.0, p["sigma"])

    def moments(self, p):
        """Return (mean, variance) in ns, ns^2."""
        return p["shift"] + p["tau"], p["sigma"] ** 2 + p["tau"] ** 2

    def sample(self, rng, p, n):
        """Return n draws (numpy)."""
        return float(p["shift"]) + rng.normal(0.0, float(p["sigma"]), n) + rng.exponential(float(p["tau"]), n)

    def support(self, ranges):
        """Return the static (lo, hi) lag range in ns."""
        s = SPREAD * ranges["sigma"][1]
        return ranges["shift"][0] - s, ranges["shift"][1] + s + 2.0 * SPREAD * ranges["tau"][1]


@dataclass(frozen=True)
class TableTTS:
    """A measured time spread: masses ``pdf`` on the fixed bin edges ``edges`` (ns), uniform within bins."""

    edges: tuple
    form: ClassVar[str] = "table"

    def fields(self):
        """Return the field units and shapes."""
        return {"pdf": ("", (len(self.edges) - 1,))}

    def cdf(self, p, x, profile=None):
        """Return the delay CDF at x."""
        _refuse_profile(self.form, profile)
        return _interp_cdf(x, self.edges, p["pdf"])

    def sf(self, p, x, profile=None):
        """Return the delay survival function at x."""
        return 1.0 - self.cdf(p, x, profile)

    def widened(self, p, extra):
        """Return the values unchanged: a table's shape is not widened (the time walk's spread is not applied)."""
        return p

    def moments(self, p):
        """Return (mean, variance) in ns, ns^2."""
        e = jnp.asarray(self.edges)
        c, w = 0.5 * (e[1:] + e[:-1]), p["pdf"] / jnp.sum(p["pdf"])
        m = jnp.sum(w * c)
        return m, jnp.sum(w * ((c - m) ** 2 + (e[1:] - e[:-1]) ** 2 / 12.0))

    def sample(self, rng, p, n):
        """Return n draws (numpy)."""
        return _table_sample(rng, self.edges, p["pdf"], n)

    def support(self, ranges):
        """Return the static (lo, hi) lag range in ns: the table's edges."""
        return self.edges[0], self.edges[-1]

    def min_width(self, ranges):
        """Return the narrowest bin width (ns)."""
        return float(np.min(np.diff(self.edges))) * 4.0


def _table_sample(rng, edges, pdf, n):
    pdf = np.clip(np.asarray(pdf, np.float64), 0.0, None)
    i = rng.choice(len(pdf), size=n, p=pdf / pdf.sum())
    e = np.asarray(edges, np.float64)
    return e[i] + rng.random(n) * (e[i + 1] - e[i])


@dataclass(frozen=True)
class DelayTable:
    """A fixed delay table: probabilities ``pdf`` (values) over bins of fixed ``edges`` (ns, >= 0).

    Applied to the time histogram's node masses by re-splitting each bin's sub-points over the nodes with the
    histogram's own cubic weights (exact to the node grid's accuracy).
    """

    edges: tuple
    sub: int = 8  # sub-points per bin
    form: ClassVar[str] = "table"

    def fields(self):
        """Return the field units and shapes."""
        return {"pdf": ("", (len(self.edges) - 1,))}

    def taps_matrix(self, dt):
        """Return (first tap offset, M) with taps = M @ pdf (numpy, static) for a node spacing dt (ns)."""
        e = np.asarray(self.edges, np.float64)
        u = e[:-1, None] + (np.arange(self.sub) + 0.5)[None, :] / self.sub * np.diff(e)[:, None]
        x = u / dt
        j = np.floor(x).astype(int)
        g = x - j
        w = [(-1, -g * (g - 1) * (g - 2) / 6), (0, (g + 1) * (g - 1) * (g - 2) / 2),
             (1, -(g + 1) * g * (g - 2) / 2), (2, (g + 1) * g * (g - 1) / 6)]  # fmt: skip
        off0 = int(j.min()) - 1
        m = np.zeros((int(j.max()) + 2 - off0 + 1, len(e) - 1))
        for o, wo in w:
            for b in range(len(e) - 1):
                np.add.at(m[:, b], j[b] + o - off0, wo[b] / self.sub)
        return off0, m

    def apply(self, p, nodes, dt):
        """Return the node masses (..., nt) after the delay; mass moved past the last node is dropped."""
        off0, m = self.taps_matrix(dt)
        taps = jnp.asarray(m, jnp.float32) @ (p["pdf"] / jnp.sum(p["pdf"]))
        nt = nodes.shape[-1]
        n = 1 << int(np.ceil(np.log2(nt + len(taps) + abs(off0))))
        full = jnp.fft.irfft(jnp.fft.rfft(nodes, n) * jnp.fft.rfft(taps, n), n)
        if off0 >= 0:
            return jnp.concatenate([jnp.zeros(nodes.shape[:-1] + (off0,), nodes.dtype), full[..., : nt - off0]], -1)
        return full[..., -off0 : -off0 + nt]

    def sample(self, rng, p, n):
        """Return n draws (numpy)."""
        return _table_sample(rng, self.edges, p["pdf"], n)

    def support(self, ranges):
        """Return the static (lo, hi) delay range in ns."""
        return self.edges[0], self.edges[-1]


@dataclass(frozen=True)
class TimeWalk:
    """A per-hit time shift ``shift`` and extra Gaussian spread ``sigma`` (ns) at fixed charge knots ``knots`` (pe)."""

    knots: tuple
    form: ClassVar[str] = "table"

    def fields(self):
        """Return the field units and shapes."""
        k = (len(self.knots),)
        return {"shift": ("ns", k), "sigma": ("ns", k)}

    def at(self, p, q):
        """Return (shift, sigma) at hit charges q (pe), constant beyond the end knots."""
        k = jnp.asarray(self.knots, jnp.float32)
        return jnp.interp(q, k, p["shift"]), jnp.interp(q, k, p["sigma"])


# ------------------------------------------------------------------------------------------------ charge (SPE)
@dataclass(frozen=True)
class GaussExpCharge:
    """A Gaussian main peak (width ``sigma``, pe) plus an exponential part (``fraction``, mean ``tau``), mean 1 pe."""

    form: ClassVar[str] = "gauss_exp"

    def fields(self):
        """Return the field units and shapes."""
        return {"sigma": ("pe", ()), "fraction": ("", ()), "tau": ("pe", ())}

    def _peak(self, p):
        return (1.0 - p["fraction"] * p["tau"]) / (1.0 - p["fraction"])

    def cdf(self, p, q):
        """Return the charge CDF at q (pe)."""
        w, tau = p["fraction"], p["tau"]
        expo = jnp.where(q > 0.0, -jnp.expm1(-jnp.maximum(q, 0.0) / jnp.maximum(tau, 1e-6)), 0.0)
        return (1.0 - w) * ndtr((q - self._peak(p)) / p["sigma"]) + w * expo

    def pmf(self, p, edges):
        """Return the masses on charge bins of the given edges (pe)."""
        c = self.cdf(p, jnp.asarray(edges, jnp.float32))
        return c[..., 1:] - c[..., :-1]

    def moments(self, p):
        """Return (mean, mean square) in pe, pe^2."""
        w, tau, m = p["fraction"], p["tau"], self._peak(p)
        return 1.0, (1.0 - w) * (p["sigma"] ** 2 + m**2) + w * 2.0 * tau**2

    def sample(self, rng, p, n):
        """Return n draws (numpy)."""
        exp = rng.random(n) < float(p["fraction"])
        g = rng.normal(float(self._peak(p)), float(p["sigma"]), n)
        return np.where(exp, rng.exponential(float(p["tau"]), n), g)


@dataclass(frozen=True)
class TableCharge:
    """A measured single-photoelectron spectrum: masses ``pdf`` on fixed bin ``edges`` (pe), uniform within bins."""

    edges: tuple
    form: ClassVar[str] = "table"

    def fields(self):
        """Return the field units and shapes."""
        return {"pdf": ("", (len(self.edges) - 1,))}

    def cdf(self, p, q):
        """Return the charge CDF at q (pe)."""
        return _interp_cdf(q, self.edges, p["pdf"])

    def pmf(self, p, edges):
        """Return the masses on charge bins of the given edges (pe)."""
        c = self.cdf(p, jnp.asarray(edges, jnp.float32))
        return c[..., 1:] - c[..., :-1]

    def moments(self, p):
        """Return (mean, mean square) in pe, pe^2."""
        e = jnp.asarray(self.edges, jnp.float32)
        w = p["pdf"] / jnp.sum(p["pdf"])
        lo, hi = e[:-1], e[1:]
        return jnp.sum(w * 0.5 * (lo + hi)), jnp.sum(w * (lo * lo + lo * hi + hi * hi) / 3.0)

    def sample(self, rng, p, n):
        """Return n draws (numpy)."""
        return _table_sample(rng, self.edges, p["pdf"], n)


# ------------------------------------------------------------------------------------------------ threshold
@dataclass(frozen=True)
class TableThreshold:
    """Acceptance ``accept`` against fixed charge knots ``knots`` (pe), linear, constant beyond the end knots."""

    knots: tuple
    form: ClassVar[str] = "table"

    def fields(self):
        """Return the field units and shapes."""
        return {"accept": ("", (len(self.knots),))}

    def accept(self, p, q):
        """Return the acceptance at integrated charges q (pe)."""
        return jnp.clip(jnp.interp(q, jnp.asarray(self.knots, jnp.float32), p["accept"]), 0.0, 1.0)


@dataclass(frozen=True)
class PolyThreshold:
    """Acceptance min(1, sum_k c_k x^k), x = Q + ``offset``, for x below ``limit``; 1 above; clipped to [0, 1]."""

    order: int
    form: ClassVar[str] = "poly"

    def fields(self):
        """Return the field units and shapes."""
        return {"coefficients": ("", (self.order + 1,)), "offset": ("pe", ()), "limit": ("pe", ())}

    def accept(self, p, q):
        """Return the acceptance at integrated charges q (pe)."""
        x = q + p["offset"]
        c = p["coefficients"]
        poly = c[-1] * jnp.ones_like(x)
        for k in range(self.order - 1, -1, -1):
            poly = c[k] + x * poly
        return jnp.where(x < p["limit"], jnp.clip(poly, 0.0, 1.0), 1.0)


# ------------------------------------------------------------------------------------------------ pulse
@dataclass(frozen=True)
class BiGaussPulse:
    """Two half-Gaussians about ``t_peak`` (ns): ``sigma_rise`` before, ``sigma_fall`` after; unit peak."""

    form: ClassVar[str] = "bigauss"

    def fields(self):
        """Return the field units and shapes."""
        return {"t_peak": ("ns", ()), "sigma_rise": ("ns", ()), "sigma_fall": ("ns", ())}

    def shape(self, p, t):
        """Return the pulse at times t (ns) after the photoelectron."""
        d = t - p["t_peak"]
        s = jnp.where(d < 0.0, p["sigma_rise"], p["sigma_fall"])
        return jnp.exp(-0.5 * (d / s) ** 2)

    def support(self, ranges):
        """Return the static (lo, hi) time range in ns."""
        return ranges["t_peak"][0] - SPREAD * ranges["sigma_rise"][1], ranges["t_peak"][1] + SPREAD * ranges[
            "sigma_fall"][1]  # fmt: skip


@dataclass(frozen=True)
class Exp2Pulse:
    """A rising and falling exponential from ``t_start`` (ns): e^{-s/tau_fall} - e^{-s/tau_rise}; unit peak."""

    form: ClassVar[str] = "exp2"

    def fields(self):
        """Return the field units and shapes."""
        return {"t_start": ("ns", ()), "tau_rise": ("ns", ()), "tau_fall": ("ns", ())}

    def shape(self, p, t):
        """Return the pulse at times t (ns) after the photoelectron."""
        r, f = p["tau_rise"], p["tau_fall"]
        s = jnp.maximum(t - p["t_start"], 0.0)
        speak = r * f / (f - r) * jnp.log(f / r)
        peak = jnp.exp(-speak / f) - jnp.exp(-speak / r)
        return jnp.where(t > p["t_start"], (jnp.exp(-s / f) - jnp.exp(-s / r)) / peak, 0.0)

    def support(self, ranges):
        """Return the static (lo, hi) time range in ns."""
        return ranges["t_start"][0], ranges["t_start"][1] + 2.0 * SPREAD * ranges["tau_fall"][1]


@dataclass(frozen=True)
class TablePulse:
    """A measured pulse ``shape`` sampled at ``t_first + k step`` (ns, fixed), linear between samples, 0 outside."""

    t_first: float
    step: float
    n: int
    form: ClassVar[str] = "table"

    def fields(self):
        """Return the field units and shapes."""
        return {"shape": ("", (self.n,))}

    def shape(self, p, t):
        """Return the pulse at times t (ns) after the photoelectron."""
        k = self.t_first + self.step * jnp.arange(self.n, dtype=jnp.float32)
        return jnp.interp(t, k, p["shape"], left=0.0, right=0.0)

    def support(self, ranges):
        """Return the static (lo, hi) time range in ns."""
        return self.t_first, self.t_first + self.step * (self.n - 1)


TTS_FORMS = {"gaussian": GaussianTTS, "emg": EMGTTS, "table": TableTTS}
CHARGE_FORMS = {"gauss_exp": GaussExpCharge, "table": TableCharge}
THRESHOLD_FORMS = {"table": TableThreshold, "poly": PolyThreshold}
PULSE_FORMS = {"bigauss": BiGaussPulse, "exp2": Exp2Pulse, "table": TablePulse}
