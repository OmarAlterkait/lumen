# Extending lumen

This guide is for changing what lumen simulates: a light source, a sensor response, a bulk medium or a surface. Each
of these is one plain-JAX function, or one small model registered by name, whose parameters you declare. A declared
parameter becomes a `Params` key, so `Params.set`, `wrt=`, calibrations and the gradient check take it as they take
lumen's own. You never edit the photon loop.

Every code block below comes from `examples/extending.py`, and `tests/physics/test_extend.py` checks each one's
derivatives, AD against finite differences, on the tutorial tank (`examples/data/example_tank.gdml`).

| change | lumen 0.1.0 (lines of user code) | 0.1.2 (lines of user code) |
|---|---|---|
| a value (QE, ABSLENGTH, REFLECTIVITY) | 1: `p.set(...)` | 1: the same |
| an LED (cos^m, a disc, a spectrum, a time profile) with its position derivative | 25 (`FromArrays`) + 9 (a patch of lumen's chain start, inside lumen) | 11 (`cos_m`); an LED and a laser together: 19 (`flasher`) |
| a laser with derivatives in its centre and width | 19 (`FromArrays` and score weights by hand) | part of `flasher`: one call of `gaussian_spot` |
| a response of your own after tracking | 6, reading lumen's internals | 5 (`saturating_pe`), 8 (`my_waveform`); their parameters are keys |
| a phase function | 23, a patch of `phase.RAYLEIGH`; its g is not a parameter | 4 (`water_hg`); g is a key |
| a reflectance model of a detecting surface | 19, a patch of `SENSOR_SURFACES`; no new keys | 8 (`schlick_cathode`); R0 is a key |
| check a derivative | 31 (AD, FD, paired z) | 1 call: `lumen.validate.check_gradient` |

The 0.1.0 counts are the code the extensibility study (lumen v0.1.0) had to write for the same changes; both columns
count non-blank lines without comments or docstrings.

## 1. Change a value

A value the detector file has is a key: `p = det.params().set("materials/Water/ABSLENGTH", table)`. Changing it never
recompiles, and `jax.jvp` / `jax.grad` over it give exact derivatives (the tutorial, section 4). Nothing below is
needed for that.

## 2. Write a source function

A source is one function `f(v, key, n)` returning `lumen.Photons`: `n` photons whose weights sum, in expectation, to
the photons the source stands for. `v` holds its declared parameters (traced: their derivatives flow), `key` is a JAX
key, `n` a static count. `simulate` runs it in chunks, a call per chunk with its own key, and scales each chunk's
weights by its share of the photons. Its keys are `source/<name>/<field>`; `src.parameters(det.params())` adds them at
their defaults.

```python
light = lumen.simulate(det, flasher, samples=1 << 22, seed=0, wrt=["source/flasher/led_x"])
```

`density=` says how the directions were drawn, and that decides which derivatives can move the light:

- `"isotropic"`: uniform on the sphere. A position (a parameter declared `moves=True`, listed in `wrt`) is
  differentiated by the moving-source method, as `lumen.sources.Point`'s.
- `"direction"`: any smooth density p(d | x). Return `dlogp`, the gradient of log p in d
  (`lumen.sources.grad_logpdf(logp, d)`); without it `simulate` refuses. The method needs it at the start of each
  photon's path: 0.1.0 assumed it was zero, which made the position derivative of a cos^8 LED five times too small.
- `"none"`: no density (a beam, a Cherenkov cone): a derivative that moves the photons is refused.

A parameter that changes the density (an axis, a width, a spectrum) enters in one of two ways: draw the directions
from its live value (`d` then carries its tangent), or keep the draws and weight them with
`lumen.sources.reweight(logp)`, which is 1 in value and gives the score d logp in derivative. A hard angular edge (a
uniform cone) has an edge term the method does not carry: smooth it (below). Several emitters in one source:
`lumen.concat` of their photons, each with weights scaled to its share.

<!-- example: flasher -->
```python
@lumen.source(params={"led_x": Param("m", shape=(3,), moves=True, default=(0.0, 0.0, 0.2)),
                      "led_axis": Param("", shape=(3,), default=(0.0, 0.0, -1.0)), "m": Param("", default=4.0),
                      "beam_x": Param("m", shape=(3,), default=(0.1, 0.0, 0.3)),
                      "beam_sigma": Param("m", default=0.003),
                      "t_led": Param("ns", default=0.0), "t_beam": Param("ns", default=25.0)},
              density="direction")
def flasher(v, key, n):
    k1, k2, k3 = jax.random.split(key, 3)
    h = n // 2
    a = jax.lax.stop_gradient(v["led_axis"])                               # directions fixed; the axis enters by weight
    u = jax.random.uniform(k1, (h, 2), minval=1e-7)
    d = cone_about(a, u[:, 0] ** (1 / (v["m"] + 1)), 2 * math.pi * u[:, 1])

    def logp(d_):                                                          # cos^m about the live axis (+ a constant)
        return v["m"] * jnp.log(jnp.clip(d_ @ v["led_axis"], 1e-12))

    led = lumen.Photons(jnp.broadcast_to(v["led_x"], (h, 3)), d, v["t_led"] + 2.0 * jax.random.exponential(k2, (h,)),
                        jnp.full(h, 450.0), reweight(jax.vmap(logp)(d)) * 1e6 / h, dlogp=grad_logpdf(logp, d))
    beam = gaussian_spot(k3, n - h, centre=v["beam_x"], sigma=v["beam_sigma"], direction=(0.0, 0.0, -1.0),
                         t=v["t_beam"], lam=405.0, photons=1e4)            # a fixed spot, its centre by reweight
    return lumen.concat(led, beam)                                         # beam rows: dlogp 0, no position tangent
```

The LED's directions are drawn about its axis at the axis's value, and the live axis enters by `reweight`; its
position moves (`moves=True`). The laser (`gaussian_spot`) is a pencil with a Gaussian profile: its centre and width
enter by `reweight` (for a beam, the right estimator), and it has no direction density, so its rows cannot move.

<!-- example: cos_m -->
```python
def cos_m_led(m):
    """Return an LED of profile cos^m about -z at (0, 0, 0.2), 420 nm, its position a moving parameter."""

    @lumen.source(params={"x": Param("m", shape=(3,), moves=True, default=(0.0, 0.0, 0.2))},
                  density="direction", name=f"cos{m:g}")
    def led(v, key, n):
        axis = jnp.array([0.0, 0.0, -1.0])
        u = jax.random.uniform(key, (n, 2), minval=1e-7)
        d = cone_about(axis, u[:, 0] ** (1.0 / (m + 1.0)), 2 * math.pi * u[:, 1])
        logp = lambda d_: m * jnp.log(jnp.clip(d_ @ axis, 1e-12))          # noqa: E731
        return lumen.Photons(jnp.broadcast_to(v["x"], (n, 3)), d, jnp.zeros(n), jnp.full(n, 420.0),
                             jnp.ones(n), dlogp=grad_logpdf(logp, d))

    return led
```

<!-- example: soft_cone -->
```python
CONE, EDGE = math.radians(20.0), math.radians(1.0)


@lumen.source(params={"x": Param("m", shape=(3,), moves=True, default=(0.0, 0.0, 0.2))}, density="direction")
def soft_cone(v, key, n):
    """Draw a 20 degree cone about -z whose edge is a logistic about 1 degree wide (a hard edge is refused)."""
    axis = jnp.array([0.0, 0.0, -1.0])
    top = CONE + 8 * EDGE                                                  # drawn uniformly out to 8 widths
    u = jax.random.uniform(key, (n, 2), minval=1e-7)
    d = cone_about(axis, 1.0 - u[:, 0] * (1.0 - math.cos(top)), 2 * math.pi * u[:, 1])

    def logp(d_):                                                          # the edge in cos(theta): smooth everywhere
        return jax.nn.log_sigmoid((d_ @ axis - math.cos(CONE)) / (EDGE * math.sin(CONE)))

    return lumen.Photons(jnp.broadcast_to(v["x"], (n, 3)), d, jnp.zeros(n), jnp.full(n, 420.0),
                         jnp.exp(jax.vmap(logp)(d)), dlogp=grad_logpdf(logp, d))
```

Measured (the extensibility study, 8 seeds x 2^20 photons, PMT 7): with `dlogp` the cos^m position derivatives agree
with the FD (m = 2, 4: z 1.5, 0.15; m = 8: z 0.40, where 0.1.0's start gave z -38); the 1 degree soft edge agrees
(z 0.00), while a hard 20 degree edge misses 98 % of the derivative (refused here). A soft edge of width eps costs
variance ~ 1/eps.

`lumen.extend.check_source(src, logpdf)` compares a source's weighted directions with a density you give (a chi^2 in
cells of equal solid angle) and its `dlogp` with `grad_logpdf(logpdf)`.

## 3. Write a sensor response

Tracking fills histograms per sensor: wavelength cell x hit-angle cell, and time nodes (`lumen.readout.grid`). A
response is plain JAX from them to any observable, with its parameters declared. `v` holds the response's values by
short name and every `sensors/<group>/...` key, so lumen's own `lumen.readout.pe(h, v)` and
`lumen.readout.time_nodes(h, v)` take it. `det.with_responses(r)` adds the keys `sensors/<group>/<name>`; a name the
sensor model has already (`gain`, `qe`, ...) is refused.

<!-- example: saturating_pe -->
```python
@lumen.response(group="pmt", params={"sat": Param("pe", (1, 1e4), default=300.0),
                                     "qe_scale": Param("", (0, 2), default=1.0)})
def saturating_pe(h, v):                    # h: lumen.readout.Histograms; v: this response's and the group's values
    pe = v["qe_scale"] * lumen.readout.pe(h, v)                            # reuse lumen's pieces freely
    return v["sat"] * (1.0 - jnp.exp(-pe / v["sat"]))
```

<!-- example: my_waveform -->
```python
@lumen.response(group="pmt", params={"tau_r": Param("ns", (0.5, 20), default=2.0),
                                     "tau_f": Param("ns", (1, 100), default=8.0),
                                     "adc_per_pe": Param("ADC", default=10.0)})  # not "gain": lumen:pmt has that key
def my_waveform(h, v):
    nodes, _, _ = lumen.readout.time_nodes(h, v)          # (S, nt) expected photoelectrons on t0 + j dt
    t = h.grid.dt * jnp.arange(64)
    pulse = jnp.exp(-t / v["tau_f"]) - jnp.exp(-t / v["tau_r"])
    return v["adc_per_pe"] * jax.vmap(lambda q: jnp.convolve(q, pulse)[: q.shape[0]])(nodes)
```

```python
det = det.with_responses(saturating_pe, my_waveform)
y = saturating_pe(light, det.params())      # differentiable in every key, no new simulation
```

What a response can see: the sensor, wavelength, hit angle (on a flat disc, the radius on the cathode) and time, on
the declared grid, exact at its knots. Not the azimuth, the point on the cathode or the incidence direction: the
deposit does not keep them.

## 4. Register a medium or surface model

A model is registered by name (`distribution:name`), with declared parameters; a detector binds it by that name in its
file, or at load without editing the file:

```python
det = lumen.load(path, models={"materials/Water": "mylab:water_hg", "surfaces/Liner": "mylab:schlick"})
det.params()["materials/Water/g"]           # 0.9: calibratable, differentiable, saved in calibrations
```

lumen's own models are found first, so a detector naming none builds exactly as before. Each declared parameter is a
key `materials/<M>/<name>` or `surfaces/<S>/<name>`: the file's property of that name when it has one, else the
default. A medium model edits Geant4's reading of the material (`lumen.media.MediumSpec`: `with_channel`,
`add_channel`, `with_absorption`); a surface model returns a `lumen.surfaces.SurfaceType`, the probabilities of the
ten outcomes from the incidence (`ctx`: wavelength, directions, normal, indices, Jones vector).

<!-- example: water_hg -->
```python
@lumen.medium_model("mylab:water_hg", params={"g": Param("", (-0.99, 0.99), default=0.9)})
def water_hg(material, base):               # base: the Geant4 reading of this material (lumen.media.MediumSpec)
    hg = lumen.media.henyey_greenstein("g")              # a Lobe reading its parameter "g" (declared above)
    return base.with_channel("rayleigh", lobe=hg)        # or add_channel(...), with_absorption(...)
```

<!-- example: schlick -->
```python
@lumen.surface_model("mylab:schlick", params={"R0": Param("", (0, 1), default=0.04)}, detects=False)
def schlick(surface, base):
    def probs(p, ctx):                                    # ctx: lam, wi, n, n1, n2, Jones vector (as today)
        c = lumen.surfaces.cos_incidence(ctx)
        r = p["R0"] + (1 - p["R0"]) * (1 - c) ** 5
        return lumen.surfaces.outcome_vector(specular=r, absorb=1 - r)
    return lumen.surfaces.SurfaceType("schlick", ("R0",), (lumen.surfaces.SPECULAR, lumen.surfaces.ABSORB), probs)
```

A detecting surface is the same with `detects=True`: its type gives DETECT and reads `eff`, the efficiency on entry
lumen sets (1 under `lumen:pmt`, whose QE applies after tracking).

<!-- example: schlick_cathode -->
```python
@lumen.surface_model("mylab:schlick_cathode", params={"R0": Param("", (0, 1), default=0.04)}, detects=True)
def schlick_cathode(surface, base):
    def probs(p, ctx):                                    # Schlick's reflectance, the rest detected with eff
        c = lumen.surfaces.cos_incidence(ctx)
        r = p["R0"] + (1 - p["R0"]) * (1 - c) ** 5
        return lumen.surfaces.outcome_vector(specular=r, detect=(1 - r) * p["eff"], absorb=(1 - r) * (1 - p["eff"]))
    return lumen.surfaces.SurfaceType("schlick_cathode", ("R0", "eff"),
                                      (lumen.surfaces.SPECULAR, lumen.surfaces.DETECT, lumen.surfaces.ABSORB), probs)
```

What a model cannot be in 0.1.2: a new direction lobe on a surface (the facet lobe carries one width), and optics that
vary with position (a medium is constant per volume).

## 5. Rules for code that runs in the loop

Model code runs per photon inside the photon loop, a source function before it. Both are checked:

- plain JAX, no host callbacks (`jax.pure_callback`, `io_callback`);
- no float arrays closed over: take every value as a declared parameter (the loop refuses a captured array,
  `ClosedOverValues`);
- a lobe draws at most 3 uniforms, and its sampler draws its density (a chi^2 at the first build);
- outcome probabilities lie in [0, 1] and sum to 1, only on the outcomes the type declares (checked at the first
  build over incidences and the parameters' range corners);
- declared shapes are static; a model's parameters are scalars;
- a source function's output: `Photons` of the declared shapes, unit directions, finite values, non-negative weights,
  and `dlogp` exactly when `density="direction"` (checked at first use, on 256 photons at the defaults).

## 6. Check it

`lumen.validate.check_gradient(f, theta0)` is the check every test of this guide uses: `f(theta, seed)` returns
observables with every random number from `seed`; per seed it compares the forward-mode derivative with a finite
difference on the same seed, and judges the paired z by the gradient check's rules (`lumen.validate.stats`).

```python
def f(theta, seed):
    p = led.parameters(det.params()).set("source/cos4/x", jnp.stack([theta, 0.0, 0.2]))
    return lumen.simulate(det, led, p, samples=1 << 18, seed=seed, wrt=["source/cos4/x"]).arrivals

report = lumen.validate.check_gradient(f, 0.0, h=0.01, stencil="central")
assert report.passes, report
```

Against Geant4, where Geant4 has the physics: export the detector (`lumen.gdml.export`), run g4ref's point or pencil
sources (`refs/g4ref/run.sh`) and compare the charge per sensor (`lumen.validate.charge`), as the how-to guide for your
own detector does. Where Geant4 has no such physics, compare with a toy whose answer has a closed form, and check the
derivatives with `check_gradient`.

## 7. Package it

A model registered at import (`@lumen.medium_model`, `@lumen.surface_model`) is found by name once its module is
imported. To make it available without an import, publish it as an entry point of the group `lumen.optics_models`
in your package's `pyproject.toml`:

```toml
[project.entry-points."lumen.optics_models"]
"mylab:water_hg" = "mylab.optics:water_hg"
```

A detector file that names `mylab:water_hg` then loads wherever the package is installed, and fails with the
registered names listed where it is not. Source functions and responses are Python you import; they are run inputs,
not part of the detector file.
