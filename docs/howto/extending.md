# Extending lumen

This guide is for changing what lumen simulates: a light source, a sensor response or a bulk medium. Each
of these is one plain-JAX function, or a model value you pass at load, whose parameters you declare. A declared
parameter becomes a `Params` key, so `Params.set`, `jax.jvp`, calibrations and the gradient check take it as they take
lumen's own. You never edit the photon loop.

Every code block below comes from `examples/extending.py`, and `tests/physics/test_extend.py` checks each one's
derivatives, AD against finite differences, on the tutorial tank (`examples/data/example_tank.gdml`). A derivative
that moves where the light starts (a position, a direction) is not available in this version: a tangent on one is
refused (docs/explanation/known-limits.md).

## 1. Change a value

A value the detector file has is a key: `p = det.params().set("materials/Water/ABSLENGTH", table)`. Changing it never
recompiles, and `jax.jvp` / `jax.grad` over it give exact derivatives (the tutorial, section 4). Nothing below is
needed for that.

## 2. Write a source function

A source is one function `f(v, key, n)` returning `lumen.Photons`: `n` photons whose weights sum, in expectation, to
the photons the source stands for. `v` holds its declared parameters (traced: their derivatives flow), `key` is a JAX
key, `n` a static count. `simulate` calls it once with `n` = `samples` and runs the photons in
chunks, so photon i is the same whatever the chunk size. Its keys are `source/<name>/<field>`; `src.parameters(det.params())` adds them at
their defaults.

```python
light = lumen.simulate(det, flasher, samples=1 << 22, seed=0)
```

A parameter declared `method="moving_paths"` (a position) moves where the light starts: it has no derivative in
this version, and a tangent on it, or one reaching the photons' `x`, `d` or `lam`, is refused. A parameter that changes
the density of the directions (an axis, a width, a spectrum) enters by keeping the draws and weighting them with
`lumen.sources.reweight(logp)`, which is 1 in value and gives the score d logp in derivative. Several emitters in one
source: `lumen.concat` of their photons, each with weights scaled to its share.

<!-- example: flasher -->
```python
@lumen.source(params={"led_x": Param("m", shape=(3,), method="moving_paths", default=(0.0, 0.0, 0.2)),
                      "led_axis": Param("", shape=(3,), default=(0.0, 0.0, -1.0)), "m": Param("", default=4.0),
                      "beam_x": Param("m", shape=(3,), default=(0.1, 0.0, 0.3)),
                      "beam_sigma": Param("m", default=0.003),
                      "t_led": Param("ns", default=0.0), "t_beam": Param("ns", default=25.0)})
def flasher(v, key, n):
    k1, k2, k3 = jax.random.split(key, 3)
    h = n // 2
    a = jax.lax.stop_gradient(v["led_axis"])                               # directions fixed; the axis enters by weight
    u = jax.random.uniform(k1, (h, 2), minval=1e-7)
    d = cone_about(a, u[:, 0] ** (1 / (v["m"] + 1)), 2 * math.pi * u[:, 1])

    def logp(d_):                                                          # cos^m about the live axis (+ a constant)
        return v["m"] * jnp.log(jnp.clip(d_ @ v["led_axis"], 1e-12))

    led = lumen.Photons(x=jnp.broadcast_to(v["led_x"], (h, 3)), d=d,
                        t=v["t_led"] + 2.0 * jax.random.exponential(k2, (h,)), lam=jnp.full(h, 450.0),
                        w=reweight(jax.vmap(logp)(d)) * 1e6 / h)
    beam = gaussian_spot(k3, n - h, centre=v["beam_x"], sigma=v["beam_sigma"], direction=(0.0, 0.0, -1.0),
                         t=v["t_beam"], lam=405.0, photons=1e4)            # a fixed spot, its centre by reweight
    return lumen.concat(led, beam)
```

The LED's directions are drawn about its axis at the axis's value, and the live axis enters by `reweight`; its
position moves the light (`method="moving_paths"`: no derivative). The laser (`gaussian_spot`) is a pencil with a
Gaussian profile: its centre and width enter by `reweight` (for a beam, the right estimator).

<!-- example: cos_m -->
```python
def cos_m_led(m):
    """Return an LED of profile cos^m about -z at (0, 0, 0.2), 420 nm, its position a parameter that moves the light."""

    @lumen.source(params={"x": Param("m", shape=(3,), method="moving_paths", default=(0.0, 0.0, 0.2))},
                  name=f"cos{m:g}")
    def led(v, key, n):
        axis = jnp.array([0.0, 0.0, -1.0])
        u = jax.random.uniform(key, (n, 2), minval=1e-7)
        d = cone_about(axis, u[:, 0] ** (1.0 / (m + 1.0)), 2 * math.pi * u[:, 1])
        return lumen.Photons(x=jnp.broadcast_to(v["x"], (n, 3)), d=d, t=jnp.zeros(n), lam=jnp.full(n, 420.0),
                             w=jnp.ones(n))

    return led
```

`lumen.tools.extend.check_source(src, logpdf)` compares a source's weighted directions with a density you give (a
chi^2 in cells of equal solid angle).

## 3. Write a sensor response

The readout bins the run's deposit records per sensor over the axes the sensor model declares (features of the deposit such as the
wavelength, the cathode angle or the incidence angle, `lumen.sensors.features`), and time nodes
(`lumen.readout.grid`). A response is plain JAX from them to any observable, with its parameters declared; the
library stages (`lumen.readout.detection`: `expected_pe`, `masses`, `spread`, `density`) compose freely. `v` holds the response's values by
short name and every `sensors/<group>/...` key, so lumen's own `lumen.readout.pe(h, v)` and
`lumen.readout.pe_density(h, v, dark=False).rate` (the expected photoelectron time profile, after the delay and the time spread) take
it. `lumen.load(path, responses=(r,))` adds the keys `sensors/<group>/<name>`; a name the
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
                                     "adc_per_pe": Param("ADC", default=10.0)})  # not "gain": lumen:sensor has that key
def my_waveform(h, v):
    prof = lumen.readout.pe_density(h, v, dark=False).rate  # (S, n) expected photoelectrons per bin dt over the gate
    t = h.grid.dt * jnp.arange(64)
    pulse = jnp.exp(-t / v["tau_f"]) - jnp.exp(-t / v["tau_r"])
    return v["adc_per_pe"] * jax.vmap(lambda q: jnp.convolve(q, pulse)[: q.shape[0]])(prof)
```

```python
det = lumen.load(path, responses=(saturating_pe, my_waveform))
y = saturating_pe(light, det.params())      # differentiable in every key, no new simulation
```

What a response can see: the sensor, the time and the axes the value set declares (its acceptance tables' and its
`axes` record's: any feature of the deposit, which keeps the crossing point, direction and surface normal in the
sensor's frame and the re-emission count), on the declared grid, exact at its knots.

A response reads the histograms of a group whose sensor model is `lumen:sensor`, the one sensor model: any sensor (a
PMT, a SiPM tile, an X-ARAPUCA) is a value set of it (`lumen-sensor-values/1`: its acceptance tables, time and
electronics), read from the set directly; nothing is registered. Its readout is responses like the two above.

## 4. Parts of your own: a medium model, a phase function

A part is a value (`distribution:name`, its function and its declared parameters); nothing is registered. Pass it to
the load by its path, which binds it to its material without editing the file and puts it in the build's model table:

```python
det = lumen.load(path, parts={"materials/Water": water_hg})
det.params()["materials/Water/g"]           # 0.9: calibratable, differentiable, saved in calibrations
```

The build reads lumen's own models by name and yours from the table, so a detector given none builds exactly as
before. Each declared parameter is a
key `materials/<M>/<name>`: the file's property of that name when it has one, else the default. A medium model edits
Geant4's reading of the material (`lumen.media.MediumSpec`: `with_channel`, `add_channel`, `with_absorption`).

<!-- example: water_hg -->
```python
@lumen.medium_model("mylab:water_hg", params={"g": Param("", (-0.99, 0.99), default=0.9)})
def water_hg(material, base):               # base: the Geant4 reading of this material (lumen.media.MediumSpec)
    hg = lumen.media.henyey_greenstein("g")              # a Lobe reading its parameter "g" (declared above)
    return base.with_channel("rayleigh", lobe=hg)        # or add_channel(...), with_absorption(...)
```

A phase function for one channel of Geant4's reading is a part of its own (`lumen.media.phase_function`): its
`sample(p, wi, n, rand)` and `logpdf(p, wi, n, wo)`, its declared parameters, and either the polarised pair
(`pol_sample`, `pol_logpdf`: the direction given the photon's Jones vector, as lumen's Rayleigh) or `unpolarised=True`
(the direction ignores the polarisation). Its keys are under its channel:

```python
depol = lumen.media.phase_function("mylab:rayleigh_depol", params={"a": Param("", (0.0, 3.0), default=0.83)},
                                   n_rand=3, sample=sample, logpdf=logpdf, unpolarised=True)
det = lumen.load(path, parts={"materials/Water/rayleigh": depol})
det.params()["materials/Water/rayleigh/a"]  # 0.83
```

A run records the parts it read in `Light.provenance` (`"parts"`: path, name, notes): "unpolarised", and on a fast run
"fast walk off" where a medium's only channel has a lobe that is not lumen's Rayleigh (the diffusion jump acts only on
it; that medium runs the exact walk, the result unchanged in law).

The detector file records a part's name, never its code, and lumen never imports code a file names: a build not given
the part is refused. Names of lumen's own (`lumen:...`, `geant4`) are reserved.

**Which parts are hooks.** In 0.2: a material's medium model (`materials/<M>`), a channel's phase function
(`materials/<M>/<channel>`), a source (`@lumen.source`) and a sensor response (`@lumen.response`). Not yet (later
releases, each a path of its own reserved now): a surface interaction or a surface lobe (`surfaces/<S>`,
`surfaces/<S>/<slot>`), an emission's time profile or spectrum (`materials/<M>/<channel>/emission`), a sensor component
(TTS, charge, threshold, pulse: `sensors/<group>/<component>`) and a sensor feature (`sensors/<group>/axes/<name>`).
Until then a surface is one of lumen's surface types (Geant4's, read from the detector file's opticalsurface): every
surface and sensor row is a row of one parameter table read by one kernel (`lumen.surfaces.types`); optics that vary
with position are not open either (a medium is constant per volume).

## 5. Rules for code that runs in the loop

Model code runs per photon inside the photon loop, a source function before it. Both are checked:

- plain JAX, no host callbacks (`jax.pure_callback`, `io_callback`);
- no float arrays closed over: take every value as a declared parameter (the loop refuses a captured array,
  `ClosedOverValues`);
- a lobe draws at most 3 uniforms, and its sampler draws its density (a chi^2 at the first build);
- declared shapes are static; a model's parameters are scalars;
- a source function's output: `Photons` of the declared shapes, unit directions, finite values, non-negative weights
  (checked at first use, on 256 photons at the defaults).

## 6. Check it

`lumen.tools.check_gradient(f, theta0)` is the check every test of this guide uses: `f(theta, seed)` returns
observables with every random number from `seed`; per seed it compares the forward-mode derivative with a finite
difference on the same seed, and tests their paired difference with the comparisons' one test
(`lumen.tools.stats.mean_test`). Choose
the step `h` so it moves the observables by much more than ~1e-7 relative: the photon loop's float32 shares (a
surface's 1 - R, an outcome's p / q) put a fine structure of about that size into `f` (the derivative is not biased by
it), and a finite difference over a smaller step measures that structure.

```python
def f(theta, seed):
    p = flasher.parameters(det.params()).set("source/flasher/led_axis", jnp.stack([theta, 0.0, -1.0]))
    return lumen.simulate(det, flasher, p, samples=1 << 18, seed=seed).arrivals

report = lumen.tools.check_gradient(f, 0.0, h=0.01, stencil="central")
assert report.passes, report
```

Against Geant4, where Geant4 has the physics: export the detector (`lumen.gdml.export`), run Geant4 on the same sources and
compare the charge per sensor, as [the how-to guide for your own detector](your-own-detector.md) does in its last
section. Where Geant4 has no such physics, compare with a toy whose answer has a closed form, and check the
derivatives with `check_gradient`.

## 7. Package it

A part is a value your package exports: pass it at every load (`lumen.load(path, parts=...)`); a detector file
that names it is refused by a build not given it. Source functions and responses are Python you import; they are run inputs,
not part of the detector file.
