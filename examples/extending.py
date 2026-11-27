"""The worked examples of docs/howto/extending.md, as the guide shows them; tests/physics/test_extend.py checks each.

Each example is user code: it imports lumen and nothing private. The detector is the tutorial's tank
(``example_tank.detector()``); every derivative is checked AD against FD with ``lumen.tools.check_gradient``.
"""
# ruff: noqa: D103  (the guide explains each example)
# fmt: off

import math

import jax
import jax.numpy as jnp

import lumen
from lumen import Param
from lumen.sources import cone_about, gaussian_spot, reweight


# ---------------------------------------------------------------- 2. a source function: an LED and a laser in one
# example: flasher
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
# end example


# example: cos_m
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
# end example



# ---------------------------------------------------------------- 3. a sensor response: one function after tracking
# example: saturating_pe
@lumen.response(group="pmt", params={"sat": Param("pe", (1, 1e4), default=300.0),
                                     "qe_scale": Param("", (0, 2), default=1.0)})
def saturating_pe(h, v):                    # h: lumen.readout.Histograms; v: this response's and the group's values
    pe = v["qe_scale"] * lumen.readout.pe(h, v)                            # reuse lumen's pieces freely
    return v["sat"] * (1.0 - jnp.exp(-pe / v["sat"]))
# end example


# example: my_waveform
@lumen.response(group="pmt", params={"tau_r": Param("ns", (0.5, 20), default=2.0),
                                     "tau_f": Param("ns", (1, 100), default=8.0),
                                     "adc_per_pe": Param("ADC", default=10.0)})  # not "gain": lumen:sensor has that key
def my_waveform(h, v):
    prof = lumen.readout.pe_density(h, v, dark=False).rate  # (S, n) expected photoelectrons per bin dt over the gate
    t = h.grid.dt * jnp.arange(64)
    pulse = jnp.exp(-t / v["tau_f"]) - jnp.exp(-t / v["tau_r"])
    return v["adc_per_pe"] * jax.vmap(lambda q: jnp.convolve(q, pulse)[: q.shape[0]])(prof)
# end example


# ---------------------------------------------------------------- 4. a medium model (a value)
# example: water_hg
@lumen.medium_model("mylab:water_hg", params={"g": Param("", (-0.99, 0.99), default=0.9)})
def water_hg(material, base):               # base: the Geant4 reading of this material (lumen.media.MediumSpec)
    hg = lumen.media.henyey_greenstein("g")              # a Lobe reading its parameter "g" (declared above)
    return base.with_channel("rayleigh", lobe=hg)        # or add_channel(...), with_absorption(...)
# end example



def lined_tank(out):
    """Return the tutorial tank with a liner: the opticalsurface "Liner" (a polished metal, R = 0.5) on the outside."""
    import dataclasses
    from pathlib import Path

    import example_tank

    liner = ('  <opticalsurface name="Liner" model="unified" finish="polished" type="dielectric_metal" value="0">\n'
             '    <property name="REFLECTIVITY" ref="LINER_REFLECTIVITY"/>\n  </opticalsurface>\n</solids>')
    skin = ('  <skinsurface name="LinerSkin" surfaceproperty="Liner"> <volumeref ref="World"/> </skinsurface>\n'
            '</structure>')
    gdml = example_tank.GDML.read_text().replace("</solids>", liner).replace("</structure>", skin).replace(
        "</define>", '  <matrix name="LINER_REFLECTIVITY" coldim="2" values="1.5*eV 0.5 5.0*eV 0.5"/>\n</define>')
    src = Path(out) / "lined_tank.gdml"
    src.write_text(gdml)
    # the liner is the skin of the tank's mother: leaving the tank, Geant4 meets it, and so does lumen
    recipe = dataclasses.replace(example_tank.RECIPE, name="lined_tank")
    return lumen.convert(src, Path(out) / "lined_tank.lumen", recipe=recipe)
