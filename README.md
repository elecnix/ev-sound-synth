# ev-sound-synth

Electric vehicle sounds, synthesised from a model of the drive unit rather than from
recordings. Eight 30-second stereo renders, pure numpy, no samples. The pedestrian alert is
measured against the US regulation that governs it and passes.

```bash
python3 -m evsound.render --all     # writes out/*.wav + out/manifest.json
python3 -m evsound.verify           # re-measures the rendered files
python3 -m pytest -q                # 69 tests
```

Needs Python 3.11+ and numpy. Nothing else — no scipy, no audio library, no build step.

Pre-rendered audio is attached to the [latest release](../../releases/latest), not committed
to the repository.

## Why an EV needs two different sound systems

**Outside, the pedestrian alert (AVAS).** Legally required, because an EV at walking pace is
inaudible. In the US that is [FMVSS No. 141](https://www.law.cornell.edu/cfr/text/49/571.141);
in Europe, UNECE R138. The rule is specific: measure thirteen one-third octave bands from
315 Hz to 5000 Hz and clear a minimum level in either two non-adjacent bands or four, where
four must reach across at least nine of the thirteen. The car must also get at least 3 dB
louder at each of 10, 20 and 30 km/h, and sound different in reverse. Above roughly 30 km/h the
rule stops asking, because tyre noise has taken over.

**Inside, the performance sound.** Required by nothing, entirely a design choice. Every
manufacturer now composes an interior voice for its EVs, typically a high frequency over a low
rumble, arranged to mark the moment a maximum-power mode engages. This project calls that
*boost mode*.

Under both sits what the hardware makes on its own: motor orders, gear mesh, inverter switching
and tyre roar. Those are modelled here too, because the synthetic layers have to sit on top of
them convincingly.

## The eight generators

| # | Name | What it is |
|---|------|-----------|
| 1 | `avas_forward` | Pedestrian alert, forward. Four bands at 400/800/1600/3150 Hz over the FMVSS speed staircase |
| 2 | `avas_reverse` | Reverse alert. Different bands (500/1000/2000/4000 Hz), pulsing at 1.6 Hz |
| 3 | `motor_orders` | Drive-unit orders: the 48th slot order, torque ripple, both gear mesh stages |
| 4 | `inverter_pwm` | 10 kHz PWM carrier with sidebands that open and close with speed |
| 5 | `boost_performance` | Low rumble under a bright harmonic lead, plus the engagement stinger |
| 6 | `road_wind` | Tyre and wind noise, five crossfaded bands with different speed exponents |
| 7 | `cabin_mix` | Driver's ear: 3 + 4 + 5 + 6 balanced through a full launch cycle |
| 8 | `exterior_passby` | Kerbside: alert over tyre noise, motor barely there |

Files 1, 2 and 8 run the FMVSS speed staircase — stationary, then 10, 20 and 30 km/h, each held
long enough to measure. Files 3 to 7 run a performance cycle: a launch to 60 mph in 3.4 s, a
pull to 168 km/h, a lift with regeneration, a second pull, then a hard stop.

## How each one works

Everything hangs off road speed. There is no engine speed to borrow, so the chain is road speed
→ wheel rotation → motor shaft through a single-speed 11.1:1 reduction → electrical frequency
at 4 pole pairs → motor order. That chain lives in `vehicle.py`, and it is why the pitch tracks
the car instead of drifting free of it.

**Pedestrian alert.** Put energy only where the rule measures it. Four one-third octave bands,
each a tone plus a slightly detuned partner for movement plus a whisper of narrowband noise so
it does not sound like a test tone. Each band's level follows the rule's own table plus 4 dB,
which means the required 3 dB step per speed increment falls out of the design instead of being
tuned in afterwards. Pitch glides 5% with speed; the whole thing fades out by 34 km/h.

**Motor orders.** Additive synthesis on *orders* — components locked to a fixed multiple of
shaft speed. The 48th order, one per stator slot, is the dominant radial force and is what
people call EV whine. The 24th is the 6th electrical harmonic from torque ripple. The 23rd and
6.16th are the two gear mesh stages. Level follows load, not speed, so a coasting car goes
quiet without going silent.

**Inverter.** A traction inverter chops DC at a fixed rate, and the chopping beats against the
motor's electrical fundamental. So the spectrum is a fixed 10 kHz tone with a comb of sidebands
at 10 kHz ± k × the electrical frequency: the carrier stays put while the comb opens with
speed. That is what makes an EV sound electronic rather than mechanical.

**Boost mode.** Two layers that can be measured separately. Both follow a virtual rev that
rises mostly with road speed but jumps a little with pedal — that jump is what makes the sound
feel like a response to the driver rather than a speedometer read aloud. A dive-then-climb
stinger marks engagement at t = 2 s.

**Road and wind.** Tyre noise grows about 30·log10(v), wind about 60·log10(v), so the balance
tips from mid-band roar to high hiss as speed rises. Five fixed noise bands crossfaded by the
speed curve, rather than a swept filter.

## Two implementation choices worth knowing

**Phase accumulators, not per-block frequencies.** Every oscillator integrates an instantaneous
frequency array into a continuous phase, so a pitch that glides with road speed never clicks.
`test_swept_sine_is_phase_continuous` measures that rather than assuming it.

**Frequency-domain filtering.** A recursive filter would need a Python loop over 1.4 million
samples. Masking an rFFT is exact for a fixed response and runs in milliseconds. Where a filter
genuinely has to move — road noise brightening with speed — the generator crossfades fixed
bands instead of sweeping one.

## Levels are real, and the files say so

Digital samples carry no absolute level, so every generator declares the dB SPL a full-scale
sample would make at its own microphone: 80 dB at the FMVSS position 2 m out, 100 dB at the
driver's ear. That declaration is what makes a compliance measurement mean anything.

The written files are boosted to a comfortable listening level, because the alert at rest is
only about 48 dB(A) and would be near-silent on a laptop. `out/manifest.json` records the boost
and the resulting calibration for every file, so any measurement can be wound back to real
sound pressure. `verify.py` does exactly that — it reads the WAVs back, undoes the boost, and
re-measures:

```
avas_forward - measured from the rendered file
  stationary    48.7 dB(A)   4 non-adjacent bands, span 10   PASS
  10 km/h       54.4 dB(A)   4 non-adjacent bands, span 10   PASS
  20 km/h       61.1 dB(A)   4 non-adjacent bands, span 10   PASS
  30 km/h       65.6 dB(A)   4 non-adjacent bands, span 10   PASS
  speed steps   +5.7, +6.7, +4.5 dB   PASS
```

Reverse is checked against Table 2 but not against the step rule: FMVSS S5.4 covers the four
forward conditions only, so a reversing car is never asked to get louder as it backs up faster.

## Testing

69 tests, and they measure output rather than check flags — filter rejection in dB, pink-noise
slope per octave, the whine's frequency against the 48th order, the inverter's sidebands at
fsw ± k·f_elec, the alert's band levels against the regulation's own tables.

Two of them encode a limit rather than a success. A moving average cannot cancel a ripple
inside its own first half-window, so `test_smooth_preserves_the_mean_and_removes_the_ripple`
checks the interior and separately bounds the edges. And the alert's forward/reverse distinction
is measured as a modulation index and a pulse rate, not as a peak-to-mean ratio, because
peak-to-mean cannot tell a pulse apart from a speed change.

## Retargeting to another vehicle

Everything vehicle-specific is in one frozen dataclass, `VehicleSpec` in `vehicle.py`. The
reference is a dual-motor performance SUV: 615 hp, 650 lb-ft, 0–60 mph in 3.4 s, 275/40R21
tyres. Manufacturers do not publish driveline internals, so these are stated assumptions — each
sets pitch, none affects whether the code is right.

| Assumed | Value | Sets |
|---|---|---|
| Final drive | 11.1:1 | Motor rpm for a given road speed |
| Motor poles | 8 (4 pole pairs) | The electrical fundamental |
| Stator slots | 48 | The whine order |
| Gear teeth | 23/71, 19 | The two mesh orders |
| Inverter switching | 10 kHz | The carrier |

## Layout

```
evsound/
  dsp.py          oscillators, FFT filters, noise, weighting, WAV i/o
  analysis.py     third-octave and wideband SPL, dominant frequency, envelope
  vehicle.py      VehicleSpec and the drive cycles
  fmvss141.py     the regulation's tables and its band-selection rules
  generators/     avas, motor, inverter, boost, road, mixes
  render.py       CLI: writes out/*.wav and the manifest
  verify.py       CLI: re-measures the written files
tests/            69 tests
```

## Sources

- [49 CFR § 571.141 — Minimum Sound Requirements for Hybrid and Electric Vehicles](https://www.law.cornell.edu/cfr/text/49/571.141)
- [FMVSS No. 141 final rule, Federal Register](https://www.federalregister.gov/documents/2018/02/26/2018-03721/federal-motor-vehicle-safety-standard-no-141-minimum-sound-requirements-for-hybrid-and-electric)
- [System-level harmonic NVH engineering in electric drivetrains (MDPI review)](https://www.mdpi.com/2032-6653/17/5/240)
- [Artificial engine sound synthesis for electric vehicles (Shock and Vibration)](https://www.hindawi.com/journals/sv/2018/5209207/)
- [Electric vehicles get alert signals to be heard by pedestrians (Acoustics Today)](https://acousticstoday.org/wp-content/uploads/2020/12/Electric-Vehicles-Get-Alert-Signals-to-be-Heard-by-Pedestrians-Benefits-and-Drawbacks-Andre%CC%81-Fiebig.pdf)

## Licence

MIT. Not affiliated with, endorsed by, or derived from any vehicle manufacturer. Every driveline
figure is either a stated assumption or a generic performance-EV specification.
