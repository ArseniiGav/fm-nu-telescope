# Literature values and the toy defaults

Every number below is taken from the cited source (sentence, table or equation) or derived from it with the
stated formula ("derived"). The toy is two-dimensional and not a fit to any site; the "toy default" column lists the
values of the toy configuration, the "comment" column says how they relate to the literature.

## 1. Optical properties of the medium

| quantity | literature value(s) | source | toy default | comment |
|---|---|---|---|---|
| absorption length, sea water | 60 m (blue, 473 nm), 26 m (UV, 375 nm); per epoch 49-69 m (blue) | [1] abstract, Tab. 2-3 | water: 60 m | matches ANTARES blue |
| absorption length, sea water | 66.5 m at 440 nm (Capo Passero) | [2] | | |
| absorption length, ocean / lake | ocean (> 1.5 km) 40-70 m; Lake Baikal 22 m (at the wavelength of maximum transparency) | [6] Tab. 1; [4] | | Baikal is much shorter than the toy water |
| attenuation length, Cascadia Basin | 27.7 m at 450 nm (absorption and scattering combined) | [3] | | not an absorption length |
| effective scattering length lambda_eff = lambda_s / (1 - <cos theta>), sea water | 265 m (blue), 122 m (UV); per epoch 228-301 m (blue) | [1] Eq. 2, abstract, Tab. 2 | water: 300 m | upper end of ANTARES blue |
| lambda_eff, ocean / lake | ocean 200-400 m; Baikal 150-400 m (seasonal), 480 m (1997) | [6] Tab. 1; [4] | | |
| geometric scattering length lambda_s, sea water | 38-62 m (blue) | [1] Tab. 2 | water: 60 m (= 300 m x 0.2) | consistent |
| mean scattering cosine g, water | ANTARES: (1 - eta) x 0.924 with eta = 0.05-0.19, i.e. 0.75-0.88; natural waters 0.85-0.96 | [1] Eq. 6, Tab. 2; [4] | 0.8 | consistent |
| absorption length, South Pole ice | ~95 m (1.5-2.0 km depth), ~150 m (2.2-2.5 km); 20-120 m depending on wavelength (AMANDA depths); ~125 m average (IceCube) | [6] Tab. 1; [7]; [13] | ice: 100 m | consistent |
| lambda_eff, South Pole ice | ~20 m (1.5-2.0 km), ~40 m (2.2-2.5 km); "on average 25 m" | [6] Tab. 1; [7] | ice: 25 m | consistent |
| g, ice | 0.9 in SPICE Mie (0.8 in the earlier AMANDA model, 0.94 from Mie theory); effective coefficients almost unchanged by g | [8] | 0.8 | effective length is what matters |
| refractive index, ice | phase 1.319, group 1.356 at 400 nm (derived from the n(lambda), n_g(lambda) fits); AMANDA used fixed n = 1.32 | [9] Eq. 3-4; [8] | ice: 1.32; `n_group` = n unless set | one n for angle and speed by default; `n_group` sets the speed |
| refractive index, sea water | group velocity 0.2178 m/ns (blue), 0.2134 m/ns (UV), i.e. n_g = 1.376, 1.405 (derived); Cherenkov angle 42 deg, i.e. phase n ~ 1.35 (derived) | [1] Eq. 12; [10] | water: 1.35 (theta_c = 42.2 deg) | 1.33 would give 41.2 deg |
| Cherenkov angle | ~41 deg (n ~ 1.33, ice or water); 42 deg (sea water); 40.7 deg (ice, 400 nm, derived) | [11]; [10]; [9] | arccos(1/n) | |

## 2. Muon light and energy loss

| quantity | literature value(s) | source | toy default | comment |
|---|---|---|---|---|
| Cherenkov photons per length, beta = 1 | ~250 /cm for 300-500 nm, n ~ 1.33; Frank-Tamm: 266 /cm (300-500 nm), 332 /cm (300-600 nm) for n = 1.33 (derived) | [11] Eq. 2 | Y_t = 1.75 /m (2-D, calibrated: median full water event ~30 detected photons) | toy yield is a scaled 2-D quantity |
| extra light from secondaries along the track | ~30 % more than the bare muon; l_sec / l_mu = 0.172 + 0.032 ln(E/GeV) (older fit: 0.32 at 100 GeV, 0.47 at 10 TeV, derived) | [11] Eq. 4 and Sec. 3 | none (continuous light independent of E) | toy simplification |
| dE/dx = a + b E, ice | a = 0.259 GeV/mwe, b = 0.363e-3 /mwe (fit 20 GeV-1e11 GeV); a = 0.262, b = 0.350e-3 (20 GeV-1e7 GeV); i.e. a = 0.24 GeV/m, b = 3.3e-4 /m of ice (rho = 0.917, derived) | [14] Tab. 3 | a = 0.25 GeV/m, b = 3.3e-4 /m | between ice and water |
| dE/dx, water | same per mwe as ice (same composition): a ~ 0.26 GeV/m, b ~ 3.6e-4 /m (derived, not verified in a water table) | [14] | a = 0.25 GeV/m, b = 3.3e-4 /m (both media) | b 10 % below water |
| muon critical energy, range | E_muc = a/b; x0 ~ (1/b) ln(1 + E0/E_muc); ice: E_muc ~ 710 GeV, x0 ~ 0.39 km (100 GeV), 2.6 km (1 TeV), 8.1 km (10 TeV) (derived) | [16] Eq. 34.40; [14] | continuous loss a plus bursts; the muon stops (no more light) below 1 GeV | a 100 GeV muon stops after ~0.4 km; stopped muons are flagged in the truth (`stopped`) |
| stochastic-loss spectrum in v = E_loss/E | bremsstrahlung ~ 1/v; pair production v^-3 to v^-2 | [16] Sec. 34.6 | p(v) ~ 1/v | bremsstrahlung-like toy |
| stochastic cut v_min | v_cut 0.05-0.1 for atmospheric muons, 1e-3-1e-4 for monoenergetic beams (MMC); e_cut = 500 MeV, v_cut = 0.05 (PROPOSAL example); E_max ~ 0.5 GeV in IceCube | [14]; [15]; [11] | v_min = 1e-3 | within the MMC range |

## 3. Cascades

| quantity | literature value(s) | source | toy default | comment |
|---|---|---|---|---|
| Cherenkov track length per GeV, electromagnetic | 532.1 cm/GeV (ice); 484 cm/GeV (water, density-rescaled); earlier 437 (n = 1.33), 521 (ice), 466 cm/GeV (n = 1.35) | [12] Eq. 8-9 | 5 m of track per GeV | consistent |
| hadronic / electromagnetic light | 74 % at 100 GeV, 89 % at 100 TeV; shower-to-shower spread 17 % (6 %) | [13] | `hadronic_light_factor` = 1 (no distinction; option) | |
| fluctuation of the light yield | sigma proportional to sqrt(E) (EM) | [12] | Poisson only | |
| longitudinal profile | Gamma distribution in t = z/X0; X0 = 39.75 cm (ice), 36.08 cm (water); maximum at 1.6 m (10 GeV) to 4.3 m (10 TeV) in ice (derived from Tab. B.4); "typical lengths of 10 m"; "O(5 m-10 m)" | [12] Eq. 11-13; [13]; [26] App. A.1 | point-like | small vs module spacing |
| angular emission profile | dn/dOmega = a exp(b abs(cos theta - cos theta_c)^c) + d, a = 4.27, b = -6.03, c = 0.299, d = -0.001 (e-); polar-angle distribution (derived): mode at theta_c, median 42 deg, 68 % in [34.7, 59.0] deg, 90 % in [23, 86] deg, 65 % within +-10 deg of theta_c, 4 % backward | [12] Eq. 15, Tab. B.5 | this profile (`cascade_profile = "radel-wiebusch"`, sampled as the polar angle; measured median 42.0 deg, 65 % within +-10 deg, 4.2 % backward); option `gaussian` with sigma = 10 deg | matches |

## 4. Inelasticity of nu_mu CC

| quantity | literature value(s) | source | toy default | comment |
|---|---|---|---|---|
| mean y, nu CC | 0.483 (10 GeV), 0.477 (100 GeV), 0.472 (1 TeV), 0.426 (10 TeV) | [17] Tab. 1 | p(y) ~ 1 + eps (1 - y)^2 on [0, 1], eps = 1: mean 0.44 (`inelasticity = "icecube"`); option `uniform` in (0.05, 0.75) | between nu and nubar |
| mean y, nubar CC | 0.333 (10 GeV), 0.340 (100 GeV), 0.354 (1 TeV), 0.345 (10 TeV) | [17] Tab. 2 | | |
| measured mean y (IceCube starting tracks) | 0.42 (E_vis 1-3 TeV), 0.42 (3-10 TeV), 0.38 (10-30 TeV) | [20] Tab. I | | |
| shape | dp/dy = N (1 + eps (1 - y)^2) y^(lambda - 1) | [20] Eq. 11 | this shape with lambda = 1, eps = 1 | |

## 5. Spectra and training-set conventions

| quantity | literature value(s) | source | toy default | comment |
|---|---|---|---|---|
| atmospheric muons at sea level | dN/dE ~ 0.14 E^-2.7 x (pion + kaon terms); asymptotically (E >> 1 TeV) one power steeper than the primaries | [21] Eq. 30.4 | | |
| atmospheric muons at depth, energy | dN/dE ~ E^-gamma with gamma ~ 3.8 (2 km w.e.) to 3.6 (5 km w.e.); at 2 km w.e. vertical: peak of dN/dlogE at 143 GeV, median 114 GeV, 90 % below 514 GeV (derived) | [22] Eq. 11-13, Tab. 4 | log-uniform 1e2-1e3 GeV | covers the bulk |
| atmospheric muons at depth, zenith | single-muon intensity proportional to cos theta exp(K1(h) sec theta), K1 = -0.581 h + 0.034 (h in km w.e.); at 2 km w.e.: median cos theta = 0.83, 6 % with cos theta < 0.5 (derived) | [22] Eq. 3-5, Tab. 3 | this law at h = 2 km w.e. on cos(zenith) in [0.1, 1] (`atmospheric_zenith = "depth"`, measured median 0.83); option `uniform` (median 0.55) | matches |
| conventional atmospheric neutrinos | approximately E^-3.7 (100 GeV-100 TeV) | [6] Sec. 6.1 | | |
| astrophysical neutrinos | generic E^-2; measured 2.37 +- 0.09 (nu_mu tracks, 15 TeV-5 PeV), 2.53 +- 0.07 (cascades, 16 TeV-2.6 PeV), 2.87 +0.20 -0.19 (HESE) | [6]; [23]; [24]; [25] | injection flat in log E, E^-2 by reweighting | |
| NuBench | nu_mu CC + NC only, 10 GeV-100 TeV (Flower S and Hexagon Ice LE: 10 GeV-1 TeV), CC fraction 35-88 %; cos(zenith) uniform in [-1, 1] (checked on 25k Triangle events) | [26] Tab. 1, Sec. 3.3 | log-uniform: tracks 1e2-1e4 GeV, starting tracks (E_nu) 1e2-1e3 GeV, cascades 10-1e3 GeV, atmospheric muons 1e2-1e3 GeV | spectral indices not given in the text |
| Prometheus injection default | 1e2-1e6 GeV, E^-1 | [27] App. A | | |
| IceCube Kaggle set | 100 GeV-100 PeV, all flavours | [28] | | |

## 6. Readout

| quantity | literature value(s) | source | toy default | comment |
|---|---|---|---|---|
| IceCube DOM noise | 560 Hz (standard), 780 Hz (high QE); uncorrelated 230-250 Hz + correlated 280-340 Hz; PMT dark rate 500 Hz at -40 C | [29] Sec. 3.8; [30] Tab. 1 | from the medium: ice 0.5 kHz per module | as an IceCube DOM |
| KM3NeT | 6-8 kHz per 3-inch PMT (40K), 190-250 kHz per 31-PMT DOM; simulation: 5 kHz per PMT + coincidences | [10] Sec. 1.5, 2.2; [31] | from the medium: water 50 kHz per module | ~4x lower than a 31-PMT DOM |
| ANTARES 10-inch OM | baseline 80-90 kHz (Sept. 2005) to ~60 kHz (Nov. 2005); 40-60 kHz | [34]; [35] | | |
| Baikal-GVD 10-inch OM | 20-50 kHz (April-June), > 100 kHz at the topmost OMs in other periods | [36] | | |
| P-ONE site (STRAW, 3-inch PMT) | 9 / 38 / 548 kHz (10th / 50th / 90th percentile, bioluminescence-dominated) | [3] Fig. 11 | | |
| NuBench noise per PMT | 7.5 kHz (water sets), 60 kHz (GVD-like), 0.3 kHz (ice); OM rate = N_PMT x R_PMT | [26] Tab. 10, Eq. B.3 | | |
| PMT transit-time spread | IceCube 3.2 ns, sigma = 2.0 ns measured; KM3NeT 4.5 ns FWHM; ANTARES 2.6 ns FWHM; Baikal-GVD 3.4 ns FWHM; STRAW 2.5-4 ns | [30]; [10]; [32]; [37]; [3] | jitter sigma = 1 ns | |
| detector time resolution | ANTARES 1 ns; IceCube DOM-to-DOM 1.2 ns | [33]; [29] Sec. 3.3 | | |
| NuBench pulse building | merge window = TTS (1.5-4.5 ns), pulse time = mean of merged photons, charge = number of photons; then time smearing sigma = 1 ns, charge smearing 0.25 p.e.; trigger window >= 5 us; test sets without noise pulses | [26] Sec. 3.2-3.3, App. B.1 | merge window 3.5 ns, pulse time of the first hit (option `pulse_time = "mean"`); window [0, 5000) ns; no charge smearing | |
| Prometheus output | Cherenkov photons at the OMs, no noise, no directions | [26] Sec. 3.2 | | |

## 7. Direction convention

| quantity | literature value(s) | source | toy default | comment |
|---|---|---|---|---|
| IceCube zenith / azimuth | "Zenith/Azimuth angles refer to the direction which is opposite to the particle's direction; i.e. they refer to the direction the particle (track) came from" | [38] | zenith = arccos(-dir_z), azimuth of the came-from vector | same as IceCube |
| NuBench zenith / azimuth | "Zenith and azimuthal angles describe the direction of travel from the perspective of the respective particle"; Prometheus: zenith "0 corresponds to up-going neutrinos" | [26] Tab. 3; [27] App. A | | NuBench zenith = pi - toy zenith |

## References

1. ANTARES Collaboration (J. A. Aguilar et al.), Transmission of light in deep sea water at the site of the ANTARES neutrino telescope, Astropart. Phys. 23, 131-155 (2005), arXiv:astro-ph/0412126, doi:10.1016/j.astropartphys.2004.11.006.
2. G. Riccobene et al. (NEMO Collaboration), Deep seawater inherent optical properties in the Southern Ionian Sea, Astropart. Phys. 27, 1-9 (2007), arXiv:astro-ph/0603701, doi:10.1016/j.astropartphys.2006.08.006.
3. N. Bailly et al., Two-year optical site characterization for the Pacific Ocean Neutrino Experiment (P-ONE) in the Cascadia Basin, Eur. Phys. J. C 81, 1071 (2021), arXiv:2108.04961, doi:10.1140/epjc/s10052-021-09872-5.
4. Baikal Collaboration (V. A. Balkanov et al.), In situ measurements of optical parameters in Lake Baikal with the help of a neutrino telescope, Appl. Opt. 38, 6818 (1999), arXiv:astro-ph/9903342, doi:10.1364/AO.38.006818.
5. V. M. Aynutdinov et al. (Baikal-GVD), Monitoring of optical properties of deep waters of Lake Baikal in 2021-2022, PoS(ICRC2023)977 (2023), arXiv:2309.16300, doi:10.22323/1.444.0977.
6. U. F. Katz, C. Spiering, High-energy neutrino astrophysics: status and perspectives, Prog. Part. Nucl. Phys. 67, 651-704 (2012), arXiv:1111.0507, doi:10.1016/j.ppnp.2011.12.001.
7. J. Lundberg et al., Light tracking through ice and water: scattering and absorption in heterogeneous media with Photonics, Nucl. Instrum. Meth. A 581, 619-631 (2007), arXiv:astro-ph/0702108, doi:10.1016/j.nima.2007.07.143.
8. IceCube Collaboration (M. G. Aartsen et al.), Measurement of South Pole ice transparency with the IceCube LED calibration system, Nucl. Instrum. Meth. A 711, 73-89 (2013), arXiv:1301.5361, doi:10.1016/j.nima.2013.01.054.
9. P. B. Price, K. Woschnagg, Role of group and phase velocity in high-energy neutrino observatories, Astropart. Phys. 15, 97-100 (2001), arXiv:hep-ex/0008001, doi:10.1016/S0927-6505(00)00142-0.
10. KM3NeT Collaboration (S. Adrian-Martinez et al.), Letter of intent for KM3NeT 2.0, J. Phys. G 43, 084001 (2016), arXiv:1601.07459, doi:10.1088/0954-3899/43/8/084001.
11. L. Raedel, C. Wiebusch, Calculation of the Cherenkov light yield from low energetic secondary particles accompanying high-energy muons in ice and water with Geant4 simulations, Astropart. Phys. 38, 53-67 (2012), arXiv:1206.5530, doi:10.1016/j.astropartphys.2012.09.008.
12. L. Raedel, C. Wiebusch, Calculation of the Cherenkov-light yield from electromagnetic cascades in ice with Geant4, Astropart. Phys. 44, 102-113 (2013), arXiv:1210.5140, doi:10.1016/j.astropartphys.2013.01.015.
13. IceCube Collaboration (M. G. Aartsen et al.), Energy reconstruction methods in the IceCube neutrino telescope, JINST 9, P03009 (2014), arXiv:1311.4767, doi:10.1088/1748-0221/9/03/P03009.
14. D. Chirkin, W. Rhode, Propagating leptons through matter with Muon Monte Carlo (MMC), arXiv:hep-ph/0407075 (2004).
15. M. Dunsch et al., Recent improvements for the lepton propagator PROPOSAL, Comput. Phys. Commun. 242, 132-144 (2019), arXiv:1809.07740, doi:10.1016/j.cpc.2019.03.021.
16. Particle Data Group (S. Navas et al.), Review of particle physics, Phys. Rev. D 110, 030001 (2024), chapter 34 "Passage of particles through matter" (revised March 2025).
17. R. Gandhi, C. Quigg, M. H. Reno, I. Sarcevic, Ultrahigh-energy neutrino interactions, Astropart. Phys. 5, 81-110 (1996), arXiv:hep-ph/9512364, doi:10.1016/0927-6505(96)00008-4.
18. R. Gandhi, C. Quigg, M. H. Reno, I. Sarcevic, Neutrino interactions at ultrahigh energies, Phys. Rev. D 58, 093009 (1998), arXiv:hep-ph/9807264, doi:10.1103/PhysRevD.58.093009.
19. A. Cooper-Sarkar, P. Mertsch, S. Sarkar, The high energy neutrino cross-section in the Standard Model and its uncertainty, JHEP 08, 042 (2011), arXiv:1106.3723, doi:10.1007/JHEP08(2011)042.
20. IceCube Collaboration (M. G. Aartsen et al.), Measurements using the inelasticity distribution of multi-TeV neutrino interactions in IceCube, Phys. Rev. D 99, 032004 (2019), arXiv:1808.07629, doi:10.1103/PhysRevD.99.032004.
21. Particle Data Group (P. A. Zyla et al.), Review of particle physics, Prog. Theor. Exp. Phys. 2020, 083C01 (2020), chapter 30 "Cosmic rays".
22. Y. Becherini, A. Margiotta, M. Sioli, M. Spurio, A parameterisation of single and multiple muons in the deep water or ice, Astropart. Phys. 25, 1-13 (2006), arXiv:hep-ph/0507228, doi:10.1016/j.astropartphys.2005.10.005.
23. IceCube Collaboration (R. Abbasi et al.), Improved characterization of the astrophysical muon-neutrino flux with 9.5 years of IceCube data, Astrophys. J. 928, 50 (2022), arXiv:2111.10299, doi:10.3847/1538-4357/ac4d29.
24. IceCube Collaboration (M. G. Aartsen et al.), Characteristics of the diffuse astrophysical electron and tau neutrino flux with six years of IceCube high energy cascade data, Phys. Rev. Lett. 125, 121104 (2020), arXiv:2001.09520, doi:10.1103/PhysRevLett.125.121104.
25. IceCube Collaboration (R. Abbasi et al.), The IceCube high-energy starting event sample: description and flux characterization with 7.5 years of data, Phys. Rev. D 104, 022002 (2021), arXiv:2011.03545, doi:10.1103/PhysRevD.104.022002.
26. R. F. Orsoe et al., NuBench: an open benchmark for deep learning-based event reconstruction in neutrino telescopes, arXiv:2511.13111 (2025, v3 2026).
27. J. Lazar et al., Prometheus: an open-source neutrino telescope simulation, Comput. Phys. Commun. 304, 109298 (2024), arXiv:2304.14526, doi:10.1016/j.cpc.2024.109298.
28. H. Bukhari et al., IceCube - Neutrinos in Deep Ice: the top 3 solutions from the public Kaggle competition, arXiv:2310.15674 (2023).
29. IceCube Collaboration (M. G. Aartsen et al.), The IceCube Neutrino Observatory: instrumentation and online systems, JINST 12, P03012 (2017), arXiv:1612.05093, doi:10.1088/1748-0221/12/03/P03012.
30. IceCube Collaboration (R. Abbasi et al.), Calibration and characterization of the IceCube photomultiplier tube, Nucl. Instrum. Meth. A 618, 139-152 (2010), arXiv:1002.2442, doi:10.1016/j.nima.2010.03.102.
31. KM3NeT Collaboration (S. Aiello et al.), The KM3NeT multi-PMT optical module, JINST 17, P07038 (2022), arXiv:2203.10048, doi:10.1088/1748-0221/17/07/P07038.
32. ANTARES Collaboration (P. Amram et al.), The ANTARES optical module, Nucl. Instrum. Meth. A 484, 369-383 (2002), arXiv:astro-ph/0112172, doi:10.1016/S0168-9002(01)02026-5.
33. ANTARES Collaboration (M. Ageron et al.), ANTARES: the first undersea neutrino telescope, Nucl. Instrum. Meth. A 656, 11-38 (2011), arXiv:1104.1607, doi:10.1016/j.nima.2011.06.103.
34. ANTARES Collaboration (J. A. Aguilar et al.), First results of the Instrumentation Line for the deep-sea ANTARES neutrino telescope, Astropart. Phys. 26, 314-324 (2006), arXiv:astro-ph/0606229, doi:10.1016/j.astropartphys.2006.07.004.
35. N. Reeb et al., Studying bioluminescence flashes with the ANTARES deep sea neutrino telescope, arXiv:2107.08063 (2021).
36. G. Safronov (for the Baikal-GVD Collaboration), Baikal-GVD: status and first results, ICHEP2020 proceedings, arXiv:2012.03373 (2020).
37. A. D. Avrorin et al. (Baikal-GVD), The optical module of the Baikal-GVD neutrino telescope, PoS(ICRC2015)1163 (2015).
38. IceCube Collaboration, icetray-public, `dataclasses/public/dataclasses/I3Direction.h` (class documentation of I3Direction), https://github.com/icecube/icetray-public.
