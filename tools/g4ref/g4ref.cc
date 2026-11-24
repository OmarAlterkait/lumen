// g4ref: minimal Geant4 optical-photon transport reference from a GDML file.
//
// Defaults: the correct physics (exact Cherenkov count, binomial split, nrefl before the hit); --geant4-stock
//           for Geant4 10.7's own (bitwise the stock binary dd7c09cd).
// Physics : G4OpticalPhysics only (absorption, Rayleigh, Mie, boundary, WLS; Cerenkov and
//           scintillation switched off -- photons are primaries).
// Sources : point sources from a text config (position, N, energy or spectrum file),
//           isotropic direction, random linear polarisation perpendicular to the direction.
// Sensors : a photon whose step STARTS inside a logical volume matching --sens-regex (or with a
//           GDML auxiliary tag --sens-aux TYPE[=VALUE]) is recorded at that entry point and
//           killed together with any secondaries of that step (LArSoft OpDetSensitive style).
//           --wcsim-hits: instead a photon KILLED on a boundary step into such a volume, or DETECTED on one
//           leaving it (WCSimWCSD::ProcessHits and ProcessHits_boundary).
// Seeds   : every event (= one batch of photons from one source) is reseeded from
//           (seed, global event index), so any split of the run into chunks reproduces the
//           unsplit run exactly.
// Output  : <out>_hits.npy (structured), <out>_sensors.csv, <out>_totals.csv, <out>_summary.txt
//
// Sensor paths are /pvname:copy@index-in-mother/... (unique even for unnamed GDML physvols).
// Built and tested against Geant4 10.7.4 (the version in develop.sif), sequential mode.

#include "G4RunManager.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserEventAction.hh"
#include "G4VModularPhysicsList.hh"
#include "G4OpticalPhysics.hh"
#include "G4EmStandardPhysics.hh"
#include "G4DecayPhysics.hh"
#include "G4Scintillation.hh"
#include "G4Cerenkov.hh"
#include "G4LossTableManager.hh"
#include "G4Poisson.hh"
#include "G4ScintillationTrackInformation.hh"
#include "Randomize.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4UserLimits.hh"
#include "G4ParticleTable.hh"
#include "G4UserTrackingAction.hh"
#include "G4UserEventAction.hh"
#include "G4BosonConstructor.hh"
#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4IonConstructor.hh"
#include "G4OpticalPhoton.hh"
#include "G4OpBoundaryProcess.hh"
#include "G4GDMLParser.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4MaterialTable.hh"
#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4PrimaryVertex.hh"
#include "G4PrimaryParticle.hh"
#include "G4Event.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4ProcessManager.hh"
#include "G4TouchableHistory.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4UImanager.hh"
#include "G4TransportationManager.hh"
#include "G4Navigator.hh"
#include "G4Version.hh"
#include "Randomize.hh"
#include "CLHEP/Random/MixMaxRng.h"
#if G4VERSION_NUMBER >= 1070
#include "G4OpticalParameters.hh"
#include "G4EmCalculator.hh"
#include "G4Electron.hh"
#include "G4ProductionCutsTable.hh"
#include "G4ProductionCuts.hh"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------- options
struct Options {
  std::string gdml, config, out = "g4ref";
  std::string sensRegex;          // matched against logical AND physical volume names
  std::string sensAux;            // TYPE or TYPE=VALUE of a GDML <auxiliary>
  std::string dump;               // print placements (global centre, bbox) matching this regex, exit
  std::string probe;              // points file (x y z mm per line): Geant4's own locate -> <out>_probe.csv, exit
  long seed = 12345;
  int batch = 1000;               // photons per event
  int chunk = 0, nchunks = 1;     // this job runs global events g with g % nchunks == chunk
  double splitEV = 6.19;          // per-sensor totals split at this energy
  bool writeHits = true;
  bool rayleigh = true, absorption = true, wls = true, mie = true, boundary = true;
  bool fixVelocity = true;        // reset photon speed to GROUPVEL of the medium entered (see README)
  bool revive = true;             // a sub-micron step out of a no-RINDEX volume is not a kill (see SteppingAction)
  // physics defaults: the correct physics (exact Cherenkov count, binomial fast/slow split, reflections counted before
  // the sensor is entered); --geant4-stock restores Geant4 10.7's own (bitwise the dd7c09cd binary's outputs)
  std::string nreflCount = "before_hit";   // --nrefl-count stock|before_hit: a hit's nrefl (before_hit: lumen's
                                           // definition, without the reflection ending its first step in the sensor)
  std::string wlsProfile = "delta";
  long printEvery = 1000;
  // particle-gun mode (--gun NAME): charged tracks make the light (EM physics + Cerenkov / Scintillation)
  std::string gun;                // e.g. e-, mu-; empty: optical-photon sources from --config
  double gunMeV = 0;
  G4ThreeVector gunPos, gunDir = G4ThreeVector(0, 0, 1);
  long events = 0;
  bool cherenkov = false, scint = false;
  int maxCerenkovPerStep = 100;   // Geant4 10.7 default
  std::string cerenkovCount = "exact";   // --cerenkov-count stock|exact|monotonic_fix (G4RefCerenkov)
  std::string scintSplit = "binomial";   // --scint-split stock|binomial (G4RefScintillation)
  std::vector<std::pair<std::string, double>> birks;   // --birks MAT=kB [cm/MeV]: G4EmSaturation's constant
  std::vector<std::pair<std::string, double>> scintYield;   // --scint-yield MAT=Y [/MeV]: SCINTILLATIONYIELD
  std::vector<std::string> iscalc;                          // --iscalc MAT: LArSoft's ISCalcCorrelated light
  double iscalcPrescale = 0.0392, iscalcQProton = 0.71, iscalcTemperature = 88.4;   // SBND's (ICARUS: 0.073 1 87.5)
  std::vector<std::pair<std::string, double>> efield;       // --efield LV=kV/cm: the drift field in a volume
  std::vector<std::pair<std::string, double>> stepLimit;    // --step-limit LV=mm: G4UserLimits max step
  std::string emTables;           // --em-tables MAT: write Geant4's energy-loss tables for MAT and exit
  bool wcsimHits = false;
  bool wcsimBoundary = false;    // --wcsim-boundary: WCSimOpBoundaryProcess replaces OpBoundary (g4ref_wcb build)        // --wcsim-hits: WCSim's hit (WCSimWCSD), not entry into a sensor
  bool killOptical = false;       // --kill-optical: count each step's optical photons, never transport them
  bool dumpPhotons = false;       // --dump-photons: gun mode, every Cerenkov / scintillation photon at birth
};

static void usage() {
  std::cout <<
    "usage: g4ref --gdml FILE --config FILE [options]\n"
    "  --gdml FILE          geometry (optical properties/surfaces read from it)\n"
    "  --config FILE        sources, one per line:  x y z N ENERGY_EV   (mm)\n"
    "                       or                      x y z N spectrum:FILE\n"
    "                       (FILE: two columns 'E_eV weight'; lines starting with nm: are\n"
    "                       read as 'wavelength_nm weight'). '#' comments.\n"
    "  --sens-regex RE      sensitive if logical or physical volume name matches RE (ECMAScript, search)\n"
    "  --sens-aux T[=V]     sensitive if the logical volume has GDML auxiliary type T (value V)\n"
    "  --out PREFIX         output prefix (default g4ref)\n"
    "  --seed S             base seed (default 12345)\n"
    "  --batch K            photons per event (default 1000)\n"
    "  --chunk k --nchunks n  run only global events g with g % n == k\n"
    "  --split-ev E         VUV/visible split for totals (default 6.19)\n"
    "  --no-hits            do not write the per-hit file\n"
    "  --no-rayleigh --no-absorption --no-wls --no-mie   switch processes off\n"
    "  --wcsim-hits         a hit is a photon KILLED on a boundary step into a sensitive volume (any OpBoundary\n"
    "                       status: Detection, Absorption, ...), as WCSim's WCSimWCSD counts it; entering is not\n"
    "  --wcsim-boundary     WCSim's WCSimOpBoundaryProcess instead of Geant4's (binary g4ref_wcb only; no boundary\n"
    "                       status bookkeeping, no velocity fix)\n"
    "  --no-fix-velocity    keep Geant4 10.7's photon speed after reflection off a no-RINDEX volume (c)\n"
    "  --no-revive          keep Geant4's NoRINDEX kill of a sub-micron step out of a no-RINDEX volume\n"
    "  --wls-profile delta|exponential (default delta)\n"
    "  --print-every N      progress every N events\n"
    "  --gun NAME --gun-mev E --gun-pos x,y,z --gun-dir dx,dy,dz --events N\n"
    "                       particle-gun mode (no --config): EM physics; light from --cherenkov and/or\n"
    "                       --scintillation (Geant4's processes, GDML properties); every charged step is\n"
    "                       written to PREFIX_steps.npy (with the photons Geant4 made in it)\n"
    "  --max-cerenkov-per-step N  (Geant4 10.7 default 100)\n"
    "  --geant4-stock       Geant4 10.7's own physics and bookkeeping: --cerenkov-count stock --scint-split stock\n"
    "                       --nrefl-count stock (outputs bitwise those of the earlier stock binary dd7c09cd);\n"
    "                       later options override single settings. Default: exact, binomial, before_hit\n"
    "  --cerenkov-count stock|exact|monotonic_fix  the mean Cherenkov photon number per step (default exact;\n"
    "                       stock: G4Cerenkov 10.7). exact: the Frank-Tamm integral of (1 - 1/(beta n)^2)_+ over the RINDEX\n"
    "                       table with n linear in energy between nodes, exact on every segment (lumen's);\n"
    "                       monotonic_fix: 10.7's trapezoid on 1/n^2, with the threshold found on every segment\n"
    "                       (10.7 assumes RINDEX increasing). Used for the photon number and the step limit alike\n"
    "  --scint-split stock|binomial  the fast/slow split of a step's scintillation photons (default binomial:\n"
    "                       fast ~ Binomial(N, YIELDRATIO); stock: G4Scintillation 10.7, fast = int(YIELDRATIO N))\n"
    "  --birks MAT=KB       Birks constant of material MAT in cm/MeV (G4EmSaturation: G4OpticalPhysics\n"
    "                       applies it to scintillation; the GDML never sets it, so it is 0 by default)\n"
    "  --scint-yield MAT=Y  SCINTILLATIONYIELD of material MAT [/MeV] in place of the GDML's (LArSoft's\n"
    "                       MaterialPropertyLoader: ScintYield x ScintPreScale)\n"
    "  --iscalc MAT         scintillation of MAT per step from LArSoft's ISCalcCorrelated with SBND's\n"
    "                       configuration (EllipsModBox + LArQL recombination at the step's --efield, the\n"
    "                       fast ratio by particle, ScintPreScale 0.0392): replaces the yield, Birks and\n"
    "                       YIELDRATIO (not with --birks / --scint-yield on MAT)\n"
    "  --nrefl-count stock|before_hit  a hit's reflection count (default before_hit: only reflections before the\n"
    "                       step that enters the sensor; stock: also the reflection ending that first step inside\n"
    "                       the sensor, e.g. off a steel plate behind it). --nrefl-before-hit = --nrefl-count before_hit\n"
    "  --iscalc-prescale P --iscalc-qproton Q --iscalc-temperature K   the --iscalc configuration in place of\n"
    "                       SBND's (0.0392, 0.71, 88.4 K); ICARUS: 0.073 1 87.5 (icarusalg 692ba35c)\n"
    "  --efield LV=KV       drift field [kV/cm] in logical volume LV (along x; elsewhere 0), recorded per step\n"
    "  --step-limit LV=MM   maximum step [mm] of charged particles in logical volume LV (G4StepLimiter; LArSoft's\n"
    "                       LArG4Detector stepLimits)\n"
    "  --em-tables MAT      with --gun (EM physics on): write PREFIX_emtables.csv -- G4EmCalculator's\n"
    "                       restricted, electronic (unrestricted) and total dE/dx, the range from the\n"
    "                       restricted dE/dx and the delta-ray cross-section above the production cut, for\n"
    "                       mu-, mu+, e-, e+, proton in material MAT, kinetic energy 1 keV - 100 GeV -- and exit\n"
    "  --dump-photons       gun mode: write every Cerenkov / scintillation photon at birth (not WLS photons) to\n"
    "                       PREFIX_photons.npy: evt, track, x, y, z [mm], t [ns], dx, dy, dz, polx, poly, polz, E [eV],\n"
    "                       origin (2 Cerenkov, 3 scintillation), mat (index into PREFIX_materials.csv). Draws no\n"
    "                       random numbers: hits and steps are unchanged\n"
    "  --kill-optical       gun mode: optical photons are counted in their step's record (nc, ns) and\n"
    "                       killed at birth (no optical transport, no hits): the charged steps alone\n"
    "  --probe FILE         locate each point (x y z mm per line) with Geant4's navigator: <out>_probe.csv\n"
    "                       (material, PV, copy number, depth), then exit\n"
    "  --dump RE            list placements whose LV/PV name matches RE (global centre and\n"
    "                       global axis-aligned bbox, mm) and exit (no --config needed)\n"
    "environment (point sources, --config):\n"
    "  G4REF_PENCIL=dx,dy,dz          every photon along that direction (diagnostic)\n"
    "  G4REF_PROFILE=KIND:p1,p2,...   an emission profile about the unit axis a = (ax,ay,az), in place of isotropic:\n"
    "    laser:SIGMA,ax,ay,az         direction a; position = source + 2-D Gaussian offset in the plane normal to a,\n"
    "                                 SIGMA [mm] per axis (r = SIGMA sqrt(-2 ln(1-u1)), phi = 2 pi u2)\n"
    "    led:M,ax,ay,az               intensity per solid angle ~ cos^M(theta) on theta in [0, 90 deg], 0 behind:\n"
    "                                 cos(theta) = u1^(1/(M+1)), phi = 2 pi u2\n"
    "    cone:THETA,W,ax,ay,az        intensity per solid angle 1 for theta <= THETA, falling linearly IN cos(theta)\n"
    "                                 to 0 at THETA + W, 0 beyond (degrees; THETA + W <= 180; W = 0 a hard cone)\n"
    "    phi about a; time 0; polarisation random linear perpendicular to the direction, as for isotropic.\n";
}

static Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(2); }
      return argv[++i];
    };
    if (a == "--gdml") o.gdml = next();
    else if (a == "--config") o.config = next();
    else if (a == "--out") o.out = next();
    else if (a == "--sens-regex") o.sensRegex = next();
    else if (a == "--sens-aux") o.sensAux = next();
    else if (a == "--dump") o.dump = next();
    else if (a == "--probe") o.probe = next();
    else if (a == "--seed") o.seed = std::stol(next());
    else if (a == "--batch") o.batch = std::stoi(next());
    else if (a == "--chunk") o.chunk = std::stoi(next());
    else if (a == "--nchunks") o.nchunks = std::stoi(next());
    else if (a == "--split-ev") o.splitEV = std::stod(next());
    else if (a == "--no-hits") o.writeHits = false;
    else if (a == "--no-rayleigh") o.rayleigh = false;
    else if (a == "--no-absorption") o.absorption = false;
    else if (a == "--no-wls") o.wls = false;
    else if (a == "--no-mie") o.mie = false;
    else if (a == "--no-fix-velocity") o.fixVelocity = false;
    else if (a == "--no-revive") o.revive = false;
    else if (a == "--wls-profile") o.wlsProfile = next();
    else if (a == "--print-every") o.printEvery = std::stol(next());
    else if (a == "--gun") o.gun = next();
    else if (a == "--gun-mev") o.gunMeV = std::stod(next());
    else if (a == "--gun-pos" || a == "--gun-dir") {
      double v[3]; std::string t = next(); std::replace(t.begin(), t.end(), ',', ' ');
      std::istringstream ss(t); ss >> v[0] >> v[1] >> v[2];
      if (a == "--gun-pos") o.gunPos = G4ThreeVector(v[0] * mm, v[1] * mm, v[2] * mm);
      else o.gunDir = G4ThreeVector(v[0], v[1], v[2]).unit();
    }
    else if (a == "--events") o.events = std::stol(next());
    else if (a == "--cherenkov") o.cherenkov = true;
    else if (a == "--scintillation") o.scint = true;
    else if (a == "--max-cerenkov-per-step") o.maxCerenkovPerStep = std::stoi(next());
    else if (a == "--cerenkov-count") o.cerenkovCount = next();
    else if (a == "--scint-split") o.scintSplit = next();
    else if (a == "--nrefl-before-hit") o.nreflCount = "before_hit";
    else if (a == "--nrefl-count") o.nreflCount = next();
    else if (a == "--geant4-stock") { o.cerenkovCount = "stock"; o.scintSplit = "stock"; o.nreflCount = "stock"; }
    else if (a == "--birks") {
      std::string t = next(); auto k = t.find('=');
      if (k == std::string::npos) { std::cerr << "--birks MAT=KB\n"; std::exit(2); }
      o.birks.emplace_back(t.substr(0, k), std::stod(t.substr(k + 1)));
    }
    else if (a == "--scint-yield") {
      std::string t = next(); auto k = t.find('=');
      if (k == std::string::npos) { std::cerr << "--scint-yield MAT=Y\n"; std::exit(2); }
      o.scintYield.emplace_back(t.substr(0, k), std::stod(t.substr(k + 1)));
    }
    else if (a == "--iscalc") o.iscalc.push_back(next());
    else if (a == "--iscalc-prescale") o.iscalcPrescale = std::stod(next());
    else if (a == "--iscalc-qproton") o.iscalcQProton = std::stod(next());
    else if (a == "--iscalc-temperature") o.iscalcTemperature = std::stod(next());
    else if (a == "--efield" || a == "--step-limit") {
      std::string t = next(); auto k = t.find('=');
      if (k == std::string::npos) { std::cerr << a << " LV=VALUE\n"; std::exit(2); }
      (a == "--efield" ? o.efield : o.stepLimit).emplace_back(t.substr(0, k), std::stod(t.substr(k + 1)));
    }
    else if (a == "--em-tables") o.emTables = next();
    else if (a == "--kill-optical") o.killOptical = true;
    else if (a == "--dump-photons") o.dumpPhotons = true;
    else if (a == "--wcsim-hits") o.wcsimHits = true;
    else if (a == "--wcsim-boundary") o.wcsimBoundary = true;
    else if (a == "-h" || a == "--help") { usage(); std::exit(0); }
    else { std::cerr << "unknown option " << a << "\n"; usage(); std::exit(2); }
  }
  if (o.gdml.empty() || (o.config.empty() && o.dump.empty() && o.gun.empty() && o.probe.empty())) { usage(); std::exit(2); }
  if (o.cerenkovCount != "stock" && o.cerenkovCount != "exact" && o.cerenkovCount != "monotonic_fix") {
    std::cerr << "--cerenkov-count stock|exact|monotonic_fix\n"; std::exit(2); }
  if (o.scintSplit != "stock" && o.scintSplit != "binomial") { std::cerr << "--scint-split stock|binomial\n"; std::exit(2); }
  if (o.nreflCount != "stock" && o.nreflCount != "before_hit") { std::cerr << "--nrefl-count stock|before_hit\n"; std::exit(2); }
  if (!o.gun.empty() && (o.gunMeV <= 0 || o.events <= 0)) { std::cerr << "--gun needs --gun-mev and --events\n"; std::exit(2); }
  if (o.dumpPhotons && o.gun.empty()) { std::cerr << "--dump-photons needs --gun\n"; std::exit(2); }
  if (o.sensRegex.empty() && o.sensAux.empty())
    std::cerr << "WARNING: no sensitive-volume rule given; no hits will be recorded\n";
  for (auto& m : o.iscalc)
    for (auto& [n, v] : o.birks) if (n == m) { std::cerr << "--iscalc " << m << " with --birks\n"; std::exit(2); }
  for (auto& m : o.iscalc)
    for (auto& [n, v] : o.scintYield) if (n == m) { std::cerr << "--iscalc " << m << " with --scint-yield\n"; std::exit(2); }
  if (o.nchunks < 1 || o.chunk < 0 || o.chunk >= o.nchunks) { std::cerr << "bad chunk\n"; std::exit(2); }
  return o;
}

// ---------------------------------------------------------------- sources
struct Spectrum {                 // tabulated pdf in energy, sampled by inverse CDF
  std::vector<double> e, cdf;     // e in G4 units
  double sample(double u) const {
    auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
    size_t j = std::max<size_t>(1, std::min<size_t>(it - cdf.begin(), cdf.size() - 1));
    double c0 = cdf[j - 1], c1 = cdf[j];
    double f = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.5;
    return e[j - 1] + f * (e[j] - e[j - 1]);
  }
};

struct Source {
  G4ThreeVector pos;
  long n = 0;
  double energy = 0;              // used when spec < 0
  int spec = -1;
  std::string desc;
};

static Spectrum readSpectrum(const std::string& fn) {
  std::ifstream in(fn);
  if (!in) { std::cerr << "cannot open spectrum " << fn << "\n"; std::exit(2); }
  std::vector<std::pair<double, double>> pts;
  std::string line;
  while (std::getline(in, line)) {
    auto h = line.find('#'); if (h != std::string::npos) line.resize(h);
    bool nm = false;
    if (line.rfind("nm:", 0) == 0) { nm = true; line = line.substr(3); }
    std::istringstream ss(line); double x, w;
    if (!(ss >> x >> w)) continue;
    double e = nm ? (1239.84198 / x) * eV : x * eV;
    pts.push_back({e, std::max(0.0, w)});
  }
  std::sort(pts.begin(), pts.end());
  if (pts.size() < 2) { std::cerr << "spectrum needs >= 2 points\n"; std::exit(2); }
  Spectrum s; double acc = 0; s.e.push_back(pts[0].first); s.cdf.push_back(0);
  for (size_t i = 1; i < pts.size(); ++i) {   // trapezoid integral of the tabulated pdf
    acc += 0.5 * (pts[i].second + pts[i - 1].second) * (pts[i].first - pts[i - 1].first);
    s.e.push_back(pts[i].first); s.cdf.push_back(acc);
  }
  for (auto& c : s.cdf) c /= acc;
  return s;
}

struct Config {
  std::vector<Source> sources;
  std::vector<Spectrum> spectra;
};

static Config readConfig(const std::string& fn) {
  Config c; std::ifstream in(fn);
  if (!in) { std::cerr << "cannot open config " << fn << "\n"; std::exit(2); }
  std::string line;
  while (std::getline(in, line)) {
    auto h = line.find('#'); if (h != std::string::npos) line.resize(h);
    std::istringstream ss(line); double x, y, z; long n; std::string e;
    if (!(ss >> x >> y >> z >> n >> e)) continue;
    Source s; s.pos = G4ThreeVector(x * mm, y * mm, z * mm); s.n = n;
    if (e.rfind("spectrum:", 0) == 0) {
      c.spectra.push_back(readSpectrum(e.substr(9))); s.spec = (int)c.spectra.size() - 1;
    } else s.energy = std::stod(e) * eV;
    std::ostringstream d; d << x << " " << y << " " << z << " " << n << " " << e; s.desc = d.str();
    c.sources.push_back(s);
  }
  if (c.sources.empty()) { std::cerr << "no sources in config\n"; std::exit(2); }
  return c;
}

// ---------------------------------------------------------------- shared state
struct SensorInfo { std::string path, lv; G4ThreeVector centre; };

static std::unordered_map<const G4VPhysicalVolume*, int> gDaughterIdx;
static std::string pathElem(const G4VPhysicalVolume* pv, int copy) {
  auto it = gDaughterIdx.find(pv);
  return "/" + pv->GetName() + ":" + std::to_string(copy) + "@" +
         std::to_string(it == gDaughterIdx.end() ? 0 : it->second);
}

struct State {
  Options opt;
  Config cfg;
  std::vector<long> evFirst;                  // global event index of each source's first batch
  long nGlobalEvents = 0;
  std::unordered_set<const G4LogicalVolume*> sensLV;
  std::vector<SensorInfo> sensors;
  std::unordered_map<std::string, int> sensorId;
  // per source x sensor counts: total, >= split, < split, wls
  std::map<std::pair<int, int>, std::array<long, 4>> totals;
  std::map<std::string, long> fates;          // photon end states
  std::vector<long> launched;                 // per source
  int curSource = -1; long curGlobalEvent = -1;
  FILE* hits = nullptr; long nHits = 0;
  G4OpBoundaryProcess* boundary = nullptr;
  FILE* steps = nullptr; long nSteps = 0;     // gun mode: every charged step
  FILE* photons = nullptr; long nPhotons = 0; // gun mode, --dump-photons: every optical photon at birth
  std::unordered_map<const G4LogicalVolume*, double> fieldLV;   // --efield [kV/cm]
  std::unordered_set<const G4Material*> iscalcMat;              // --iscalc
  std::map<int, long> bstat;                  // OpBoundary status of every geometry-boundary step
};
static State S;

// the physics mode a run's outputs carry (summary: physics_mode): "correct" = exact Cherenkov count, binomial split,
// reflections before the hit; "stock" = Geant4 10.7's own on all three; anything else "mixed"
static std::string physicsMode() {
  const auto& o = S.opt;
  if (o.cerenkovCount == "exact" && o.scintSplit == "binomial" && o.nreflCount == "before_hit") return "correct";
  if (o.cerenkovCount == "stock" && o.scintSplit == "stock" && o.nreflCount == "stock") return "stock";
  return "mixed";
}

// Startup check: G4Cerenkov 10.7 assumes RINDEX increases with energy (its threshold search, Rindex->GetEnergy(1/beta))
// and takes the vector's last value as its maximum (GetMaxValue). A table that is non-monotonic, or whose last value is
// not its maximum, gets an approximate stock Cherenkov count (and threshold) -- campaign s3x: +52 % for 300 MeV protons
// in LAr. One warning per such material, naming the count in use; also written to the summary (rindex_warning).
static std::vector<std::string> gRindexWarn;
static void rindexCheck() {
  std::cout << "[g4ref] physics mode: " << physicsMode() << " (cerenkov_count " << S.opt.cerenkovCount
            << ", scint_split " << S.opt.scintSplit << ", nrefl_count " << S.opt.nreflCount << ")\n";
  for (auto m : *G4Material::GetMaterialTable()) {
    auto mpt = m->GetMaterialPropertiesTable();
    G4MaterialPropertyVector* v = mpt ? mpt->GetProperty(kRINDEX) : nullptr;
    if (!v || v->GetVectorLength() < 2) continue;
    size_t n = v->GetVectorLength();
    bool nonmono = false;
    double vmax = (*v)[0];
    for (size_t i = 1; i < n; ++i) { if ((*v)[i] < (*v)[i - 1]) nonmono = true; vmax = std::max(vmax, (*v)[i]); }
    bool lastNotMax = (*v)[n - 1] < vmax;
    if (!nonmono && !lastNotMax) continue;
    std::string what = nonmono ? (lastNotMax ? "non-monotonic, last value not its maximum" : "non-monotonic")
                               : "last value not its maximum";
    bool stock = S.opt.cerenkovCount == "stock";
    std::string w = m->GetName() + " (" + what + "; Cerenkov count " + S.opt.cerenkovCount +
                    (stock ? ": APPROXIMATE for this material" : "") + ")";
    gRindexWarn.push_back(w);
    std::cout << "[g4ref] WARNING: RINDEX of " << m->GetName() << " is " << what << " (max " << vmax << ", last "
              << (*v)[n - 1] << "). G4Cerenkov 10.7's count and threshold assume an increasing table; Cerenkov count in "
              << "use: " << S.opt.cerenkovCount
              << (stock ? " -- the stock count is APPROXIMATE for this material" : " (exact for any table)") << "\n";
  }
}

// ---------------------------------------------------------------- npy writer
// structured record: src,i4 evt,i4 sensor,i4 wls,i4 x,y,z,t,E,dx,dy,dz,vx,vy,vz f4  (60 bytes; v: the track's
// creation vertex, mm -- for a WLS photon where it was re-emitted; then the track's own path history:
// cy, cz f4 (where it first crossed the plane x = 0, mm), ncross, nrefl (reflection boundary steps), gas (entered
// a material named GAr) i4), origin i4; then (appended 2026-11-11, the fields before them unchanged) track (the
// photon's Geant4 track id: with evt it joins a hit to PREFIX_photons.npy), nray, nmie (OpRayleigh / OpMieHG steps of
// this track; per track like nrefl, so a WLS photon's count starts at 0) i4
#pragma pack(push, 1)
struct HitRec { int32_t src, evt, sensor, wls; float x, y, z, t, E, dx, dy, dz, vx, vy, vz, cy, cz;
                int32_t ncross, nrefl, gas, origin, track, nray, nmie; };
#pragma pack(pop)
static void writeNpyHeader(FILE* f, long n) {
  std::ostringstream h;
  h << "{'descr': [('src', '<i4'), ('evt', '<i4'), ('sensor', '<i4'), ('wls', '<i4'), "
       "('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('t', '<f4'), ('E', '<f4'), "
       "('dx', '<f4'), ('dy', '<f4'), ('dz', '<f4'), ('vx', '<f4'), ('vy', '<f4'), ('vz', '<f4'), "
       "('cy', '<f4'), ('cz', '<f4'), ('ncross', '<i4'), ('nrefl', '<i4'), ('gas', '<i4'), ('origin', '<i4'), "
       "('track', '<i4'), ('nray', '<i4'), ('nmie', '<i4')], "
       "'fortran_order': False, 'shape': ("
    << n << ",), }";
  std::string s = h.str();
  const size_t total = 512;                   // 10-byte preamble + dict, padded, ends in \n
  while (s.size() + 10 + 1 < total) s += ' ';
  s += '\n';
  unsigned short hl = (unsigned short)s.size();
  std::fseek(f, 0, SEEK_SET);
  std::fwrite("\x93NUMPY\x01\x00", 1, 8, f);
  std::fwrite(&hl, 2, 1, f);
  std::fwrite(s.data(), 1, s.size(), f);
}

// gun mode: one record per charged-particle step (mm, ns, MeV); len: the TRUE step length (multiple scattering
// makes it longer than |x1 - x0|; G4Cerenkov counts photons over it); nc / ns: the Cerenkov / scintillation photons
// Geant4 generated in that step; mat: index into PREFIX_materials.csv; ef: the drift field there [kV/cm] (--efield)
#pragma pack(push, 1)
struct StepRec { int32_t evt, pdg; float q, x0, y0, z0, x1, y1, z1, t0, t1, b0, b1, edep, len; int32_t nc, ns, mat; float ef; };
#pragma pack(pop)
static void writeStepsHeader(FILE* f, long n) {
  std::ostringstream h;
  h << "{'descr': [('evt', '<i4'), ('pdg', '<i4'), ('q', '<f4'), ('x0', '<f4'), ('y0', '<f4'), ('z0', '<f4'), "
       "('x1', '<f4'), ('y1', '<f4'), ('z1', '<f4'), ('t0', '<f4'), ('t1', '<f4'), ('b0', '<f4'), ('b1', '<f4'), "
       "('edep', '<f4'), ('len', '<f4'), ('nc', '<i4'), ('ns', '<i4'), ('mat', '<i4'), ('ef', '<f4')], 'fortran_order': False, 'shape': ("
    << n << ",), }";
  std::string s = h.str();
  const size_t total = 512;
  while (s.size() + 10 + 1 < total) s += ' ';
  s += '\n';
  unsigned short hl = (unsigned short)s.size();
  std::fseek(f, 0, SEEK_SET);
  std::fwrite("\x93NUMPY\x01\x00", 1, 8, f);
  std::fwrite(&hl, 2, 1, f);
  std::fwrite(s.data(), 1, s.size(), f);
  std::fseek(f, 0, SEEK_END);
}

// gun mode, --dump-photons: one record per Cerenkov / scintillation photon at birth (mm, ns, eV); mat: the material it
// was born in (index into PREFIX_materials.csv)
#pragma pack(push, 1)
struct PhotonRec { int32_t evt, track; float x, y, z, t, dx, dy, dz, px, py, pz, E; int32_t origin, mat; };
#pragma pack(pop)
static void writePhotonsHeader(FILE* f, long n) {
  std::ostringstream h;
  h << "{'descr': [('evt', '<i4'), ('track', '<i4'), ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('t', '<f4'), "
       "('dx', '<f4'), ('dy', '<f4'), ('dz', '<f4'), ('polx', '<f4'), ('poly', '<f4'), ('polz', '<f4'), ('E', '<f4'), "
       "('origin', '<i4'), ('mat', '<i4')], 'fortran_order': False, 'shape': ("
    << n << ",), }";
  std::string s = h.str();
  const size_t total = 512;
  while (s.size() + 10 + 1 < total) s += ' ';
  s += '\n';
  unsigned short hl = (unsigned short)s.size();
  std::fseek(f, 0, SEEK_SET);
  std::fwrite("\x93NUMPY\x01\x00", 1, 8, f);
  std::fwrite(&hl, 2, 1, f);
  std::fwrite(s.data(), 1, s.size(), f);
  std::fseek(f, 0, SEEK_END);
}

// ---------------------------------------------------------------- geometry
static bool auxMatch(G4GDMLParser& p, const G4LogicalVolume* lv, const std::string& rule) {
  if (rule.empty()) return false;
  std::string t = rule, v; auto eq = rule.find('=');
  if (eq != std::string::npos) { t = rule.substr(0, eq); v = rule.substr(eq + 1); }
  G4GDMLAuxListType aux = p.GetVolumeAuxiliaryInformation(const_cast<G4LogicalVolume*>(lv));
  for (auto& a : aux) if (std::string(a.type) == t && (v.empty() || std::string(a.value) == v)) return true;
  return false;
}

static void enumerateSensors(const G4VPhysicalVolume* pv, const std::string& parent,
                             const G4RotationMatrix& R, const G4ThreeVector& T) {
  std::string path = parent + pathElem(pv, pv->GetCopyNo());
  G4RotationMatrix Rc = R * pv->GetObjectRotationValue();
  G4ThreeVector Tc = R * pv->GetObjectTranslation() + T;
  const G4LogicalVolume* lv = pv->GetLogicalVolume();
  if (const char* pf = std::getenv("G4REF_PLACEMENTS")) {    // every placement: x_world = R x_local + T (mm)
    static std::ofstream pfo(pf);
    static bool head = (pfo << "path,pv,lv,material,copy,r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz\n", true);
    (void)head;
    pfo << path << "," << pv->GetName() << "," << lv->GetName() << "," << lv->GetMaterial()->GetName() << ","
        << pv->GetCopyNo() << std::setprecision(12);
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) pfo << "," << Rc(i, j);
    pfo << "," << Tc.x() / mm << "," << Tc.y() / mm << "," << Tc.z() / mm << "\n";
  }
  if (!S.opt.dump.empty()) {
    static std::regex dre(S.opt.dump);
    if (std::regex_search(lv->GetName(), dre) || std::regex_search(pv->GetName(), dre)) {
      G4ThreeVector lo, hi; lv->GetSolid()->BoundingLimits(lo, hi);
      G4ThreeVector glo(1e30, 1e30, 1e30), ghi(-1e30, -1e30, -1e30);
      for (int c = 0; c < 8; ++c) {
        G4ThreeVector q((c & 1) ? hi.x() : lo.x(), (c & 2) ? hi.y() : lo.y(), (c & 4) ? hi.z() : lo.z());
        G4ThreeVector g = Rc * q + Tc;
        for (int k = 0; k < 3; ++k) { glo[k] = std::min(glo[k], g[k]); ghi[k] = std::max(ghi[k], g[k]); }
      }
      std::cout << "[dump] " << path << "  lv=" << lv->GetName() << " mat=" << lv->GetMaterial()->GetName()
                << "  centre " << Tc / mm << "  bbox " << glo / mm << " .. " << ghi / mm << "\n";
    }
  }
  if (S.sensLV.count(lv) && !S.sensorId.count(path)) {
    S.sensorId[path] = (int)S.sensors.size();
    S.sensors.push_back({path, lv->GetName(), Tc});
  }
  for (size_t i = 0; i < lv->GetNoDaughters(); ++i) {
    const G4VPhysicalVolume* d = lv->GetDaughter(i);
    if (d->IsReplicated() || d->IsParameterised()) {
      // copies are discovered lazily during tracking (ids appended, centre from touchable)
      continue;
    }
    enumerateSensors(d, path, Rc, Tc);
  }
}

class Detector : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    parser_.SetOverlapCheck(std::getenv("G4REF_CHECK_OVERLAPS") != nullptr);
    parser_.Read(S.opt.gdml, false);
    G4VPhysicalVolume* world = parser_.GetWorldVolume();
    for (auto& [name, kb] : S.opt.birks) {   // read by G4EmSaturation::VisibleEnergyDeposition per step
      G4Material* m = G4Material::GetMaterial(name, false);
      if (!m) { std::cerr << "--birks: no material " << name << "\n"; std::exit(2); }
      m->GetIonisation()->SetBirksConstant(kb * cm / MeV);
      std::cout << "[g4ref] Birks constant of " << name << ": " << m->GetIonisation()->GetBirksConstant() / (cm / MeV)
                << " cm/MeV\n";
    }
    for (auto& [name, y] : S.opt.scintYield) {   // read by G4Scintillation per step
      G4Material* m = G4Material::GetMaterial(name, false);
      G4MaterialPropertiesTable* mpt = m ? m->GetMaterialPropertiesTable() : nullptr;
      if (!mpt || !mpt->ConstPropertyExists("SCINTILLATIONYIELD")) {
        std::cerr << "--scint-yield: no SCINTILLATIONYIELD in " << name << "\n"; std::exit(2);
      }
      double old = mpt->GetConstProperty("SCINTILLATIONYIELD") * MeV;
      mpt->AddConstProperty("SCINTILLATIONYIELD", y / MeV);
      std::cout << "[g4ref] SCINTILLATIONYIELD of " << name << ": " << old << " -> "
                << mpt->GetConstProperty("SCINTILLATIONYIELD") * MeV << " /MeV\n";
    }
    auto lvNamed = [](const std::string& n, const char* opt) {
      std::vector<G4LogicalVolume*> v;
      for (auto lv : *G4LogicalVolumeStore::GetInstance()) if (std::string(lv->GetName()) == n) v.push_back(lv);
      if (v.empty()) { std::cerr << opt << ": no logical volume " << n << "\n"; std::exit(2); }
      return v;
    };
    for (auto& [n, kv] : S.opt.efield)
      for (auto lv : lvNamed(n, "--efield")) {
        S.fieldLV[lv] = kv;
        std::cout << "[g4ref] drift field in " << n << ": " << kv << " kV/cm\n";
      }
    for (auto& [n, mmax] : S.opt.stepLimit)
      for (auto lv : lvNamed(n, "--step-limit")) {
        lv->SetUserLimits(new G4UserLimits(mmax * mm));
        std::cout << "[g4ref] step limit in " << n << ": " << mmax << " mm\n";
      }
    for (auto& name : S.opt.iscalc) {
      G4Material* m = G4Material::GetMaterial(name, false);
      G4MaterialPropertiesTable* mpt = m ? m->GetMaterialPropertiesTable() : nullptr;
      if (!mpt || !mpt->ConstPropertyExists("SCINTILLATIONYIELD") || !mpt->ConstPropertyExists("YIELDRATIO")) {
        std::cerr << "--iscalc: no SCINTILLATIONYIELD / YIELDRATIO in " << name << "\n"; std::exit(2);
      }
      S.iscalcMat.insert(m);
      std::cout << "[g4ref] ISCalcCorrelated scintillation in " << name << " (ScintPreScale " << S.opt.iscalcPrescale
                << ", QProton " << S.opt.iscalcQProton << ", temperature " << S.opt.iscalcTemperature << " K)\n";
    }
    std::regex re;
    bool useRe = !S.opt.sensRegex.empty();
    if (useRe) re = std::regex(S.opt.sensRegex);
    // a logical volume is sensitive if its name, or the name of any placement of it, matches
    for (auto lv : *G4LogicalVolumeStore::GetInstance()) {
      if ((useRe && std::regex_search(lv->GetName(), re)) || auxMatch(parser_, lv, S.opt.sensAux))
        S.sensLV.insert(lv);
    }
    if (useRe)
      for (auto pv : *G4PhysicalVolumeStore::GetInstance())
        if (std::regex_search(pv->GetName(), re)) S.sensLV.insert(pv->GetLogicalVolume());
    std::cout << "[g4ref] sensitive logical volumes: " << S.sensLV.size() << "\n";
    for (auto lv : S.sensLV)
      std::cout << "[g4ref]   " << lv->GetName() << "  material " << lv->GetMaterial()->GetName()
                << (lv->GetMaterial()->GetMaterialPropertiesTable() &&
                    lv->GetMaterial()->GetMaterialPropertiesTable()->GetProperty("RINDEX")
                        ? "  (has RINDEX)" : "  (NO RINDEX: photons are killed at its boundary"
                                              " and never enter -> never recorded)")
                << "\n";
    for (auto lv : *G4LogicalVolumeStore::GetInstance())   // placement index within its mother
      for (size_t i = 0; i < lv->GetNoDaughters(); ++i) gDaughterIdx[lv->GetDaughter(i)] = (int)i;
    enumerateSensors(world, "", G4RotationMatrix(), G4ThreeVector());
    std::cout << "[g4ref] sensitive placements enumerated: " << S.sensors.size() << "\n";
    // optics inventory
    std::cout << "[g4ref] materials with optical properties:\n";
    for (auto m : *G4Material::GetMaterialTable()) {
      auto mpt = m->GetMaterialPropertiesTable();
      if (!mpt) continue;
      std::cout << "[g4ref]   " << m->GetName() << ":";
      for (auto& kv : *mpt->GetPropertyMap())
        std::cout << " " << mpt->GetMaterialPropertyNames()[kv.first];
      for (auto& kv : *mpt->GetConstPropertyMap())
        std::cout << " " << mpt->GetMaterialConstPropertyNames()[kv.first] << "(const)";
      std::cout << "\n";
    }
    rindexCheck();
    std::cout << "[g4ref] skin surfaces: " << G4LogicalSkinSurface::GetNumberOfSkinSurfaces()
              << "  border surfaces: " << G4LogicalBorderSurface::GetNumberOfBorderSurfaces() << "\n";
    if (!S.opt.dump.empty()) std::exit(0);
    return world;
  }
 private:
  G4GDMLParser parser_;
};

// ---------------------------------------------------------------- primaries
static void reseed(long g) {
  long seeds[5] = {S.opt.seed, g, 0x5eed, 7, 0};
  G4Random::getTheEngine()->setSeeds(seeds, 4);
}

static G4ThreeVector isotropic() {
  double cz = 2 * G4UniformRand() - 1, ph = twopi * G4UniformRand();
  double sz = std::sqrt(std::max(0.0, 1 - cz * cz));
  return G4ThreeVector(sz * std::cos(ph), sz * std::sin(ph), cz);
}

// G4REF_PROFILE (point sources): an emission profile about a unit axis a in place of the isotropic draw (usage()).
// Per photon, after the energy draw: laser u1, u2 (the spot offset); led / cone u1 (cos theta), u2 (phi); then the
// polarisation angle as always. Without G4REF_PROFILE nothing changes (the isotropic draw is untouched).
struct Profile { int kind = 0; double p1 = 0, p2 = 0; G4ThreeVector a, e1, e2; std::string desc; };   // 1 laser 2 led 3 cone
static Profile gProf;
static void parseProfile() {
  const char* env = std::getenv("G4REF_PROFILE");
  if (!env) return;
  std::string t(env); auto c = t.find(':');
  std::string kind = c == std::string::npos ? t : t.substr(0, c);
  std::vector<double> v;
  if (c != std::string::npos) {
    std::string r = t.substr(c + 1); std::replace(r.begin(), r.end(), ',', ' ');
    std::istringstream ss(r); double x; while (ss >> x) v.push_back(x);
  }
  size_t want = kind == "cone" ? 5 : 4;
  if ((kind != "laser" && kind != "led" && kind != "cone") || v.size() != want) {
    std::cerr << "G4REF_PROFILE: laser:SIGMA,ax,ay,az | led:M,ax,ay,az | cone:THETA,W,ax,ay,az (got " << t << ")\n";
    std::exit(2);
  }
  if (std::getenv("G4REF_PENCIL")) { std::cerr << "G4REF_PROFILE with G4REF_PENCIL\n"; std::exit(2); }
  gProf.kind = kind == "laser" ? 1 : (kind == "led" ? 2 : 3);
  gProf.p1 = v[0]; gProf.p2 = kind == "cone" ? v[1] : 0.0;
  G4ThreeVector a(v[want - 3], v[want - 2], v[want - 1]);
  if (a.mag() <= 0) { std::cerr << "G4REF_PROFILE: zero axis\n"; std::exit(2); }
  gProf.a = a.unit(); gProf.e1 = gProf.a.orthogonal().unit(); gProf.e2 = gProf.a.cross(gProf.e1);
  if ((gProf.kind == 1 && gProf.p1 < 0) || (gProf.kind == 2 && gProf.p1 < 0) ||
      (gProf.kind == 3 && (gProf.p1 < 0 || gProf.p2 < 0 || gProf.p1 + gProf.p2 > 180))) {
    std::cerr << "G4REF_PROFILE: parameter out of range (" << t << ")\n"; std::exit(2);
  }
  gProf.desc = t;
  std::cout << "[g4ref] emission profile " << t << "\n";
}
// one photon of the profile: its direction, and its position offset from the source (laser only)
static void profileDraw(G4ThreeVector& d, G4ThreeVector& off) {
  double u1 = G4UniformRand(), u2 = G4UniformRand();
  double ph = twopi * u2;
  off = G4ThreeVector();
  if (gProf.kind == 1) {
    double r = gProf.p1 * mm * std::sqrt(-2.0 * std::log(1.0 - u1));
    off = r * (std::cos(ph) * gProf.e1 + std::sin(ph) * gProf.e2);
    d = gProf.a;
    return;
  }
  double ct;
  if (gProf.kind == 2) ct = std::pow(u1, 1.0 / (gProf.p1 + 1.0));
  else {
    double ma = std::cos(gProf.p1 * deg), mb = std::cos((gProf.p1 + gProf.p2) * deg), D = ma - mb;
    double A1 = 1.0 - ma, A2 = 0.5 * D, s = u1 * (A1 + A2);
    ct = (s < A1 || D <= 0) ? 1.0 - s : mb + std::sqrt(std::max(0.0, D * D - 2.0 * D * (s - A1)));
  }
  ct = std::min(1.0, std::max(-1.0, ct));
  double st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
  d = (st * std::cos(ph) * gProf.e1 + st * std::sin(ph) * gProf.e2 + ct * gProf.a).unit();
}

class Primaries : public G4VUserPrimaryGeneratorAction {
 public:
  void GeneratePrimaries(G4Event* ev) override {
    // event ids in this job map to global events chunk, chunk+n, chunk+2n, ...
    long g = S.opt.chunk + (long)ev->GetEventID() * S.opt.nchunks;
    if (!S.opt.gun.empty()) {                 // one charged primary per event
      S.curSource = 0; S.curGlobalEvent = g;
      reseed(g);
      auto def = G4ParticleTable::GetParticleTable()->FindParticle(S.opt.gun);
      if (!def) { std::cerr << "unknown particle " << S.opt.gun << "\n"; std::exit(2); }
      auto vtx = new G4PrimaryVertex(S.opt.gunPos, 0.);
      auto p = new G4PrimaryParticle(def);
      p->SetKineticEnergy(S.opt.gunMeV * MeV);
      p->SetMomentumDirection(S.opt.gunDir);
      vtx->SetPrimary(p);
      ev->AddPrimaryVertex(vtx);
      S.launched[0] += 1;
      return;
    }
    int s = (int)(std::upper_bound(S.evFirst.begin(), S.evFirst.end(), g) - S.evFirst.begin()) - 1;
    const Source& src = S.cfg.sources[s];
    long b = g - S.evFirst[s];
    long nThis = std::min<long>(S.opt.batch, src.n - b * S.opt.batch);
    S.curSource = s; S.curGlobalEvent = g;
    reseed(g);
    auto vtx = new G4PrimaryVertex(src.pos, 0.);
    for (long i = 0; i < nThis; ++i) {
      double E = (src.spec >= 0) ? S.cfg.spectra[src.spec].sample(G4UniformRand()) : src.energy;
      if (gProf.kind) {                       // G4REF_PROFILE: one vertex per photon (the laser spot moves it)
        G4ThreeVector d, off;
        profileDraw(d, off);
        G4ThreeVector e1 = d.orthogonal().unit(), e2 = d.cross(e1);
        double ph = twopi * G4UniformRand();
        G4ThreeVector pol = std::cos(ph) * e1 + std::sin(ph) * e2;
        auto v1 = new G4PrimaryVertex(src.pos + off, 0.);
        auto p = new G4PrimaryParticle(G4OpticalPhoton::Definition());
        p->SetKineticEnergy(E);
        p->SetMomentumDirection(d);
        p->SetPolarization(pol.x(), pol.y(), pol.z());
        v1->SetPrimary(p);
        ev->AddPrimaryVertex(v1);
        continue;
      }
      G4ThreeVector d = isotropic();
      if (const char* pd = std::getenv("G4REF_PENCIL")) {   // diagnostic: every photon along dx,dy,dz
        double a[3]; std::sscanf(pd, "%lf,%lf,%lf", a, a + 1, a + 2);
        d = G4ThreeVector(a[0], a[1], a[2]).unit();
      }
      G4ThreeVector e1 = d.orthogonal().unit(), e2 = d.cross(e1);
      double ph = twopi * G4UniformRand();
      G4ThreeVector pol = std::cos(ph) * e1 + std::sin(ph) * e2;
      auto p = new G4PrimaryParticle(G4OpticalPhoton::Definition());
      p->SetKineticEnergy(E);
      p->SetMomentumDirection(d);
      p->SetPolarization(pol.x(), pol.y(), pol.z());
      vtx->SetPrimary(p);
    }
    if (gProf.kind) delete vtx;
    else ev->AddPrimaryVertex(vtx);
    S.launched[s] += nThis;
  }
};

// ---------------------------------------------------------------- stepping
static const char* bStatus(int st) {
  switch (st) {
    case Undefined: return "Undefined";
    case Transmission: return "Transmission";
    case FresnelRefraction: return "FresnelRefraction";
    case FresnelReflection: return "FresnelReflection";
    case TotalInternalReflection: return "TotalInternalReflection";
    case LambertianReflection: return "LambertianReflection";
    case LobeReflection: return "LobeReflection";
    case SpikeReflection: return "SpikeReflection";
    case BackScattering: return "BackScattering";
    case Absorption: return "Absorption";
    case Detection: return "Detection";
    case NotAtBoundary: return "NotAtBoundary";
    case SameMaterial: return "SameMaterial";
    case StepTooSmall: return "StepTooSmall";
    case NoRINDEX: return "NoRINDEX";
    default: return "Other";
  }
}

static int sensorFor(const G4TouchableHandle& th) {
  std::string path;
  int depth = th->GetHistoryDepth();
  for (int i = depth; i >= 0; --i)
    path += pathElem(th->GetVolume(i), th->GetCopyNumber(i));
  auto it = S.sensorId.find(path);
  if (it != S.sensorId.end()) return it->second;
  int id = (int)S.sensors.size();
  S.sensorId[path] = id;
  G4ThreeVector c = th->GetHistory()->GetTopTransform().Inverse().TransformPoint(G4ThreeVector());
  S.sensors.push_back({path, th->GetVolume()->GetLogicalVolume()->GetName(), c});
  return id;
}

// the origin of each optical photon: 0 primary, 2 Cerenkov, 3 Scintillation, 1 other; a WLS photon inherits its
// parent's (so a TPB-shifted scintillation photon is still 3); -1 unknown
static std::unordered_map<int, int> gOrigin;
static long gOriginEvt = -1;
static int originOf(const G4Track* tr) {
  auto it = gOrigin.find(tr->GetTrackID());
  return it == gOrigin.end() ? -1 : it->second;
}
class Tracking : public G4UserTrackingAction {
 public:
  void PreUserTrackingAction(const G4Track* tr) override {
    if (gOriginEvt != S.curGlobalEvent) { gOrigin.clear(); gOriginEvt = S.curGlobalEvent; }
    if (tr->GetDefinition() != G4OpticalPhoton::Definition()) return;
    const G4VProcess* c = tr->GetCreatorProcess();
    int o = 0;
    bool born = false;                        // made by Cerenkov / Scintillation itself (not a WLS daughter)
    if (c) {
      const std::string& n = c->GetProcessName();
      if (n == "Cerenkov") { o = 2; born = true; }
      else if (n == "Scintillation") { o = 3; born = true; }
      else if (n == "OpWLS" || n == "OpWLS2") {
        auto it = gOrigin.find(tr->GetParentID());
        o = it == gOrigin.end() ? -1 : it->second;
      } else o = 1;
    }
    gOrigin[tr->GetTrackID()] = o;
    if (S.photons && born) {                 // --dump-photons: the photon as born (no random numbers drawn)
      const G4VTouchable* th = tr->GetTouchable();
      const G4VPhysicalVolume* pv = th ? th->GetVolume() : nullptr;
      G4ThreeVector x = tr->GetPosition(), d = tr->GetMomentumDirection(), pol = tr->GetPolarization();
      PhotonRec r{(int32_t)S.curGlobalEvent, (int32_t)tr->GetTrackID(),
                  (float)(x.x() / mm), (float)(x.y() / mm), (float)(x.z() / mm), (float)(tr->GetGlobalTime() / ns),
                  (float)d.x(), (float)d.y(), (float)d.z(), (float)pol.x(), (float)pol.y(), (float)pol.z(),
                  (float)(tr->GetKineticEnergy() / eV), o,
                  pv ? (int32_t)pv->GetLogicalVolume()->GetMaterial()->GetIndex() : -1};
      std::fwrite(&r, sizeof(r), 1, S.photons); S.nPhotons++;
    }
  }
};

// the current track's path history (tracks are followed one at a time, to the end)
// the drift field of a step [kV/cm]: its pre-step volume's (--efield; a Geant4 step lies in one volume). LArSoft
// (ISCalcCorrelated::EFieldAtStep) takes the field at the step's midpoint, zero outside the TPC active volume.
static double fieldAt(const G4Step& st) {
  auto it = S.fieldLV.find(st.GetPreStepPoint()->GetTouchableHandle()->GetVolume()->GetLogicalVolume());
  return it == S.fieldLV.end() ? 0.0 : it->second;
}

// LArSoft's ISCalcCorrelated (larsim v10_23_00 IonizationScintillation/ISCalcCorrelated.cxx, ISCalc.cxx) with SBND's
// configuration, every constant from the public fcl:
//   sbndcode v10_23_00 simulationservices_sbnd.fcl: UseEllipsModBoxRecomb true, UseModLarqlRecomb true, QProton 0.71;
//   larsim v10_23_00 simulationservices.fcl standard_largeantparameters: EllipsModBox A 0.904, B 0.204, R 1.25,
//     ModBox A 0.930, B 0.212 (the step-angle-undefined fallback), LArQL chi0 A..D, alpha, beta, QAlpha 0.43, Wph 19.5 eV;
//     LArG4Parameters.cc: UseBinomialFlucts true (default);
//   larcoreobj PhysicalConstants.h: kGeVToElectrons 4.237e7 (Wion 23.6 eV);
//   sbndcode detectorproperties_sbnd.fcl: Temperature 88.4 K (density 1.928 - 0.00615 T g/cm3,
//     DetectorPropertiesStandard::Density); opticalproperties_sbnd.fcl: ScintPreScale 0.0392, ScintByParticleType true;
//   lardataalg larproperties.fcl: the fast ratios (Muon / Pion / Kaon 0.23, Proton 0.29, Alpha 0.56, Electron 0.27).
// The step's MEAN photon count (electrons at their binomial mean, floor(E / Wion) R: CLHEP::RandBinomial takes a long)
// is set as G4Scintillation's yield for that step, and the particle's fast ratio as its YIELDRATIO; G4Scintillation
// then draws the count and the components as always (Birks off: G4EmSaturation at kB = 0 returns the deposit).
namespace iscalc {
const double A = 0.904, B = 0.204, R = 1.25, mbA = 0.930, mbB = 0.212;
const double chi0A = 0.00338427, chi0B = -6.57037, chi0C = 1.88418, chi0D = 0.000129379, qlAlpha = 0.0372, qlBeta = 0.0124;
const double QAlpha = 0.43, Wph = 19.5e-6, Wion = 1.0 / 4.237e7 * 1e3;
double photons(double E, double ds_cm, double ef, double cosphi, bool phi_ok, int pdg) {   // E [MeV]
  const double QProton = S.opt.iscalcQProton, prescale = S.opt.iscalcPrescale;      // --iscalc-* (default SBND)
  const double rho = -0.00615 * S.opt.iscalcTemperature + 1.928;                     // g/cm3
  double num_ions = E >= Wion ? E / Wion : 0.0, num_quanta = E / Wph;
  double dEdx = ds_cm <= 0.0 ? 0.0 : E / ds_cm;
  dEdx = dEdx < 1.0 ? 1.0 : dEdx;
  double recomb = 0.0;
  if (ef > 0.0) {
    if (!phi_ok) { double Xi = mbB / rho * dEdx / ef; recomb = std::log(mbA + Xi) / Xi; }
    else {
      double sinphi = std::sqrt(std::max(0.0, 1.0 - cosphi * cosphi));
      double Be = B / rho * dEdx / (ef * std::hypot(sinphi, cosphi / R));
      recomb = std::log(A + Be) / Be;
    }
  }
  if (pdg != 1000020040)
    recomb += chi0A / (chi0B + std::exp(chi0C + chi0D * dEdx)) * std::exp(-ef / (qlAlpha * std::log(dEdx) + qlBeta));
  recomb = std::min(1.0, std::max(0.0, recomb));
  double ne = num_ions > 0.0 ? std::floor(num_ions) * recomb : 0.0;
  double nph = (num_quanta - ne) * prescale;
  if (pdg == 2212) nph *= QProton;
  if (pdg == 1000020040) nph *= QAlpha;
  return nph;
}
double fastRatio(int pdg) {
  switch (pdg) {
    case 2212: return 0.29;
    case 13: case -13: case 211: case -211: case 321: case -321: return 0.23;
    case 1000020040: return 0.56;
    default: return 0.27;
  }
}
}  // namespace iscalc

// ---------------------------------------------------------------- exact-physics variants (campaign s3x)
// G4RefCerenkov (--cerenkov-count exact|monotonic_fix): G4Cerenkov 10.7 with its mean photon number replaced.
// G4Cerenkov::GetAverageNumberOfPhotons is not virtual in 10.7, so the two virtual methods that call it
// (PostStepGetPhysicalInteractionLength: the step limit; PostStepDoIt: the photons) are copied here from
// G4Cerenkov.cc 10.7.4 verbatim except that they call Count(); private members are read through their getters.
// Not used with --cerenkov-count stock (the stock G4Cerenkov runs, so stock outputs are unchanged).
class G4RefCerenkov : public G4Cerenkov {
 public:
  explicit G4RefCerenkov(bool exact) : G4Cerenkov("Cerenkov"), fExact(exact) {}
  // photons per mm: 369.81/(eV cm) q^2 sum over RINDEX segments of the part above threshold (n > 1/beta, n linear in
  // E) of [dE - beta^-2 int dE / n^2]; exact: int_a^b dE / n^2 = (b - a) / (n_a n_b) (n linear); monotonic_fix:
  // the trapezoid (b - a) (1/n_a^2 + 1/n_b^2) / 2 (10.7's CerenkovAngleIntegrals). Zero where 10.7 makes none (no
  // CerenkovAngleIntegrals: the table's first value <= 1).
  G4double Count(G4double charge, G4double beta, const G4Material* m, G4MaterialPropertyVector* Rindex) const {
    const G4double Rfact = 369.81 / (eV * cm);
    if (beta <= 0.0) return 0.0;
    auto cai = (G4PhysicsOrderedFreeVector*)((*thePhysicsTable)(m->GetIndex()));
    if (!cai || !cai->IsFilledVectorExist()) return 0.0;
    const G4double bi = 1.0 / beta, nt = bi;           // threshold index
    G4double sum = 0.0;
    size_t N = Rindex->GetVectorLength();
    for (size_t k = 0; k + 1 < N; ++k) {
      G4double e0 = Rindex->Energy(k), e1 = Rindex->Energy(k + 1), n0 = (*Rindex)[k], n1 = (*Rindex)[k + 1];
      if (e1 <= e0) continue;
      bool a0 = n0 > nt, a1 = n1 > nt;
      if (!a0 && !a1) continue;
      G4double a = e0, b = e1, na = n0, nb = n1;
      if (!a0 || !a1) {                                // crossing inside the segment (n linear)
        G4double ec = e0 + (nt - n0) * (e1 - e0) / (n1 - n0);
        if (a0) { b = ec; nb = nt; } else { a = ec; na = nt; }
      }
      G4double inv2 = fExact ? (b - a) / (na * nb) : 0.5 * (b - a) * (1.0 / (na * na) + 1.0 / (nb * nb));
      sum += (b - a) - bi * bi * inv2;
    }
    return Rfact * charge / eplus * charge / eplus * sum;
  }

  G4double PostStepGetPhysicalInteractionLength(const G4Track& aTrack, G4double, G4ForceCondition* condition) override {
    *condition = NotForced;
    G4double StepLimit = DBL_MAX;
    const G4Material* aMaterial = aTrack.GetMaterial();
    G4int materialIndex = aMaterial->GetIndex();
    if (!(*thePhysicsTable)[materialIndex]) return StepLimit;
    const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
    const G4MaterialCutsCouple* couple = aTrack.GetMaterialCutsCouple();
    G4double kineticEnergy = aParticle->GetKineticEnergy();
    const G4ParticleDefinition* particleType = aParticle->GetDefinition();
    G4double mass = particleType->GetPDGMass();
    G4double beta = aParticle->GetTotalMomentum() / aParticle->GetTotalEnergy();
    G4double gamma = aParticle->GetTotalEnergy() / mass;
    G4MaterialPropertiesTable* mpt = aMaterial->GetMaterialPropertiesTable();
    G4MaterialPropertyVector* Rindex = mpt ? mpt->GetProperty(kRINDEX) : nullptr;
    if (!Rindex) return StepLimit;
    G4double nMax = Rindex->GetMaxValue();
    G4double BetaMin = 1. / nMax;
    if (BetaMin >= 1.) return StepLimit;
    G4double GammaMin = 1. / std::sqrt(1. - BetaMin * BetaMin);
    if (gamma < GammaMin) return StepLimit;
    G4double kinEmin = mass * (GammaMin - 1.);
    G4double RangeMin = G4LossTableManager::Instance()->GetRange(particleType, kinEmin, couple);
    G4double Range = G4LossTableManager::Instance()->GetRange(particleType, kineticEnergy, couple);
    G4double Step = Range - RangeMin;
    static const G4double minAllowedStep = G4ThreeVector::getTolerance();
    if (Step < minAllowedStep) return StepLimit;
    if (Step < StepLimit) StepLimit = Step;
    if (GetMaxNumPhotonsPerStep() > 0) {
      const G4double charge = aParticle->GetDefinition()->GetPDGCharge();
      G4double MeanNumberOfPhotons = Count(charge, beta, aMaterial, Rindex);
      Step = 0.;
      if (MeanNumberOfPhotons > 0.0) Step = GetMaxNumPhotonsPerStep() / MeanNumberOfPhotons;
      if (Step > 0. && Step < StepLimit) StepLimit = Step;
    }
    if (GetMaxBetaChangePerStep() > 0.) {
      G4double dedx = G4LossTableManager::Instance()->GetDEDX(particleType, kineticEnergy, couple);
      G4double mb = GetMaxBetaChangePerStep();
      G4double deltaGamma = gamma - 1. / std::sqrt(1. - beta * beta * (1. - mb) * (1. - mb));
      Step = mass * deltaGamma / dedx;
      if (Step > 0. && Step < StepLimit) StepLimit = Step;
    }
    *condition = StronglyForced;
    return StepLimit;
  }

  G4VParticleChange* PostStepDoIt(const G4Track& aTrack, const G4Step& aStep) override {
    aParticleChange.Initialize(aTrack);
    const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
    const G4Material* aMaterial = aTrack.GetMaterial();
    G4StepPoint* pPreStepPoint = aStep.GetPreStepPoint();
    G4StepPoint* pPostStepPoint = aStep.GetPostStepPoint();
    G4ThreeVector x0 = pPreStepPoint->GetPosition();
    G4ThreeVector p0 = aStep.GetDeltaPosition().unit();
    G4double t0 = pPreStepPoint->GetGlobalTime();
    G4MaterialPropertiesTable* MPT = aMaterial->GetMaterialPropertiesTable();
    if (!MPT) return pParticleChange;
    G4MaterialPropertyVector* Rindex = MPT->GetProperty(kRINDEX);
    if (!Rindex) return pParticleChange;
    G4double charge = aParticle->GetDefinition()->GetPDGCharge();
    G4double beta = (pPreStepPoint->GetBeta() + pPostStepPoint->GetBeta()) * 0.5;
    G4double MeanNumberOfPhotons = Count(charge, beta, aMaterial, Rindex);
    if (MeanNumberOfPhotons <= 0.0) { aParticleChange.SetNumberOfSecondaries(0); return pParticleChange; }
    G4double step_length = aStep.GetStepLength();
    MeanNumberOfPhotons = MeanNumberOfPhotons * step_length;
    G4int nph = (G4int)G4Poisson(MeanNumberOfPhotons);
    if (nph <= 0 || !GetStackPhotons()) { aParticleChange.SetNumberOfSecondaries(0); return pParticleChange; }
    aParticleChange.SetNumberOfSecondaries(nph);
    if (GetTrackSecondariesFirst() && aTrack.GetTrackStatus() == fAlive) aParticleChange.ProposeTrackStatus(fSuspend);
    G4double Pmin = Rindex->GetMinLowEdgeEnergy();
    G4double Pmax = Rindex->GetMaxLowEdgeEnergy();
    G4double dp = Pmax - Pmin;
    G4double nMax = Rindex->GetMaxValue();
    G4double BetaInverse = 1. / beta;
    G4double maxCos = BetaInverse / nMax;
    G4double maxSin2 = (1.0 - maxCos) * (1.0 + maxCos);
    G4double beta1 = pPreStepPoint->GetBeta();
    G4double beta2 = pPostStepPoint->GetBeta();
    G4double MeanNumberOfPhotons1 = Count(charge, beta1, aMaterial, Rindex);
    G4double MeanNumberOfPhotons2 = Count(charge, beta2, aMaterial, Rindex);
    for (G4int i = 0; i < nph; ++i) {
      G4double rand, sampledEnergy, sampledRI, cosTheta, sin2Theta;
      do {
        rand = G4UniformRand();
        sampledEnergy = Pmin + rand * dp;
        sampledRI = Rindex->Value(sampledEnergy);
        cosTheta = BetaInverse / sampledRI;
        sin2Theta = (1.0 - cosTheta) * (1.0 + cosTheta);
        rand = G4UniformRand();
      } while (rand * maxSin2 > sin2Theta);
      rand = G4UniformRand();
      G4double phi = twopi * rand;
      G4double sinPhi = std::sin(phi), cosPhi = std::cos(phi);
      G4double sinTheta = std::sqrt(sin2Theta);
      G4ParticleMomentum photonMomentum(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
      photonMomentum.rotateUz(p0);
      G4ThreeVector photonPolarization(cosTheta * cosPhi, cosTheta * sinPhi, -sinTheta);
      photonPolarization.rotateUz(p0);
      auto aCerenkovPhoton = new G4DynamicParticle(G4OpticalPhoton::OpticalPhoton(), photonMomentum);
      aCerenkovPhoton->SetPolarization(photonPolarization);
      aCerenkovPhoton->SetKineticEnergy(sampledEnergy);
      G4double NumberOfPhotons, N;
      do {
        rand = G4UniformRand();
        NumberOfPhotons = MeanNumberOfPhotons1 - rand * (MeanNumberOfPhotons1 - MeanNumberOfPhotons2);
        N = G4UniformRand() * std::max(MeanNumberOfPhotons1, MeanNumberOfPhotons2);
      } while (N > NumberOfPhotons);
      G4double delta = rand * aStep.GetStepLength();
      G4double deltaTime = delta / (pPreStepPoint->GetVelocity() +
                                    rand * (pPostStepPoint->GetVelocity() - pPreStepPoint->GetVelocity()) * 0.5);
      auto aSecondaryTrack = new G4Track(aCerenkovPhoton, t0 + deltaTime, x0 + rand * aStep.GetDeltaPosition());
      aSecondaryTrack->SetTouchableHandle(aStep.GetPreStepPoint()->GetTouchableHandle());
      aSecondaryTrack->SetParentID(aTrack.GetTrackID());
      aParticleChange.AddSecondary(aSecondaryTrack);
    }
    return pParticleChange;
  }

 private:
  bool fExact;
};

// G4RefScintillation: G4Scintillation 10.7 with an optional binomial fast/slow split (--scint-split binomial).
// stock: forwards to G4Scintillation (bitwise as before). binomial: G4Scintillation::PostStepDoIt's old-method
// (FAST/SLOWCOMPONENT) path copied from G4Scintillation.cc 10.7.4, the only change being the fast count:
// Binomial(N, min(YIELDRATIO, 1)) (or fExcitationRatio when that is neither 0 nor 1, as 10.7) instead of
// int(YIELDRATIO N). The enhanced-time-constants path and a finite rise time are refused (not used by the campaign).
class G4RefScintillation : public G4Scintillation {
 public:
  G4RefScintillation() : G4Scintillation("Scintillation") {}
  G4VParticleChange* PostStepDoIt(const G4Track& t, const G4Step& s) override {
    if (S.opt.scintSplit != "binomial") return G4Scintillation::PostStepDoIt(t, s);
    return Binomial(t, s);
  }
  G4VParticleChange* AtRestDoIt(const G4Track& t, const G4Step& s) override {
    if (S.opt.scintSplit != "binomial") return G4Scintillation::AtRestDoIt(t, s);
    return G4RefScintillation::PostStepDoIt(t, s);   // 10.7's AtRestDoIt is its PostStepDoIt
  }
 private:
  G4VParticleChange* Binomial(const G4Track& aTrack, const G4Step& aStep) {
    if (GetEnhancedTimeConstants() || GetFiniteRiseTime())
      G4Exception("g4ref", "scintsplit", FatalException,
                  "--scint-split binomial: enhanced time constants / finite rise time not implemented");
    aParticleChange.Initialize(aTrack);
    const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
    const G4Material* aMaterial = aTrack.GetMaterial();
    G4StepPoint* pPreStepPoint = aStep.GetPreStepPoint();
    G4StepPoint* pPostStepPoint = aStep.GetPostStepPoint();
    G4ThreeVector x0 = pPreStepPoint->GetPosition();
    G4double t0 = pPreStepPoint->GetGlobalTime();
    G4double TotalEnergyDeposit = aStep.GetTotalEnergyDeposit();
    G4MaterialPropertiesTable* MPT = aMaterial->GetMaterialPropertiesTable();
    if (!MPT) return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    G4int N_timeconstants = 1;
    G4MaterialPropertyVector* Fast_Intensity = MPT->GetProperty(kFASTCOMPONENT);
    G4MaterialPropertyVector* Slow_Intensity = MPT->GetProperty(kSLOWCOMPONENT);
    if (!Fast_Intensity && !Slow_Intensity) return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    if (Fast_Intensity && Slow_Intensity) N_timeconstants = 2;
    G4double ResolutionScale = MPT->GetConstProperty(kRESOLUTIONSCALE);
    G4double MeanNumberOfPhotons;
    if (GetScintillationByParticleType()) MeanNumberOfPhotons = GetScintillationYieldByParticleType(aTrack, aStep);
    else {
      MeanNumberOfPhotons = MPT->GetConstProperty(kSCINTILLATIONYIELD) * GetScintillationYieldFactor();
      if (GetSaturation()) MeanNumberOfPhotons *= GetSaturation()->VisibleEnergyDepositionAtAStep(&aStep);
      else MeanNumberOfPhotons *= TotalEnergyDeposit;
    }
    G4int nph;
    if (MeanNumberOfPhotons > 10.) {
      G4double sigma = ResolutionScale * std::sqrt(MeanNumberOfPhotons);
      nph = G4int(G4RandGauss::shoot(MeanNumberOfPhotons, sigma) + 0.5);
    } else nph = G4int(G4Poisson(MeanNumberOfPhotons));
    if (nph <= 0 || !GetStackPhotons()) {
      aParticleChange.SetNumberOfSecondaries(0);
      return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }
    aParticleChange.SetNumberOfSecondaries(nph);
    if (GetTrackSecondariesFirst() && aTrack.GetTrackStatus() == fAlive) aParticleChange.ProposeTrackStatus(fSuspend);
    G4int materialIndex = aMaterial->GetIndex();
    size_t numPhot = nph;
    G4double scintTime = 0.;
    G4PhysicsOrderedFreeVector* scintIntegral = nullptr;
    for (G4int scnt = 0; scnt < N_timeconstants; ++scnt) {
      if (scnt == 0) {
        if (N_timeconstants == 1) {
          if (Fast_Intensity) {
            scintTime = MPT->GetConstProperty(kFASTTIMECONSTANT);
            scintIntegral = (G4PhysicsOrderedFreeVector*)((*fIntegralTable1)(materialIndex));
          }
          if (Slow_Intensity) {
            scintTime = MPT->GetConstProperty(kSLOWTIMECONSTANT);
            scintIntegral = (G4PhysicsOrderedFreeVector*)((*fIntegralTable3)(materialIndex));
          }
        } else {
          G4double yieldRatio = MPT->GetConstProperty(kYIELDRATIO);
          G4double ex = GetScintillationExcitationRatio();
          G4double p = (ex == 1.0 || ex == 0.0) ? std::min(yieldRatio, 1.0) : std::min(ex, 1.0);
          numPhot = 0;                                   // the one change: each photon fast with probability p
          for (G4int j = 0; j < nph; ++j) if (G4UniformRand() < p) ++numPhot;
          scintTime = MPT->GetConstProperty(kFASTTIMECONSTANT);
          scintIntegral = (G4PhysicsOrderedFreeVector*)((*fIntegralTable1)(materialIndex));
        }
      } else {
        numPhot = nph - numPhot;
        scintTime = MPT->GetConstProperty(kSLOWTIMECONSTANT);
        scintIntegral = (G4PhysicsOrderedFreeVector*)((*fIntegralTable3)(materialIndex));
      }
      if (!scintIntegral) continue;
      G4double CIImax = scintIntegral->GetMaxValue();
      for (size_t i = 0; i < numPhot; ++i) {
        G4double CIIvalue = G4UniformRand() * CIImax;
        G4double sampledEnergy = scintIntegral->GetEnergy(CIIvalue);
        G4double cost = 1. - 2. * G4UniformRand();
        G4double sint = std::sqrt((1. - cost) * (1. + cost));
        G4double phi = twopi * G4UniformRand();
        G4double sinp = std::sin(phi), cosp = std::cos(phi);
        G4ParticleMomentum photonMomentum(sint * cosp, sint * sinp, cost);
        G4ThreeVector photonPolarization(cost * cosp, cost * sinp, -sint);
        G4ThreeVector perp = photonMomentum.cross(photonPolarization);
        phi = twopi * G4UniformRand();
        sinp = std::sin(phi); cosp = std::cos(phi);
        photonPolarization = (cosp * photonPolarization + sinp * perp).unit();
        auto scintPhoton = new G4DynamicParticle(G4OpticalPhoton::OpticalPhotonDefinition(), photonMomentum);
        scintPhoton->SetPolarization(photonPolarization);
        scintPhoton->SetKineticEnergy(sampledEnergy);
        G4double rand = G4UniformRand();
        if (aParticle->GetDefinition()->GetPDGCharge() == 0) rand = 1.0;
        G4double delta = rand * aStep.GetStepLength();
        G4double deltaTime = delta / (pPreStepPoint->GetVelocity() +
                                      rand * (pPostStepPoint->GetVelocity() - pPreStepPoint->GetVelocity()) / 2.);
        deltaTime -= scintTime * std::log(G4UniformRand());
        auto secTrack = new G4Track(scintPhoton, t0 + deltaTime, x0 + rand * aStep.GetDeltaPosition());
        secTrack->SetTouchableHandle(aStep.GetPreStepPoint()->GetTouchableHandle());
        secTrack->SetParentID(aTrack.GetTrackID());
        if (GetScintillationTrackInfo())
          secTrack->SetUserInformation(new G4ScintillationTrackInformation(scnt == 0 && N_timeconstants == 2 ? Fast : (N_timeconstants == 1 && Fast_Intensity && !Slow_Intensity ? Fast : Slow)));
        aParticleChange.AddSecondary(secTrack);
      }
    }
    return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
  }
};

class ISCalcScintillation : public G4RefScintillation {
 public:
  ISCalcScintillation() : G4RefScintillation() {}
  G4VParticleChange* PostStepDoIt(const G4Track& t, const G4Step& s) override {
    set(t, s);
    return G4RefScintillation::PostStepDoIt(t, s);
  }
  G4VParticleChange* AtRestDoIt(const G4Track& t, const G4Step& s) override {
    set(t, s);
    return G4RefScintillation::AtRestDoIt(t, s);
  }
 private:
  void set(const G4Track& t, const G4Step& s) {
    const G4Material* m = t.GetMaterial();
    if (!S.iscalcMat.count(m)) return;
    G4MaterialPropertiesTable* mpt = m->GetMaterialPropertiesTable();
    double E = s.GetTotalEnergyDeposit() / MeV;
    G4ThreeVector d = s.GetPreStepPoint()->GetPosition() - s.GetPostStepPoint()->GetPosition();
    double L = d.mag();
    int pdg = t.GetDefinition()->GetPDGEncoding();
    double n = E > 0.0 ? iscalc::photons(E, s.GetStepLength() / cm, fieldAt(s), L > 0.0 ? d.x() / L : 0.0, L > 0.0, pdg)
                       : 0.0;
    mpt->AddConstProperty("SCINTILLATIONYIELD", E > 0.0 ? n / E / MeV : 0.0);
    mpt->AddConstProperty("YIELDRATIO", iscalc::fastRatio(pdg));
  }
};

struct TrackHist { long evt = -1; int id = -1; float cy = NAN, cz = NAN; int ncross = 0, nrefl = 0, gas = 0, side = 0;
                   int nray = 0, nmie = 0; };
static TrackHist gH;
static int nreflPre = 0;   // gH.nrefl before this step's end-of-step reflection
static constexpr int kMaxPhotonSteps = 1000000;   // an optical photon's steps before it is ended (fate step_cap)

class Stepping : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    G4Track* tr = step->GetTrack();
    const G4StepPoint* pre = step->GetPreStepPoint();
    const G4StepPoint* post = step->GetPostStepPoint();
    if (tr->GetDefinition() != G4OpticalPhoton::Definition()) {   // charged tracks: recorded, never hits
      if (S.steps && tr->GetDefinition()->GetPDGCharge() != 0.0) {
        // n_cer / n_sci, not nc / ns: a local `ns` would shadow CLHEP's ns in the times below
        int n_cer = 0, n_sci = 0;
        if (auto sec = step->GetSecondaryInCurrentStep())
          for (auto t2 : *sec) {
            auto c = t2->GetCreatorProcess();
            if (!c) continue;
            if (c->GetProcessName() == "Cerenkov") n_cer++;
            else if (c->GetProcessName() == "Scintillation") n_sci++;
            if (S.opt.killOptical && t2->GetDefinition() == G4OpticalPhoton::Definition())
              const_cast<G4Track*>(t2)->SetTrackStatus(fStopAndKill);
          }
        G4ThreeVector a = pre->GetPosition(), b = post->GetPosition();
        StepRec r{(int32_t)S.curGlobalEvent, tr->GetDefinition()->GetPDGEncoding(),
                  (float)(tr->GetDefinition()->GetPDGCharge() / eplus),
                  (float)(a.x() / mm), (float)(a.y() / mm), (float)(a.z() / mm),
                  (float)(b.x() / mm), (float)(b.y() / mm), (float)(b.z() / mm),
                  (float)(pre->GetGlobalTime() / ns), (float)(post->GetGlobalTime() / ns),
                  (float)pre->GetBeta(), (float)post->GetBeta(), (float)(step->GetTotalEnergyDeposit() / MeV),
                  (float)(step->GetStepLength() / mm), n_cer, n_sci, (int32_t)pre->GetMaterial()->GetIndex(),
                  (float)fieldAt(*step)};
        std::fwrite(&r, sizeof(r), 1, S.steps); S.nSteps++;
      }
      return;
    }
    if (post->GetStepStatus() == fGeomBoundary && S.boundary) S.bstat[S.boundary->GetStatus()]++;
    if (std::getenv("G4REF_DEBUG_NORINDEX") && S.boundary && S.boundary->GetStatus() == NoRINDEX) {
      static int nprint = 0;
      if (nprint++ < 40) {
        const G4ThreeVector& q = post->GetPosition();
        std::cout << "[norindex] pre " << (pre->GetPhysicalVolume() ? pre->GetPhysicalVolume()->GetName() : "?")
                  << " (" << pre->GetMaterial()->GetName() << ") post "
                  << (post->GetPhysicalVolume() ? post->GetPhysicalVolume()->GetName() : "?") << " ("
                  << (post->GetMaterial() ? post->GetMaterial()->GetName() : "?") << ") at " << q / mm
                  << " step " << step->GetStepLength() / mm << " mm\n";
      }
    }
    if (gH.evt != (long)S.curGlobalEvent || gH.id != tr->GetTrackID()) {
      gH = TrackHist();
      gH.evt = (long)S.curGlobalEvent; gH.id = tr->GetTrackID();
    }
    {
      // a crossing of x = 0 is a change of the last NONZERO side (volumes meet at x = 0: a step can end and the
      // next start exactly there)
      G4ThreeVector a = pre->GetPosition(), b = post->GetPosition();
      if (gH.side == 0 && a.x() != 0.0) gH.side = a.x() > 0.0 ? 1 : -1;
      int sb = b.x() > 0.0 ? 1 : (b.x() < 0.0 ? -1 : 0);
      if (sb != 0 && gH.side != 0 && sb != gH.side) {
        if (gH.ncross == 0) {
          double f = (a.x() == b.x()) ? 0.0 : a.x() / (a.x() - b.x());
          gH.cy = (float)((a.y() + f * (b.y() - a.y())) / mm); gH.cz = (float)((a.z() + f * (b.z() - a.z())) / mm);
        }
        gH.ncross++;
      }
      if (sb != 0) gH.side = sb;
      nreflPre = gH.nrefl;
      if (post->GetStepStatus() == fGeomBoundary && S.boundary) {
        int st = S.boundary->GetStatus();
        if ((st >= FresnelReflection && st <= BackScattering) || (st >= PolishedLumirrorAirReflection && st != Dichroic))
          gH.nrefl++;
      }
      if (post->GetMaterial() && post->GetMaterial()->GetName() == "GAr") gH.gas = 1;
    }
    // Geant4 10.7: after a reflection off a volume whose material has no GROUPVEL (no RINDEX),
    // the zero-length step taken inside it gives the photon speed c, and OpBoundary
    // (StepTooSmall, then SameMaterial/reflections) keeps proposing that speed in the medium it
    // returns to. Reset the speed to the GROUPVEL of the medium the photon is now in.
    // (Same for a Fresnel reflection off a medium with a different GROUPVEL.) The medium the
    // photon continues in is the pre-step one after a reflection, the post-step one otherwise.
    // Geant4 10.7: a reflection off a dielectric_metal surface leaves the photon in the volume behind it; the step
    // back out is StepTooSmall only if <= kCarTolerance (1e-9 mm), and the intersection of a curved solid at a
    // grazing angle is good to ~1e-7 mm only, so a longer step out of the no-RINDEX volume is killed as NoRINDEX
    // (measured: 1.4 % of WCTE's photons, 1e-7 - 2e-4 mm inside a cathode). A sub-micron step out of a no-RINDEX
    // volume into a RINDEX one is that artefact: the photon lives on in the medium it re-enters.
    bool revived = false;
    if (S.opt.revive && post->GetStepStatus() == fGeomBoundary && S.boundary &&
        S.boundary->GetStatus() == NoRINDEX && step->GetStepLength() < 1e-3 * mm) {
      auto m1 = pre->GetMaterial()->GetMaterialPropertiesTable();
      auto m2 = post->GetMaterial() ? post->GetMaterial()->GetMaterialPropertiesTable() : nullptr;
      if (!(m1 && m1->GetProperty(kRINDEX)) && m2 && m2->GetProperty(kRINDEX)) {
        tr->SetTrackStatus(fAlive);
        S.fates["n_micro_revived"]++;
        revived = true;
      }
    }
    const G4Material* cont = revived ? post->GetMaterial() : nullptr;
    if (!revived && S.opt.fixVelocity && post->GetStepStatus() == fGeomBoundary && S.boundary &&
        tr->GetTrackStatus() == fAlive) {
      int st = S.boundary->GetStatus();
      if (st == FresnelRefraction || st == Transmission || st == SameMaterial || st == StepTooSmall)
        cont = post->GetMaterial();
      else if ((st >= FresnelReflection && st <= BackScattering) ||
               (st >= PolishedLumirrorAirReflection && st != Dichroic)) {
        // A reflection where Geant4 relocated the photon into a volume that holds the point inside it (an
        // overlapping sibling, at the exit of the one it was in): no zero-length step takes it back, it continues
        // in that volume, so in its medium. A point on the post-step volume's own surface: the usual return.
        const G4VTouchable* th = post->GetTouchable();
        G4ThreeVector local = th->GetHistory()->GetTopTransform().TransformPoint(post->GetPosition());
        cont = th->GetSolid()->Inside(local) == kInside ? post->GetMaterial() : pre->GetMaterial();
      }
    }
    if (cont) {
      auto mpt = cont->GetMaterialPropertiesTable();
      G4MaterialPropertyVector* gv = mpt ? mpt->GetProperty(kGROUPVEL) : nullptr;
      if (gv) {
        double v = gv->Value(post->GetKineticEnergy());
        if (std::abs(v - post->GetVelocity()) > 1e-9 * v) {
          const_cast<G4StepPoint*>(post)->SetVelocity(v);
          tr->SetVelocity(v);
          S.fates["n_velocity_resets"]++;
        }
      }
    }
    bool isWLS = tr->GetCreatorProcess() && tr->GetCreatorProcess()->GetProcessName() == "OpWLS";
    std::string who = isWLS ? "wls:" : (tr->GetParentID() == 0 ? "primary:" : "other:");
    const G4LogicalVolume* lv = pre->GetTouchableHandle()->GetVolume()->GetLogicalVolume();
    // a sensor reached through an optical surface with EFFICIENCY (OpBoundary status Detection: e.g. a
    // dielectric_metal photocathode with a complex index) is recorded at the boundary point, on the volume the
    // photon was absorbed into (the post-step one); a photon that ENTERS a sensitive volume at its first step there
    const G4StepPoint* at = nullptr;
    const G4StepPoint* idAt = nullptr;      // the point whose touchable names the sensor (default: at)
    if (S.opt.wcsimHits) {                  // WCSim: killed in the glass on a step whose post volume is the sensor
      if (post->GetStepStatus() == fGeomBoundary && tr->GetTrackStatus() == fStopAndKill &&
          post->GetTouchableHandle()->GetVolume() &&
          S.sensLV.count(post->GetTouchableHandle()->GetVolume()->GetLogicalVolume()))
        at = post;
      // and DETECTED leaving the sensor (the interior -> glass cathode): WCSimOpBoundaryProcess's Detection invokes the
      // glass's SD (InvokeSD, the post volume's), whose ProcessHits_boundary counts a step starting in InteriorWCPMT
      else if (post->GetStepStatus() == fGeomBoundary && S.boundary && S.boundary->GetStatus() == Detection &&
               S.sensLV.count(lv)) {
        at = post;
        idAt = pre;
      }
    } else if (post->GetStepStatus() == fGeomBoundary && S.boundary && S.boundary->GetStatus() == Detection &&
        post->GetTouchableHandle()->GetVolume() &&
        S.sensLV.count(post->GetTouchableHandle()->GetVolume()->GetLogicalVolume()))
      at = post;
    else if (S.sensLV.count(lv) && lv->GetMaterial()->GetMaterialPropertiesTable() &&
             lv->GetMaterial()->GetMaterialPropertiesTable()->GetProperty(kRINDEX))
      at = pre;                             // (a sensor without RINDEX is never entered: the zero-length step a
                                            // reflection off its surface takes inside it is not a hit)
    if (at) {
      int id = sensorFor((idAt ? idAt : at)->GetTouchableHandle());
      double E = at->GetKineticEnergy();
      auto& tot = S.totals[{S.curSource, id}];
      tot[0]++; tot[E >= S.opt.splitEV * eV ? 1 : 2]++; if (isWLS) tot[3]++;
      if (S.hits) {
        G4ThreeVector x = at->GetPosition(), d = at->GetMomentumDirection(), v = tr->GetVertexPosition();
        HitRec h{S.curSource, (int32_t)S.curGlobalEvent, id, isWLS ? 1 : 0,
                 (float)(x.x() / mm), (float)(x.y() / mm), (float)(x.z() / mm),
                 (float)(at->GetGlobalTime() / ns), (float)(E / eV),
                 (float)d.x(), (float)d.y(), (float)d.z(),
                 (float)(v.x() / mm), (float)(v.y() / mm), (float)(v.z() / mm), gH.cy, gH.cz,
                 gH.ncross, (S.opt.nreflCount == "before_hit" && at == pre) ? nreflPre : gH.nrefl, gH.gas, originOf(tr),
                 (int32_t)tr->GetTrackID(), gH.nray, gH.nmie};
        std::fwrite(&h, sizeof(h), 1, S.hits); S.nHits++;
      }
      S.fates[who + "detected"]++;
      tr->SetTrackStatus(fStopAndKill);
      for (auto sec : *step->GetSecondaryInCurrentStep())
        const_cast<G4Track*>(sec)->SetTrackStatus(fStopAndKill);
      return;
    }
    const G4VProcess* pr = post->GetProcessDefinedStep();
    std::string pn = pr ? pr->GetProcessName() : "none";
    if (pn == "OpRayleigh") { S.fates["n_rayleigh_steps"]++; gH.nray++; }
    else if (pn == "OpMieHG") gH.nmie++;
    // A photon held by total internal reflection in a medium that barely absorbs (ANNIE's veto-PMT glass, ABSLENGTH
    // 1e10 mm: 56 million steps, 141 km in 10 minutes) would stall its event: ended at a cap no photon that can reach
    // a sensor comes near, and counted.
    if (tr->GetTrackStatus() == fAlive && tr->GetCurrentStepNumber() >= kMaxPhotonSteps) {
      tr->SetTrackStatus(fStopAndKill);
      S.fates[who + "step_cap"]++;
      return;
    }
    if (tr->GetTrackStatus() == fStopAndKill) {
      std::string f;
      // OpBoundary is a forced process: a boundary step is "defined" by Transportation
      if (post->GetStepStatus() == fWorldBoundary) f = "escaped_world";
      else if (post->GetStepStatus() == fGeomBoundary && S.boundary)
        f = std::string("boundary_") + bStatus(S.boundary->GetStatus());
      else f = pn;
      S.fates[who + f]++;
    }
  }
};

// ---------------------------------------------------------------- physics
#ifdef G4REF_WCSIM_BOUNDARY
G4VProcess* g4refMakeWCSimBoundary();   // wcsim_boundary.cc (its own TU: WCSim's status enum clashes with Geant4's)
#endif
class OpticalOnly : public G4VModularPhysicsList {
 public:
  OpticalOnly() {
    SetVerboseLevel(0);
    if (!S.opt.gun.empty()) {                 // charged primaries: standard EM (and decays)
      RegisterPhysics(new G4EmStandardPhysics());
      RegisterPhysics(new G4DecayPhysics());
    }
    RegisterPhysics(new G4OpticalPhysics());
    if (!S.opt.stepLimit.empty()) RegisterPhysics(new G4StepLimiterPhysics());
  }
  void ConstructProcess() override {
    G4VModularPhysicsList::ConstructProcess();
    if (S.opt.wcsimBoundary) {
#ifdef G4REF_WCSIM_BOUNDARY
      G4ProcessManager* pm = G4OpticalPhoton::Definition()->GetProcessManager();
      pm->RemoveProcess(pm->GetProcess("OpBoundary"));
      pm->AddDiscreteProcess(g4refMakeWCSimBoundary());
      std::cout << "[g4ref] OpBoundary replaced by WCSimOpBoundaryProcess\n";
#else
      G4Exception("g4ref", "wcsim", FatalException, "--wcsim-boundary needs the g4ref_wcb build");
#endif
    }
    if (S.opt.cerenkovCount != "stock") {
      // G4OpticalPhysics' G4Cerenkov replaced by G4RefCerenkov (same name and ordering; settings from
      // G4OpticalParameters in PreparePhysicsTable, as G4Cerenkov)
      G4RefCerenkov* cer = nullptr;
      auto ic = GetParticleIterator();
      ic->reset();
      int nc = 0;
      while ((*ic)()) {
        G4ProcessManager* pm = ic->value()->GetProcessManager();
        G4VProcess* old = pm ? pm->GetProcess("Cerenkov") : nullptr;
        if (!old) continue;
        if (!cer) cer = new G4RefCerenkov(S.opt.cerenkovCount == "exact");
        bool active = pm->GetProcessActivation(old);
        pm->RemoveProcess(old);
        pm->AddProcess(cer);
        pm->SetProcessOrdering(cer, idxPostStep);
        pm->SetProcessActivation(cer, active);
        nc++;
      }
      std::cout << "[g4ref] Cerenkov photon number: " << S.opt.cerenkovCount << " (G4RefCerenkov on " << nc
                << " particles)\n";
    }
    if (S.opt.iscalc.empty() && S.opt.scintSplit == "stock") return;
    // G4OpticalPhysics' G4Scintillation replaced by the ISCalc one or G4RefScintillation (same name, settings and
    // ordering)
    G4RefScintillation* mine = nullptr;
    auto it = GetParticleIterator();
    it->reset();
    int n = 0;
    while ((*it)()) {
      G4ProcessManager* pm = it->value()->GetProcessManager();
      G4VProcess* old = pm ? pm->GetProcess("Scintillation") : nullptr;
      if (!old) continue;
      if (!mine) {
        mine = S.opt.iscalc.empty() ? new G4RefScintillation() : new ISCalcScintillation();
        mine->AddSaturation(static_cast<G4Scintillation*>(old)->GetSaturation());
      }
      bool active = pm->GetProcessActivation(old);
      pm->RemoveProcess(old);
      pm->AddProcess(mine);
      pm->SetProcessOrderingToLast(mine, idxAtRest);
      pm->SetProcessOrderingToLast(mine, idxPostStep);
      pm->SetProcessActivation(mine, active);
      n++;
    }
    if (S.opt.scintSplit == "stock") std::cout << "[g4ref] ISCalcCorrelated scintillation process on " << n << " particles\n";
    else std::cout << "[g4ref] " << (S.opt.iscalc.empty() ? "G4RefScintillation" : "ISCalcCorrelated scintillation")
                   << " process on " << n << " particles; fast/slow split " << S.opt.scintSplit << "\n";
  }
  void ConstructParticle() override {
    // G4OpticalPhysics only defines the optical photon; its (inactive) scintillation set-up
    // needs e-/proton to exist, so define the standard particle zoo (no processes for them).
    G4BosonConstructor().ConstructParticle();
    G4LeptonConstructor().ConstructParticle();
    G4MesonConstructor().ConstructParticle();
    G4BaryonConstructor().ConstructParticle();
    G4IonConstructor().ConstructParticle();
    G4VModularPhysicsList::ConstructParticle();
  }
  void SetCuts() override { SetCutsWithDefault(); }
};

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
  S.opt = parse(argc, argv);
  if (S.opt.dump.empty() && S.opt.gun.empty() && S.opt.probe.empty()) S.cfg = readConfig(S.opt.config);
  if (!S.cfg.sources.empty()) parseProfile();
  for (auto& s : S.cfg.sources) {
    S.evFirst.push_back(S.nGlobalEvents);
    S.nGlobalEvents += (s.n + S.opt.batch - 1) / S.opt.batch;
  }
  S.launched.assign(S.cfg.sources.size(), 0);
  if (!S.opt.gun.empty()) { S.nGlobalEvents = S.opt.events; S.launched.assign(1, 0); }
  long nLocal = 0;
  for (long g = S.opt.chunk; g < S.nGlobalEvents; g += S.opt.nchunks) nLocal++;

  G4Random::setTheEngine(new CLHEP::MixMaxRng());
  auto rm = new G4RunManager();
  rm->SetUserInitialization(new Detector());

  auto phys = new OpticalOnly();
#if G4VERSION_NUMBER >= 1070
  auto op = G4OpticalParameters::Instance();
  op->SetProcessActivation("Cerenkov", S.opt.cherenkov);
  op->SetProcessActivation("Scintillation", S.opt.scint);
  op->SetCerenkovMaxPhotonsPerStep(S.opt.maxCerenkovPerStep);
  op->SetProcessActivation("OpAbsorption", S.opt.absorption);
  op->SetProcessActivation("OpRayleigh", S.opt.rayleigh);
  op->SetProcessActivation("OpMieHG", S.opt.mie);
  op->SetProcessActivation("OpWLS", S.opt.wls);
  op->SetWLSTimeProfile(S.opt.wlsProfile);
#endif
  rm->SetUserInitialization(phys);
  rm->SetUserAction(new Primaries());
  rm->SetUserAction(new Stepping());
  rm->SetUserAction(new Tracking());
  G4UImanager::GetUIpointer()->ApplyCommand("/run/verbose 0");
  G4UImanager::GetUIpointer()->ApplyCommand("/event/verbose 0");
  G4UImanager::GetUIpointer()->ApplyCommand(            // diagnostic: G4REF_TRACKING_VERBOSE=N
      std::string("/tracking/verbose ") + (std::getenv("G4REF_TRACKING_VERBOSE") ? std::getenv("G4REF_TRACKING_VERBOSE") : "0"));
  rm->Initialize();
  G4UImanager::GetUIpointer()->ApplyCommand(
      "/run/printProgress " + std::to_string(S.opt.printEvery));
  if (!S.opt.probe.empty()) {                   // the geometry as Geant4 sees it, point by point
    G4Navigator* nav = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
    std::ifstream in(S.opt.probe);
    std::ofstream f(S.opt.out + "_probe.csv");
    f << "i,material,pv,copy,depth\n";
    double x, y, z;
    long i = 0;
    while (in >> x >> y >> z) {
      G4VPhysicalVolume* pv = nav->LocateGlobalPointAndSetup(G4ThreeVector(x * mm, y * mm, z * mm), nullptr, false, true);
      if (!pv) { f << i++ << ",OUTSIDE,,,-1\n"; continue; }
      G4TouchableHistory* th = nav->CreateTouchableHistory();
      f << i++ << "," << pv->GetLogicalVolume()->GetMaterial()->GetName() << "," << pv->GetName() << ","
        << pv->GetCopyNo() << "," << th->GetHistoryDepth() << "\n";
      delete th;
    }
    std::cout << "[g4ref] probed " << i << " points -> " << S.opt.out << "_probe.csv" << std::endl;
    return 0;
  }
  if (!S.opt.emTables.empty()) {                // Geant4's own energy-loss tables (the physics tables need a run)
    if (S.opt.gun.empty()) { std::cerr << "--em-tables needs --gun (EM physics)\n"; return 2; }
    rm->BeamOn(0);
    const G4Material* mat = G4Material::GetMaterial(S.opt.emTables, false);
    if (!mat) { std::cerr << "--em-tables: no material " << S.opt.emTables << "\n"; return 2; }
    G4EmCalculator calc;
    std::ofstream f(S.opt.out + "_emtables.csv");
    f << "particle,T_MeV,dedx_restricted_MeV_cm,dedx_electronic_MeV_cm,dedx_total_MeV_cm,range_restricted_cm,"
         "delta_xs_per_cm,ecut_MeV\n";
    const char* parts[] = {"mu-", "mu+", "e-", "e+", "proton"};
    for (auto pn : parts) {
      const G4ParticleDefinition* pd = G4ParticleTable::GetParticleTable()->FindParticle(pn);
      std::string ioni = (pd->GetPDGMass() < 1 * MeV) ? "eIoni" : (std::string(pn).rfind("mu", 0) == 0 ? "muIoni" : "hIoni");
      double ecut = -1.0, rcut = -1.0;          // the run's own production cut for e- in MAT (its couple)
      auto pct = G4ProductionCutsTable::GetProductionCutsTable();
      for (size_t c = 0; c < pct->GetTableSize(); ++c) {
        auto cp = pct->GetMaterialCutsCouple((G4int)c);
        if (cp->GetMaterial() == mat) {
          ecut = (*pct->GetEnergyCutsVector(idxG4ElectronCut))[c];
          rcut = cp->GetProductionCuts()->GetProductionCut("e-");
          break;
        }
      }
      if (ecut < 0) { std::cerr << "--em-tables: no couple for " << S.opt.emTables << "\n"; std::exit(2); }
      if (std::string(pn) == "mu-")
        std::cout << "[g4ref] production cut for e- in " << S.opt.emTables << ": " << rcut / mm << " mm = " << ecut / MeV
                  << " MeV\n";
      for (int i = 0; i <= 800; ++i) {
        double T = std::pow(10.0, -3.0 + 8.0 * i / 800.0) * MeV;
        f << pn << "," << T / MeV << "," << calc.GetDEDX(T, pd, mat) / (MeV / cm) << ","
          << calc.ComputeElectronicDEDX(T, pd, mat) / (MeV / cm) << "," << calc.ComputeTotalDEDX(T, pd, mat) / (MeV / cm)
          << "," << calc.GetRangeFromRestricteDEDX(T, pd, mat) / cm << ","
          << calc.ComputeCrossSectionPerVolume(T, pd, ioni, mat, ecut) * cm << "," << ecut / MeV << "\n";
      }
    }
    // the delta-ray spectrum as Geant4 samples it: the ionisation cross-section above a cut, on a grid of cuts
    // (its derivative in the cut is d sigma / dT of the delta's kinetic energy)
    std::ofstream fx(S.opt.out + "_deltaxs.csv");
    fx << "particle,T_MeV,cut_MeV,xs_per_cm\n";
    for (auto pn : {"mu-", "mu+"}) {
      const G4ParticleDefinition* pd = G4ParticleTable::GetParticleTable()->FindParticle(pn);
      for (int i = 0; i <= 50; ++i) {
        double T = std::pow(10.0, 0.0 + 5.0 * i / 50.0) * MeV;          // 1 MeV - 100 GeV
        for (int j = 0; j <= 160; ++j) {
          double cut = std::pow(10.0, -1.0 + 6.0 * j / 160.0) * MeV;     // 0.1 MeV - 100 GeV
          fx << pn << "," << T / MeV << "," << cut / MeV << ","
             << calc.ComputeCrossSectionPerVolume(T, pd, "muIoni", mat, cut) * cm << "\n";
        }
      }
    }
    { std::ofstream fm(S.opt.out + "_material.csv");   // the material constants the track model needs
      fm << "material,density_g_cm3,I_eV,X0_cm\n" << S.opt.emTables << "," << mat->GetDensity() / (g / cm3) << ","
         << mat->GetIonisation()->GetMeanExcitationEnergy() / eV << "," << mat->GetRadlen() / cm << "\n"; }
    std::cout << "[g4ref] em tables for " << S.opt.emTables << " (density " << mat->GetDensity() / (g / cm3)
              << " g/cm3, I " << mat->GetIonisation()->GetMeanExcitationEnergy() / eV << " eV) -> "
              << S.opt.out << "_emtables.csv\n";
    delete rm;
    return 0;
  }

  // find the boundary process to read its status
  auto pm = G4OpticalPhoton::Definition()->GetProcessManager();
  auto pl = pm->GetProcessList();
  std::cout << "[g4ref] optical-photon processes:";
  for (size_t i = 0; i < pl->size(); ++i) {
    std::cout << " " << (*pl)[i]->GetProcessName()
              << (pm->GetProcessActivation((*pl)[i]) ? "" : "(off)");
    if ((*pl)[i]->GetProcessName() == "OpBoundary")
      S.boundary = dynamic_cast<G4OpBoundaryProcess*>((*pl)[i]);
  }
  std::cout << "\n";

  if (!S.opt.gun.empty()) {
    S.steps = std::fopen((S.opt.out + "_steps.npy").c_str(), "wb+");
    writeStepsHeader(S.steps, 0);
    std::ofstream f(S.opt.out + "_materials.csv");
    f << "index,name\n";
    for (auto m : *G4Material::GetMaterialTable()) f << m->GetIndex() << "," << m->GetName() << "\n";
    std::cout << "[g4ref] gun " << S.opt.gun << " " << S.opt.gunMeV << " MeV at " << S.opt.gunPos / mm
              << " mm along " << S.opt.gunDir << "; Cerenkov " << S.opt.cherenkov << " Scintillation "
              << S.opt.scint << " (max " << S.opt.maxCerenkovPerStep << " Cerenkov photons / step)\n";
  }
  if (S.opt.dumpPhotons) {
    S.photons = std::fopen((S.opt.out + "_photons.npy").c_str(), "wb+");
    writePhotonsHeader(S.photons, 0);
  }
  if (S.opt.writeHits) {
    S.hits = std::fopen((S.opt.out + "_hits.npy").c_str(), "wb+");
    writeNpyHeader(S.hits, 0);
  }
  std::cout << "[g4ref] Geant4 " << G4Version << "\n[g4ref] sources " << S.cfg.sources.size()
            << ", global events " << S.nGlobalEvents << ", this chunk " << S.opt.chunk << "/"
            << S.opt.nchunks << " runs " << nLocal << " events of <= " << S.opt.batch << " photons\n";
  auto t0 = std::chrono::steady_clock::now();
  rm->BeamOn((int)nLocal);
  double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  if (S.hits) { writeNpyHeader(S.hits, S.nHits); std::fclose(S.hits); }
  if (S.steps) { writeStepsHeader(S.steps, S.nSteps); std::fclose(S.steps); }
  if (S.photons) { writePhotonsHeader(S.photons, S.nPhotons); std::fclose(S.photons); }

  { std::ofstream f(S.opt.out + "_sensors.csv");
    f << "sensor,path,lv,cx_mm,cy_mm,cz_mm\n";
    for (size_t i = 0; i < S.sensors.size(); ++i) {
      auto& s = S.sensors[i];
      f << i << "," << s.path << "," << s.lv << "," << s.centre.x() / mm << ","
        << s.centre.y() / mm << "," << s.centre.z() / mm << "\n";
    } }
  { std::ofstream f(S.opt.out + "_totals.csv");
    f << "src,sensor,n,n_ge_split,n_lt_split,n_wls\n";
    for (auto& kv : S.totals)
      f << kv.first.first << "," << kv.first.second << "," << kv.second[0] << ","
        << kv.second[1] << "," << kv.second[2] << "," << kv.second[3] << "\n"; }
  { std::ofstream f(S.opt.out + "_summary.txt");
    f << "geant4 " << G4Version << "\ngdml " << S.opt.gdml << "\nconfig " << S.opt.config
      << "\nsens_regex " << S.opt.sensRegex << "\nsens_aux " << S.opt.sensAux << "\nwcsim_hits " << S.opt.wcsimHits << "\nwcsim_boundary " << S.opt.wcsimBoundary << "\nseed "
      << S.opt.seed << "\nbatch " << S.opt.batch << "\nchunk " << S.opt.chunk << "\nnchunks "
      << S.opt.nchunks << "\nsplit_ev " << S.opt.splitEV << "\nwall_s " << dt << "\nhits "
      << S.nHits << "\n";
    f << "physics_mode " << physicsMode() << "\ncerenkov_count " << S.opt.cerenkovCount << "\nscint_split "
      << S.opt.scintSplit << "\nnrefl_count " << S.opt.nreflCount << "\n";
    for (auto& w : gRindexWarn) f << "rindex_warning " << w << "\n";
    if (gProf.kind) f << "profile " << gProf.desc << "\n";
    if (S.photons) f << "photons_dumped " << S.nPhotons << "\n";
    for (size_t i = 0; i < S.cfg.sources.size(); ++i)
      f << "source " << i << " " << S.cfg.sources[i].desc << " launched " << S.launched[i] << "\n";
    for (auto& kv : S.fates) f << "fate " << kv.first << " " << kv.second << "\n";
    for (auto& kv : S.bstat) f << "boundary_status " << bStatus(kv.first) << " " << kv.second << "\n"; }
  std::cout << "[g4ref] done in " << dt << " s, hits " << S.nHits << "\n";
  for (auto& kv : S.fates) std::cout << "[g4ref] fate " << kv.first << " " << kv.second << "\n";
  for (auto& kv : S.bstat)
    std::cout << "[g4ref] boundary_status " << bStatus(kv.first) << " " << kv.second << "\n";
  delete rm;
  return 0;
}
