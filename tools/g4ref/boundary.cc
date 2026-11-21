// boundary: what G4OpBoundaryProcess does to optical photons hitting one flat interface (z = 0) between box A
// (z > 0, RINDEX n1) and box B (z < 0, RINDEX n2), with a G4LogicalBorderSurface from A to B. Every photon starts
// in A along one direction; the step that reaches the interface is recorded after the boundary process and the
// photon killed. Writes one row per photon to a .npy file (dx, dy, dz, px, py, pz, status, alive) and the settings
// to a .json sidecar. Built and run against Geant4 11.3.0 (build_boundary.sh; the status names are 11.x's).

#include "G4Box.hh"
#include "G4Electron.hh"
#include "G4Event.hh"
#include "G4Gamma.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4NistManager.hh"
#include "G4OpBoundaryProcess.hh"
#include "G4OpticalParameters.hh"
#include "G4OpticalPhoton.hh"
#include "G4OpticalPhysics.hh"
#include "G4OpticalSurface.hh"
#include "G4PVPlacement.hh"
#include "G4Positron.hh"
#include "G4PhysicalConstants.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include "G4ProcessManager.hh"
#include "G4Proton.hh"
#include "G4RunManager.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4UserSteppingAction.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4Version.hh"
#include "Randomize.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "the .npy writer assumes a little-endian host");

namespace {

constexpr int kColumns = 8;

// Names of G4OpBoundaryProcessStatus values in enum order (Geant4 11.3.0 G4OpBoundaryProcess.hh).
const char* const kStatusNames[] = {
  "Undefined", "Transmission", "FresnelRefraction", "FresnelReflection",
  "TotalInternalReflection", "LambertianReflection", "LobeReflection", "SpikeReflection",
  "BackScattering", "Absorption", "Detection", "NotAtBoundary", "SameMaterial", "StepTooSmall",
  "NoRINDEX", "PolishedLumirrorAirReflection", "PolishedLumirrorGlueReflection",
  "PolishedAirReflection", "PolishedTeflonAirReflection", "PolishedTiOAirReflection",
  "PolishedTyvekAirReflection", "PolishedVM2000AirReflection", "PolishedVM2000GlueReflection",
  "EtchedLumirrorAirReflection", "EtchedLumirrorGlueReflection", "EtchedAirReflection",
  "EtchedTeflonAirReflection", "EtchedTiOAirReflection", "EtchedTyvekAirReflection",
  "EtchedVM2000AirReflection", "EtchedVM2000GlueReflection", "GroundLumirrorAirReflection",
  "GroundLumirrorGlueReflection", "GroundAirReflection", "GroundTeflonAirReflection",
  "GroundTiOAirReflection", "GroundTyvekAirReflection", "GroundVM2000AirReflection",
  "GroundVM2000GlueReflection", "Dichroic", "CoatedDielectricReflection",
  "CoatedDielectricRefraction", "CoatedDielectricFrustratedTransmission"};
static_assert(sizeof(kStatusNames) / sizeof(kStatusNames[0]) == CoatedDielectricFrustratedTransmission + 1,
              "kStatusNames does not match G4OpBoundaryProcessStatus");

struct Options
{
  double n1 = 1.0;
  double n2 = 1.5;
  std::string model = "glisur";
  std::string finish = "polished";
  std::string type = "dielectric_dielectric";
  std::optional<double> value;
  double thetaDeg = 30.0;
  long n = 1000000;
  long seed = 1;
  std::string pol = "random";
  double energyEv = 3.0;
  std::string out;
  int perEvent = 1000;
  std::optional<double> R, T, spike, lobe, back;
};

[[noreturn]] void Fail(const std::string& msg)
{
  std::cerr << "boundary: " << msg << "\n";
  std::exit(2);
}

void Usage()
{
  std::cerr
    << "usage: boundary --out FILE.npy [--n1 1.0] [--n2 1.5] [--model glisur|unified]\n"
       "         [--finish polished|ground] [--type dielectric_dielectric|dielectric_metal]\n"
       "         [--value X] [--theta DEG] [--n N] [--seed S] [--pol random|s|p] [--energy-ev E]\n"
       "         [--per-event K] [--R x] [--T x] [--spike x] [--lobe x] [--back x]\n";
}

double ToDouble(const std::string& key, const std::string& s)
{
  char* end = nullptr;
  double v = std::strtod(s.c_str(), &end);
  if (end == s.c_str() || *end != '\0') Fail("bad number for " + key + ": " + s);
  return v;
}

long ToLong(const std::string& key, const std::string& s)
{
  char* end = nullptr;
  long v = std::strtol(s.c_str(), &end, 10);
  if (end == s.c_str() || *end != '\0') Fail("bad integer for " + key + ": " + s);
  return v;
}

Options Parse(int argc, char** argv)
{
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "-h" || key == "--help") {
      Usage();
      std::exit(0);
    }
    if (i + 1 >= argc) Fail("missing value for " + key);
    std::string v = argv[++i];
    if (key == "--n1") o.n1 = ToDouble(key, v);
    else if (key == "--n2") o.n2 = ToDouble(key, v);
    else if (key == "--model") o.model = v;
    else if (key == "--finish") o.finish = v;
    else if (key == "--type") o.type = v;
    else if (key == "--value") o.value = ToDouble(key, v);
    else if (key == "--theta") o.thetaDeg = ToDouble(key, v);
    else if (key == "--n") o.n = ToLong(key, v);
    else if (key == "--seed") o.seed = ToLong(key, v);
    else if (key == "--pol") o.pol = v;
    else if (key == "--energy-ev") o.energyEv = ToDouble(key, v);
    else if (key == "--out") o.out = v;
    else if (key == "--per-event") o.perEvent = static_cast<int>(ToLong(key, v));
    else if (key == "--R") o.R = ToDouble(key, v);
    else if (key == "--T") o.T = ToDouble(key, v);
    else if (key == "--spike") o.spike = ToDouble(key, v);
    else if (key == "--lobe") o.lobe = ToDouble(key, v);
    else if (key == "--back") o.back = ToDouble(key, v);
    else {
      Usage();
      Fail("unknown option " + key);
    }
  }
  if (o.out.empty()) {
    Usage();
    Fail("--out is required");
  }
  if (o.model != "glisur" && o.model != "unified") Fail("--model must be glisur or unified");
  if (o.finish != "polished" && o.finish != "ground") Fail("--finish must be polished or ground");
  if (o.type != "dielectric_dielectric" && o.type != "dielectric_metal")
    Fail("--type must be dielectric_dielectric or dielectric_metal");
  if (o.pol != "random" && o.pol != "s" && o.pol != "p") Fail("--pol must be random, s or p");
  if (o.n <= 0 || o.perEvent <= 0) Fail("--n and --per-event must be positive");
  if (!(o.thetaDeg >= 0.0 && o.thetaDeg < 90.0)) Fail("--theta must be in [0, 90)");
  if (!(o.energyEv > 0.0)) Fail("--energy-ev must be positive");
  return o;
}

std::string JsonString(const std::string& s)
{
  std::string r = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') r += '\\';
    r += c;
  }
  return r + "\"";
}

std::string JsonNumber(double v)
{
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  return buf;
}

std::string JsonOptional(const std::optional<double>& v)
{
  return v ? JsonNumber(*v) : "null";
}

void WriteNpy(const std::string& path, const std::vector<double>& data, std::size_t rows)
{
  std::string header = "{'descr': '<f8', 'fortran_order': False, 'shape': (" + std::to_string(rows)
                       + ", " + std::to_string(kColumns) + "), }";
  const std::size_t prefix = 10;
  header += std::string((64 - (prefix + header.size() + 1) % 64) % 64, ' ');
  header += '\n';
  if (header.size() > 0xffff) Fail("npy header too long");
  std::ofstream f(path, std::ios::binary);
  if (!f) Fail("cannot open " + path);
  const auto len = static_cast<std::uint16_t>(header.size());
  f.write("\x93NUMPY\x01\x00", 8);
  f.put(static_cast<char>(len & 0xff));
  f.put(static_cast<char>(len >> 8));
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  f.write(reinterpret_cast<const char*>(data.data()),
          static_cast<std::streamsize>(rows * kColumns * sizeof(double)));
  if (!f) Fail("error writing " + path);
}

G4OpticalSurfaceModel ModelOf(const std::string& s) { return s == "unified" ? unified : glisur; }
G4OpticalSurfaceFinish FinishOf(const std::string& s) { return s == "ground" ? ground : polished; }
G4SurfaceType TypeOf(const std::string& s)
{
  return s == "dielectric_metal" ? dielectric_metal : dielectric_dielectric;
}

class Detector : public G4VUserDetectorConstruction
{
 public:
  explicit Detector(const Options& o) : fOpt(o) {}

  G4VPhysicalVolume* Construct() override
  {
    auto* vacuum = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");
    const double e = fOpt.energyEv * eV;
    fEnergies = {0.5 * e, 2.0 * e};

    auto* matA = new G4Material("boundary_A", vacuum->GetDensity(), vacuum, kStateGas);
    auto* matB = new G4Material("boundary_B", vacuum->GetDensity(), vacuum, kStateGas);
    matA->SetMaterialPropertiesTable(RindexTable(fOpt.n1));
    matB->SetMaterialPropertiesTable(RindexTable(fOpt.n2));

    auto* worldLV = new G4LogicalVolume(new G4Box("World", 2.5 * m, 2.5 * m, 1.5 * m), vacuum, "World");
    auto* worldPV = new G4PVPlacement(nullptr, {}, worldLV, "World", nullptr, false, 0, true);
    auto* lvA = new G4LogicalVolume(new G4Box("A", 2 * m, 2 * m, 0.5 * m), matA, "A");
    auto* lvB = new G4LogicalVolume(new G4Box("B", 2 * m, 2 * m, 0.5 * m), matB, "B");
    fPvA = new G4PVPlacement(nullptr, {0, 0, 0.5 * m}, lvA, "A", worldLV, false, 0, true);
    fPvB = new G4PVPlacement(nullptr, {0, 0, -0.5 * m}, lvB, "B", worldLV, false, 0, true);

    const auto model = ModelOf(fOpt.model);
    const auto finish = FinishOf(fOpt.finish);
    const auto type = TypeOf(fOpt.type);
    fSurface = fOpt.value ? new G4OpticalSurface("AB", model, finish, type, *fOpt.value)
                          : new G4OpticalSurface("AB", model, finish, type);
    G4MaterialPropertiesTable* smpt = nullptr;
    const std::pair<const char*, const std::optional<double>*> props[] = {
      {"REFLECTIVITY", &fOpt.R},
      {"TRANSMITTANCE", &fOpt.T},
      {"SPECULARLOBECONSTANT", &fOpt.lobe},
      {"SPECULARSPIKECONSTANT", &fOpt.spike},
      {"BACKSCATTERCONSTANT", &fOpt.back}};
    for (const auto& [name, v] : props) {
      if (!*v) continue;
      if (smpt == nullptr) smpt = new G4MaterialPropertiesTable();
      smpt->AddProperty(name, fEnergies, {**v, **v});
    }
    if (smpt != nullptr) fSurface->SetMaterialPropertiesTable(smpt);
    new G4LogicalBorderSurface("AB", fPvA, fPvB, fSurface);
    return worldPV;
  }

  const G4VPhysicalVolume* PvA() const { return fPvA; }
  const G4VPhysicalVolume* PvB() const { return fPvB; }
  const G4OpticalSurface* Surface() const { return fSurface; }
  const std::vector<G4double>& Energies() const { return fEnergies; }

 private:
  G4MaterialPropertiesTable* RindexTable(double n) const
  {
    auto* mpt = new G4MaterialPropertiesTable();
    mpt->AddProperty("RINDEX", fEnergies, {n, n});
    return mpt;
  }

  const Options& fOpt;
  std::vector<G4double> fEnergies;
  G4VPhysicalVolume* fPvA = nullptr;
  G4VPhysicalVolume* fPvB = nullptr;
  G4OpticalSurface* fSurface = nullptr;
};

class PhysicsList : public G4VModularPhysicsList
{
 public:
  PhysicsList()
  {
    SetVerboseLevel(0);
    auto* params = G4OpticalParameters::Instance();
    for (const char* p : {"OpAbsorption", "OpRayleigh", "OpMieHG", "OpWLS", "OpWLS2", "Cerenkov", "Scintillation"})
      params->SetProcessActivation(p, false);
    RegisterPhysics(new G4OpticalPhysics(0));
    params->SetBoundaryVerboseLevel(1);
  }

  // G4LossTableManager (used by Geant4 11.3.0 G4OpticalPhysics::ConstructProcess) needs e-
  // defined in PreInit, and G4ProductionCutsTable looks up gamma, e-, e+ and proton.
  void ConstructParticle() override
  {
    G4VModularPhysicsList::ConstructParticle();
    G4Gamma::Definition();
    G4Electron::Definition();
    G4Positron::Definition();
    G4Proton::Definition();
  }
};

class Primaries : public G4VUserPrimaryGeneratorAction
{
 public:
  explicit Primaries(const Options& o) : fOpt(o)
  {
    const double th = o.thetaDeg * deg;
    fDir = G4ThreeVector(std::sin(th), 0.0, -std::cos(th));
    fS = G4ThreeVector(0.0, 1.0, 0.0);
    fP = fDir.cross(fS);
    fStart = -0.5 * m * fDir;
  }

  void GeneratePrimaries(G4Event* event) override
  {
    const long first = static_cast<long>(event->GetEventID()) * fOpt.perEvent;
    const long count = std::min<long>(fOpt.perEvent, fOpt.n - first);
    const G4ThreeVector momentum = fOpt.energyEv * eV * fDir;
    auto* vertex = new G4PrimaryVertex(fStart, 0.0);
    for (long k = 0; k < count; ++k) {
      auto* p = new G4PrimaryParticle(G4OpticalPhoton::Definition());
      p->SetMomentum(momentum.x(), momentum.y(), momentum.z());
      p->SetPolarization(Polarisation());
      vertex->SetPrimary(p);
    }
    event->AddPrimaryVertex(vertex);
  }

 private:
  G4ThreeVector Polarisation() const
  {
    if (fOpt.pol == "s") return fS;
    if (fOpt.pol == "p") return fP;
    const double phi = twopi * G4UniformRand();
    return std::cos(phi) * fP + std::sin(phi) * fS;
  }

  const Options& fOpt;
  G4ThreeVector fDir, fS, fP, fStart;
};

class Stepping : public G4UserSteppingAction
{
 public:
  Stepping(const Detector& det, std::vector<double>& rows) : fDet(det), fRows(rows) {}

  void UserSteppingAction(const G4Step* step) override
  {
    G4Track* track = step->GetTrack();
    const G4StepPoint* pre = step->GetPreStepPoint();
    const G4StepPoint* post = step->GetPostStepPoint();
    if (post->GetStepStatus() == fGeomBoundary && pre->GetPhysicalVolume() == fDet.PvA()
        && post->GetPhysicalVolume() == fDet.PvB())
    {
      if (fBoundary == nullptr) FindBoundaryProcess(track, pre, post);
      const G4ThreeVector& d = post->GetMomentumDirection();
      const G4ThreeVector& p = post->GetPolarization();
      fRows.insert(fRows.end(), {d.x(), d.y(), d.z(), p.x(), p.y(), p.z(),
                                 static_cast<double>(fBoundary->GetStatus()),
                                 track->GetTrackStatus() == fAlive ? 1.0 : 0.0});
    }
    else {
      ++fUnexpected;
    }
    track->SetTrackStatus(fStopAndKill);
  }

  long Unexpected() const { return fUnexpected; }
  bool SurfaceFoundInStep() const { return fSurfaceFoundInStep; }

 private:
  // Geant4 11.3.0 G4OpBoundaryProcess::PostStepDoIt looks the surface up with
  // G4LogicalBorderSurface::GetSurface(thePrePV, thePostPV); this repeats that lookup.
  void FindBoundaryProcess(const G4Track* track, const G4StepPoint* pre, const G4StepPoint* post)
  {
    const G4ProcessVector* pv =
      track->GetDefinition()->GetProcessManager()->GetPostStepProcessVector(typeDoIt);
    for (std::size_t i = 0; i < pv->entries(); ++i)
      if (auto* b = dynamic_cast<G4OpBoundaryProcess*>((*pv)[i])) fBoundary = b;
    if (fBoundary == nullptr) Fail("no G4OpBoundaryProcess attached to the optical photon");
    const G4LogicalBorderSurface* s =
      G4LogicalBorderSurface::GetSurface(pre->GetPhysicalVolume(), post->GetPhysicalVolume());
    fSurfaceFoundInStep = s != nullptr && s->GetSurfaceProperty() == fDet.Surface();
  }

  const Detector& fDet;
  std::vector<double>& fRows;
  const G4OpBoundaryProcess* fBoundary = nullptr;
  long fUnexpected = 0;
  bool fSurfaceFoundInStep = false;
};

std::vector<std::string> OpticalPhotonProcesses()
{
  std::vector<std::string> names;
  const G4ProcessVector* pv = G4OpticalPhoton::Definition()->GetProcessManager()->GetProcessList();
  for (std::size_t i = 0; i < pv->entries(); ++i) names.push_back((*pv)[i]->GetProcessName());
  return names;
}

void WriteJson(const std::string& path, const Options& o, const Detector& det, bool surfaceFound,
               bool surfaceFoundInStep, const std::vector<std::string>& processes, std::size_t rows,
               long unexpected)
{
  std::ofstream f(path);
  if (!f) Fail("cannot open " + path);
  const G4OpticalSurface* s = det.Surface();
  const auto& e = det.Energies();
  std::ostringstream procs, names;
  for (std::size_t i = 0; i < processes.size(); ++i) procs << (i ? ", " : "") << JsonString(processes[i]);
  for (std::size_t i = 0; i < std::size(kStatusNames); ++i) names << (i ? ", " : "") << JsonString(kStatusNames[i]);
  f << "{\n"
    << "  \"geant4_version\": " << JsonString(G4Version) << ",\n"
    << "  \"geant4_date\": " << JsonString(G4Date) << ",\n"
    << "  \"geant4_version_number\": " << G4VERSION_NUMBER << ",\n"
    << "  \"options\": {\n"
    << "    \"n1\": " << JsonNumber(o.n1) << ",\n"
    << "    \"n2\": " << JsonNumber(o.n2) << ",\n"
    << "    \"model\": " << JsonString(o.model) << ",\n"
    << "    \"finish\": " << JsonString(o.finish) << ",\n"
    << "    \"type\": " << JsonString(o.type) << ",\n"
    << "    \"value\": " << JsonOptional(o.value) << ",\n"
    << "    \"theta_deg\": " << JsonNumber(o.thetaDeg) << ",\n"
    << "    \"n\": " << o.n << ",\n"
    << "    \"seed\": " << o.seed << ",\n"
    << "    \"pol\": " << JsonString(o.pol) << ",\n"
    << "    \"energy_ev\": " << JsonNumber(o.energyEv) << ",\n"
    << "    \"out\": " << JsonString(o.out) << ",\n"
    << "    \"per_event\": " << o.perEvent << ",\n"
    << "    \"R\": " << JsonOptional(o.R) << ",\n"
    << "    \"T\": " << JsonOptional(o.T) << ",\n"
    << "    \"spike\": " << JsonOptional(o.spike) << ",\n"
    << "    \"lobe\": " << JsonOptional(o.lobe) << ",\n"
    << "    \"back\": " << JsonOptional(o.back) << "\n"
    << "  },\n"
    << "  \"surface\": {\n"
    << "    \"model\": " << static_cast<int>(s->GetModel()) << ",\n"
    << "    \"finish\": " << static_cast<int>(s->GetFinish()) << ",\n"
    << "    \"type\": " << static_cast<int>(s->GetType()) << ",\n"
    << "    \"polish\": " << JsonNumber(s->GetPolish()) << ",\n"
    << "    \"sigma_alpha_rad\": " << JsonNumber(s->GetSigmaAlpha()) << ",\n"
    << "    \"has_properties_table\": " << (s->GetMaterialPropertiesTable() ? "true" : "false") << ",\n"
    << "    \"border_surface_found\": " << (surfaceFound ? "true" : "false") << ",\n"
    << "    \"border_surface_found_in_step\": " << (surfaceFoundInStep ? "true" : "false") << "\n"
    << "  },\n"
    << "  \"table_energies_ev\": [" << JsonNumber(e[0] / eV) << ", " << JsonNumber(e[1] / eV) << "],\n"
    << "  \"optical_photon_processes\": [" << procs.str() << "],\n"
    << "  \"columns\": [\"dx\", \"dy\", \"dz\", \"px\", \"py\", \"pz\", \"status\", \"alive\"],\n"
    << "  \"status_names\": [" << names.str() << "],\n"
    << "  \"n_photons\": " << rows << ",\n"
    << "  \"n_unexpected_steps\": " << unexpected << "\n"
    << "}\n";
  if (!f) Fail("error writing " + path);
}

std::string SidecarPath(const std::string& npy)
{
  const std::string ext = ".npy";
  if (npy.size() >= ext.size() && npy.compare(npy.size() - ext.size(), ext.size(), ext) == 0)
    return npy.substr(0, npy.size() - ext.size()) + ".json";
  return npy + ".json";
}

}  // namespace

int main(int argc, char** argv)
{
  const Options opt = Parse(argc, argv);

  auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
  runManager->SetVerboseLevel(0);
  G4Random::setTheSeed(opt.seed);

  auto* det = new Detector(opt);
  runManager->SetUserInitialization(det);
  runManager->SetUserInitialization(new PhysicsList());

  std::vector<double> rows;
  rows.reserve(static_cast<std::size_t>(opt.n) * kColumns);
  auto* stepping = new Stepping(*det, rows);
  runManager->SetUserAction(new Primaries(opt));
  runManager->SetUserAction(stepping);
  runManager->Initialize();

  const bool surfaceFound = G4LogicalBorderSurface::GetSurface(det->PvA(), det->PvB()) != nullptr;
  const auto processes = OpticalPhotonProcesses();
  std::cout << "boundary: border surface A->B found: " << (surfaceFound ? "yes" : "no")
            << "; sigma_alpha = " << det->Surface()->GetSigmaAlpha()
            << "; polish = " << det->Surface()->GetPolish() << "; optical photon processes:";
  for (const auto& p : processes) std::cout << " " << p;
  std::cout << std::endl;

  const long events = (opt.n + opt.perEvent - 1) / opt.perEvent;
  const auto t0 = std::chrono::steady_clock::now();
  runManager->BeamOn(static_cast<G4int>(events));
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  const std::size_t nRows = rows.size() / kColumns;
  WriteNpy(opt.out, rows, nRows);
  WriteJson(SidecarPath(opt.out), opt, *det, surfaceFound, stepping->SurfaceFoundInStep(), processes,
            nRows, stepping->Unexpected());
  std::cout << "boundary: " << nRows << " photons in " << seconds << " s (BeamOn), "
            << stepping->Unexpected() << " unexpected steps -> " << opt.out << std::endl;

  delete runManager;
  return nRows == static_cast<std::size_t>(opt.n) && stepping->Unexpected() == 0 ? 0 : 1;
}
