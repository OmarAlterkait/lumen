const $ = (id) => document.getElementById(id);
const HC = 1239.841984;  // eV nm
const EMISSION = {
  delta: 'delta: WLSTIMECONSTANT itself (Geant4\'s default)',
  exponential: 'exponential: drawn from Exp(WLSTIMECONSTANT)',
};
const ROLE = {
  medium: ['#4fc3f7', 'transparent medium'],
  wls: ['#66bb6a', 'wavelength shifter'],
  scint: ['#ba68c8', 'scintillator'],
  sensor: ['#ffd54f', 'sensor (a recipe group)'],
  opaque: ['#78909c', 'opaque: no RINDEX'],
  refused: ['#ef5350', 'refused by lumen'],
  bound: ['#90a4ae', 'bound to another medium'],
  none: ['#5f6b70', 'no material'],
};
const NOT_READ = {
  GROUPVEL: 'not read: lumen times photons by the group index of RINDEX (Geant4 computes GROUPVEL from it)',
  EFFICIENCY: 'not read on a material (Geant4 reads EFFICIENCY on surfaces)',
  REFLECTIVITY: 'not read on a material (Geant4 reads REFLECTIVITY on surfaces)',
};
const PARAM_NAMES = {
  index: 'refractive index', absorption: 'absorption length', rayleigh: 'Rayleigh length', mie: 'Mie',
  wls: 'wavelength shifting', source: 'scintillation source',
  R: 'reflectance', T: 'transmittance', eff: 'detection efficiency', p_spike: 'specular spike share',
  p_lobe: 'specular lobe share', p_back: 'backscatter share', sigma_alpha: 'facet slope (sigma_alpha)', polish: 'polish',
};

let S = null;                 // the scene payload
let I = null;                 // decoded instances {lv, parent, pv, depth, inside, matrix, n}
let lvGids = [];              // per LV: Int32Array of its instance ids
let lvObjects = [];           // per LV: {mesh: InstancedMesh|null, geom, mat, color, transparent}
let geoms = [];               // per mesh index: BufferGeometry
let renderer, scene, camera, controls, light;
let axScene = null, axCam = null;  // the axis triad, drawn in a corner with the camera's orientation
const AXES = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
const ACROSS = { z: ['x', 'y'], x: ['y', 'z'], y: ['z', 'x'] };  // the horizontal axes under each up, right-handed
let upAxis = 'z';             // the axis drawn vertical; the scene's meta.up sets it
let clipPlane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0);
let clipAxis = 'off', clipFlip = false;
let sceneBox = new THREE.Box3(), envBox = new THREE.Box3();  // what is drawn, and the detector volume itself
let colorMode = 'role', showMode = 'detector';
let mediaOpacity = 0.12, solidOpacity = 1.0, opaqueSolid = true, isolate = true, edgesOn = false;
let depthMax = 99, bulkMax = 50000;
let lvVisible = [];           // per LV: 'auto' | 'on' | 'off'
let selection = null;         // {kind, name, lvs: Set<int>, gid?}
let pickHelper = null;
let tab = 'volumes';
let plotState = null;         // {title, series: [{e, v, label, color}], unit}
let logY = false, energyX = false;
let dirty = true;             // draw a frame: the view or the scene changed (no frame is drawn while idle)

function setStatus(s) { $('status').textContent = s || ''; }
function toast(msg) { const t = $('toast'); t.textContent = msg; t.classList.add('visible'); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove('visible'), 2200); }
function esc(s) { return String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
function b64(s, T) {
  const bin = atob(s); const n = bin.length; const u8 = new Uint8Array(n);
  for (let i = 0; i < n; i++) u8[i] = bin.charCodeAt(i);
  return new T(u8.buffer);
}
function fmt(x, p = 4) {
  if (x === null || x === undefined || Number.isNaN(x)) return '–';
  if (typeof x !== 'number') return esc(x);
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e5 || a < 1e-3) return x.toExponential(p - 1).replace('e+', 'e');
  return Number(x.toPrecision(p)).toString();
}
function hashHue(s) { let h = 2166136261; for (const c of s) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return ((h >>> 0) % 1000) / 1000; }
function catColor(i) { const c = new THREE.Color(); c.setHSL((i * 0.618034) % 1, 0.62, 0.6); return '#' + c.getHexString(); }
function viridis(t) {
  const stops = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  t = Math.min(1, Math.max(0, t)) * (stops.length - 1); const i = Math.min(stops.length - 2, Math.floor(t)); const f = t - i;
  const c = stops[i].map((a, k) => Math.round(a + f * (stops[i + 1][k] - a)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function badge(text, cls, title = '') { return `<span class="badge ${cls}" title="${esc(title)}">${esc(text)}</span>`; }

function material(name) { return name ? S.materials[name] : null; }
function lvRole(v) {
  if (v.sensor) return 'sensor';
  const m = material(v.material);
  if (!m) return 'none';
  if (m.status === 'refused') return 'refused';
  if (m.status === 'opaque') return 'opaque';
  if (m.status === 'bound') return 'bound';
  if (m.channels.includes('wls')) return 'wls';
  if (m.scintillation) return 'scint';
  return 'medium';
}
function hasIndex(m) { return !!m && m.properties.some((p) => p.key === 'RINDEX'); }
function lvTransparent(v) {
  const role = lvRole(v);
  if (role === 'sensor') return false;
  if (role === 'refused') return hasIndex(material(v.material));
  return ['medium', 'wls', 'scint', 'bound'].includes(role);
}
function surfaceStatusClass(s) { return s.status === 'ok' ? 'b-ok' : (s.status === 'sensor' ? 'b-sensor' : 'b-refused'); }
function surfaceLabel(s) { return s.status === 'ok' ? s.surface_type : (s.status === 'sensor' ? `sensor: ${s.sensor_type}` : 'REFUSED'); }
function lvSurfaces(v) { return [...new Set([...(v.skins || []), ...(v.borders || [])])]; }
function surfaceColor(name) {
  const s = S.surfaces[name];
  if (!s) return '#555';
  if (s.status === 'refused') return ROLE.refused[0];
  if (s.status === 'sensor') return ROLE.sensor[0];
  const c = new THREE.Color(); c.setHSL(hashHue(s.surface_type || name), 0.55, 0.6); return '#' + c.getHexString();
}
function lvColor(i) {
  const v = S.volumes[i];
  if (colorMode === 'role') return ROLE[lvRole(v)][0];
  if (colorMode === 'material') return v.material ? matColors[v.material] : ROLE.none[0];
  if (colorMode === 'depth') return viridis(maxDepth ? v.depth / maxDepth : 0);
  const surfs = lvSurfaces(v);
  return surfs.length ? surfaceColor(surfs[0]) : '#3a4247';
}
let matColors = {}, maxDepth = 1;

async function load() {
  $('loadStatus').textContent = 'fetching scene.json ...';
  const r = await fetch('scene.json');
  if (!r.ok) throw new Error(`scene.json: HTTP ${r.status}`);
  $('loadStatus').textContent = 'parsing ...';
  S = await r.json();
  if (S.instances === undefined) throw new Error((S.errors || []).map((e) => `${e.stage}: ${e.text}`).join('\n') || 'empty scene');
  const d = S.instances;
  I = { lv: b64(d.lv, Int32Array), parent: b64(d.parent, Int32Array), pv: b64(d.pv, Int32Array), depth: b64(d.depth, Int32Array),
        inside: b64(d.inside, Uint8Array), matrix: b64(d.matrix, Float32Array), n: d.n };
  const counts = new Int32Array(S.volumes.length);
  for (let g = 0; g < I.n; g++) counts[I.lv[g]]++;
  lvGids = S.volumes.map((_, i) => new Int32Array(counts[i]));
  const fill = new Int32Array(S.volumes.length);
  for (let g = 0; g < I.n; g++) { const l = I.lv[g]; lvGids[l][fill[l]++] = g; }
  Object.keys(S.materials).forEach((m, i) => { matColors[m] = catColor(i); });
  maxDepth = Math.max(1, ...S.volumes.map((v) => v.depth));
  lvVisible = S.volumes.map(() => 'auto');
  if (!S.meta.detector_volume) showMode = 'all';
  $('loadStatus').textContent = 'building meshes ...';
  geoms = S.meshes.map((m) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(b64(m.v, Float32Array), 3));
    g.setIndex(new THREE.BufferAttribute(b64(m.t, Uint32Array), 1));
    g.computeVertexNormals(); g.computeBoundingSphere(); g.computeBoundingBox();
    return g;
  });
}

function initThree() {
  const host = $('panel3d');
  renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x080808);
  renderer.localClippingEnabled = true;
  host.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, 1, 1e-4, 1e5);
  camera.up.set(...AXES[upAxis]);
  makeControls();
  initAxes();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x303040, 0.75));
  light = new THREE.DirectionalLight(0xffffff, 0.65); camera.add(light); light.position.set(0.3, 0.6, 1);
  scene.add(camera);
  new ResizeObserver(resize).observe(host);
  resize();
  let down = null;
  renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4 && e.button === 0) pick(e);
    down = null;
  });
  (function loop() {
    requestAnimationFrame(loop);
    if (controls.update() || dirty) { renderer.render(scene, camera); renderAxes(); dirty = false; }
  })();
}
// OrbitControls (r128) fixes the up axis it orbits around when it is made: a new up needs new controls.
function makeControls() {
  if (controls) controls.dispose();
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.12; controls.screenSpacePanning = true;
}
function setUp(v) {
  upAxis = v; setActive('upGrp', v); camera.up.set(...AXES[v]); makeControls(); fitCamera();
}
function axisLabel(text, color) {
  const c = document.createElement('canvas'); c.width = c.height = 64;
  const g = c.getContext('2d'); g.fillStyle = color; g.font = 'bold 44px monospace';
  g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(text, 32, 34);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  s.scale.set(0.5, 0.5, 1); return s;
}
function initAxes() {
  axScene = new THREE.Scene();
  axCam = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
  const colors = { x: '#ef5350', y: '#66bb6a', z: '#42a5f5' };
  for (const [k, v] of Object.entries(AXES)) {
    const d = new THREE.Vector3(...v);
    axScene.add(new THREE.ArrowHelper(d, new THREE.Vector3(), 1, new THREE.Color(colors[k]).getHex(), 0.28, 0.14));
    const s = axisLabel(k.toUpperCase(), colors[k]); s.position.copy(d).multiplyScalar(1.35); axScene.add(s);
  }
}
function renderAxes() {
  const host = $('panel3d'), w = host.clientWidth, n = 96, x = w - n - 10, y = 26;
  axCam.quaternion.copy(camera.quaternion);
  axCam.position.set(0, 0, 6.5).applyQuaternion(camera.quaternion);
  renderer.autoClear = false; renderer.clearDepth();
  renderer.setScissorTest(true); renderer.setScissor(x, y, n, n); renderer.setViewport(x, y, n, n);
  renderer.render(axScene, axCam);
  renderer.setScissorTest(false); renderer.setViewport(0, 0, w, host.clientHeight); renderer.autoClear = true;
}
function resize() {
  const host = $('panel3d'); const w = host.clientWidth, h = host.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); dirty = true;
}

function instanceShown(g) {
  if (showMode === 'detector' && S.meta.detector_volume && !I.inside[g]) return false;
  if (I.depth[g] === 0) return false;  // the world
  return I.depth[g] <= depthMax;
}
function lvShown(i) {
  const v = S.volumes[i];
  if (v.mesh === null || v.mesh === undefined || !lvGids[i].length) return false;
  if (lvVisible[i] === 'on') return true;
  if (lvVisible[i] === 'off') return false;
  return v.instances <= bulkMax;
}

function buildObjects() {
  lvObjects.forEach((o) => { if (o && o.mesh) { scene.remove(o.mesh); o.mat.dispose(); } });
  lvObjects = S.volumes.map((v, i) => {
    if (v.mesh === null || v.mesh === undefined) return null;
    const mat = new THREE.MeshPhongMaterial({ color: 0xffffff, side: THREE.DoubleSide, flatShading: true, shininess: 18 });
    const mesh = new THREE.InstancedMesh(geoms[v.mesh], mat, Math.max(1, lvGids[i].length));
    mesh.userData.lv = i; mesh.count = 0; mesh.frustumCulled = false;
    scene.add(mesh);
    return { mesh, mat, gids: new Int32Array(0) };
  });
  refreshInstances(); refreshMaterials();
}

function refreshInstances() {
  const m4 = new THREE.Matrix4();
  sceneBox.makeEmpty(); envBox.makeEmpty();
  const p = new THREE.Vector3();
  lvObjects.forEach((o, i) => {
    if (!o) return;
    const all = lvGids[i];
    const keep = lvShown(i) ? all.filter(instanceShown) : new Int32Array(0);
    o.gids = keep;
    const r = geoms[S.volumes[i].mesh].boundingSphere.radius;
    const envelope = S.volumes[i].name === S.meta.detector_volume;  // framed by its contents, not its own size
    const box = envelope ? envBox : sceneBox;
    for (let k = 0; k < keep.length; k++) {
      const a = I.matrix.subarray(keep[k] * 12, keep[k] * 12 + 12);
      m4.set(a[0], a[1], a[2], a[9], a[3], a[4], a[5], a[10], a[6], a[7], a[8], a[11], 0, 0, 0, 1);
      o.mesh.setMatrixAt(k, m4);
      if (k < 20000 || k % 50 === 0) { p.set(a[9], a[10], a[11]); box.expandByPoint(p.clone().addScalar(r)); box.expandByPoint(p.clone().addScalar(-r)); }
    }
    o.mesh.count = keep.length;
    o.mesh.instanceMatrix.needsUpdate = true;
    o.mesh.visible = keep.length > 0;
  });
  if (sceneBox.isEmpty()) sceneBox.copy(envBox);
  updateClip();
}

function refreshMaterials() {
  const planes = clipAxis === 'off' ? [] : [clipPlane];
  const sel = selection ? selection.lvs : null;
  lvObjects.forEach((o, i) => {
    if (!o) return;
    const v = S.volumes[i];
    const tr = lvTransparent(v);
    const chosen = sel && sel.has(i);
    let op = tr ? mediaOpacity : (opaqueSolid ? solidOpacity : Math.min(solidOpacity, 0.35));
    if (sel && isolate && !chosen) op = Math.min(op, 0.05);
    if (chosen && tr) op = Math.max(op, 0.4);
    o.mat.color.set(lvColor(i));
    o.mat.emissive.set(chosen ? lvColor(i) : '#000000');
    o.mat.emissiveIntensity = chosen ? 0.35 : 0;
    o.mat.opacity = op;
    o.mat.transparent = op < 0.999;
    o.mat.depthWrite = op >= 0.999;
    o.mat.clippingPlanes = planes;
    o.mat.needsUpdate = true;
    o.mesh.renderOrder = o.mat.transparent ? 2 : 1;
  });
  if (pickHelper) pickHelper.material.clippingPlanes = planes;
  renderLegend(); dirty = true;
}

function fitCamera() {
  if (sceneBox.isEmpty()) return;
  // From outside the scene, 45 degrees around the up axis and about 17 degrees above the horizontal.
  const c = sceneBox.getCenter(new THREE.Vector3()), ext = sceneBox.getSize(new THREE.Vector3());
  const [a, b] = ACROSS[upAxis], half = Math.max(ext[a], ext[b], ext[upAxis]) / 2;
  const dist = half * 3.2, azim = Math.PI * 0.25, elev = 0.3;
  const dir = new THREE.Vector3(...AXES[a]).multiplyScalar(Math.cos(elev) * Math.cos(azim))
    .add(new THREE.Vector3(...AXES[b]).multiplyScalar(Math.cos(elev) * Math.sin(azim)))
    .add(new THREE.Vector3(...AXES[upAxis]).multiplyScalar(Math.sin(elev)));
  camera.near = ext.length() * 1e-5; camera.far = dist * 50; camera.updateProjectionMatrix();
  camera.position.copy(c).add(dir.multiplyScalar(dist));
  controls.target.copy(c); controls.update(); dirty = true;
}

function updateClip() {
  if (clipAxis === 'off' || sceneBox.isEmpty()) return;
  const ax = { x: 0, y: 1, z: 2 }[clipAxis];
  const lo = sceneBox.min.getComponent(ax), hi = sceneBox.max.getComponent(ax);
  const pos = lo + (hi - lo) * parseFloat($('clipPos').value);
  const n = new THREE.Vector3(); n.setComponent(ax, clipFlip ? 1 : -1);
  clipPlane.set(n, clipFlip ? -pos : pos); dirty = true;
}

function pick(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  const ray = new THREE.Raycaster(); ray.setFromCamera(ndc, camera);
  const objs = lvObjects.filter((o) => o && o.mesh.visible && o.mesh.count > 0).map((o) => o.mesh);
  const hits = ray.intersectObjects(objs, false).filter((h) =>
    (clipAxis === 'off' || clipPlane.distanceToPoint(h.point) >= 0) && h.object.material.opacity > 0.06);
  if (!hits.length) { select(null); return; }
  const h = hits[0]; const lv = h.object.userData.lv; const gid = lvObjects[lv].gids[h.instanceId];
  select({ kind: 'volume', name: lv, gid });
}

function showPicked(gid, lv) {
  if (pickHelper) { scene.remove(pickHelper); pickHelper.geometry.dispose(); pickHelper = null; dirty = true; }
  if (gid === undefined || gid === null || !edgesOn) return;
  const geom = geoms[S.volumes[lv].mesh];
  const n = geom.index ? geom.index.count / 3 : 0;
  const eg = n < 300000 ? new THREE.EdgesGeometry(geom, 25) : new THREE.EdgesGeometry(new THREE.BoxGeometry(...geom.boundingBox.getSize(new THREE.Vector3()).toArray()));
  pickHelper = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: 0x29b6f6, clippingPlanes: clipAxis === 'off' ? [] : [clipPlane] }));
  const a = I.matrix.subarray(gid * 12, gid * 12 + 12); const m4 = new THREE.Matrix4();
  m4.set(a[0], a[1], a[2], a[9], a[3], a[4], a[5], a[10], a[6], a[7], a[8], a[11], 0, 0, 0, 1);
  if (n >= 300000) { const c = geom.boundingBox.getCenter(new THREE.Vector3()); m4.multiply(new THREE.Matrix4().makeTranslation(c.x, c.y, c.z)); }
  pickHelper.applyMatrix4(m4); pickHelper.renderOrder = 5; scene.add(pickHelper); dirty = true;
}

function lvsOfMaterial(name) { const s = new Set(); S.volumes.forEach((v, i) => { if (v.material === name) s.add(i); }); return s; }
function lvsOfSurface(name) { const s = new Set(); S.volumes.forEach((v, i) => { if (lvSurfaces(v).includes(name)) s.add(i); }); return s; }
function lvsOfGroup(name) { const s = new Set(); S.volumes.forEach((v, i) => { if (v.sensor === name) s.add(i); }); return s; }

function select(sel) {
  if (sel && sel.kind === 'volume') sel.lvs = new Set([sel.name]);
  if (sel && sel.kind === 'material') sel.lvs = lvsOfMaterial(sel.name);
  if (sel && sel.kind === 'surface') sel.lvs = lvsOfSurface(sel.name);
  if (sel && sel.kind === 'group') sel.lvs = lvsOfGroup(sel.name);
  selection = sel;
  refreshMaterials();
  showPicked(sel && sel.kind === 'volume' ? sel.gid : null, sel ? sel.name : null);
  renderDetails(); renderList();
}

function renderList() {
  const q = $('search').value.trim().toLowerCase();
  const el = $('list');
  if (tab === 'volumes') el.innerHTML = volumeRows(q);
  else if (tab === 'materials') el.innerHTML = materialRows(q);
  else if (tab === 'surfaces') el.innerHTML = surfaceRows(q);
  else if (tab === 'sensors') el.innerHTML = sensorRows();
  else el.innerHTML = reportHtml();
}

function volumeTree() {
  const kids = S.volumes.map(() => []); const roots = [];
  S.volumes.forEach((v, i) => {
    const g = lvGids[i][0];
    if (g === undefined) return;
    const par = I.parent[g];
    if (par < 0) roots.push(i); else kids[I.lv[par]].push(i);
  });
  return { kids, roots };
}
let treeCache = null;
function volumeRows(q) {
  const out = [];
  const row = (i, depth) => {
    const v = S.volumes[i]; const role = lvRole(v);
    const sel = selection && selection.kind === 'volume' && selection.name === i;
    const on = lvShown(i);
    const bulk = v.instances > bulkMax && lvVisible[i] === 'auto';
    const tag = bulk ? badge('HIDDEN', 'b-dim', `placed ${v.instances.toLocaleString()} times, more than the setting allows (⚙): tick to draw`)
      : badge(role === 'medium' ? 'MEDIUM' : role.toUpperCase(), 'b-' + (role === 'bound' || role === 'none' ? 'dim' : role));
    out.push(`<div class="row ${sel ? 'selected' : ''}" data-lv="${i}" style="padding-left:${6 + Math.min(depth, 10) * 11}px">
      <input type="checkbox" data-vis="${i}" ${on ? 'checked' : ''} ${v.mesh === null ? 'disabled' : ''}>
      <span class="sw" style="background:${lvColor(i)}"></span>
      <span class="nm" title="${esc(v.name)}">${esc(v.short)}</span>
      <span class="meta">${v.instances > 1 ? '×' + v.instances.toLocaleString() : ''}</span>${tag}</div>`);
  };
  if (q) {
    S.volumes.forEach((v, i) => { if (v.short.toLowerCase().includes(q) || (v.material || '').toLowerCase().includes(q)) row(i, 0); });
  } else {
    treeCache = treeCache || volumeTree();
    const walk = (i, d) => { row(i, d); treeCache.kids[i].forEach((k) => walk(k, d + 1)); };
    treeCache.roots.forEach((r) => walk(r, 0));
  }
  return out.join('') || '<div class="empty">no volume</div>';
}

function materialBadges(m) {
  const b = [badge(m.status === 'medium' ? 'MEDIUM' : m.status.toUpperCase(), 'b-' + (m.status === 'bound' ? 'dim' : m.status))];
  const short = { refraction: null, absorption: 'ABS', rayleigh: 'RAY', mie: 'MIE', wls: 'WLS' };
  m.channels.forEach((c) => { if (short[c]) b.push(badge(short[c], c === 'wls' ? 'b-wls' : 'b-medium')); });
  if (m.scintillation) b.push(badge('SCINT', 'b-scint'));
  if (m.inert.length) b.push(badge(`${m.inert.length} INERT`, 'b-dim', 'keys no Geant4 optical process reads: kept, never read'));
  return b.join(' ');
}
function materialRows(q) {
  const order = { refused: 0, medium: 1, bound: 2, opaque: 3 };
  const names = Object.keys(S.materials).filter((n) => !q || n.toLowerCase().includes(q))
    .sort((a, b) => (order[S.materials[a].status] - order[S.materials[b].status]) || a.localeCompare(b));
  return names.map((n) => {
    const m = S.materials[n]; const nv = lvsOfMaterial(n).size;
    const sel = selection && selection.kind === 'material' && selection.name === n;
    return `<div class="row ${sel ? 'selected' : ''}" data-mat="${esc(n)}">
      <span class="sw" style="background:${colorMode === 'material' ? matColors[n] : ROLE[m.status === 'medium' ? (m.channels.includes('wls') ? 'wls' : (m.scintillation ? 'scint' : 'medium')) : (m.status in ROLE ? m.status : 'none')][0]}"></span>
      <span class="nm" title="${esc(n)}">${esc(m.short)}</span><span class="meta">${nv} vol</span> ${materialBadges(m)}</div>`;
  }).join('') || '<div class="empty">no material</div>';
}
function surfaceRows(q) {
  const names = Object.keys(S.surfaces).filter((n) => !q || n.toLowerCase().includes(q))
    .sort((a, b) => (S.surfaces[a].status === 'refused' ? 0 : 1) - (S.surfaces[b].status === 'refused' ? 0 : 1) || a.localeCompare(b));
  return names.map((n) => {
    const s = S.surfaces[n]; const sel = selection && selection.kind === 'surface' && selection.name === n;
    const where = [s.skins.length ? `skin×${s.skins.length}` : '', s.n_borders ? `border×${s.n_borders}` : ''].filter(Boolean).join(' ');
    return `<div class="row ${sel ? 'selected' : ''}" data-surf="${esc(n)}">
      <span class="sw" style="background:${surfaceColor(n)}"></span>
      <span class="nm" title="${esc(n)}">${esc(s.short)}</span><span class="meta">${where}</span>
      ${badge(surfaceLabel(s), surfaceStatusClass(s))}</div>`;
  }).join('') || '<div class="empty">no optical surface</div>';
}
function sensorRows() {
  if (!S.sensor_groups.length) {
    return `<div class="msg info">No sensor groups: this GDML was read without a recipe. Read it with one
      (<code>--recipe FILE.py:NAME</code>), or as a detector package's detector (<code>package:name</code>), to see
      lumen's sensors and detector volume.</div>`;
  }
  const out = [];
  S.sensor_groups.forEach((g) => {
    const lvs = [...lvsOfGroup(g.name)]; const n = lvs.reduce((a, i) => a + S.volumes[i].inside, 0);
    const sel = selection && selection.kind === 'group' && selection.name === g.name;
    out.push(`<div class="row ${sel ? 'selected' : ''}" data-group="${esc(g.name)}"><span class="sw" style="background:${ROLE.sensor[0]}"></span>
      <span class="nm">${esc(g.name)}</span><span class="meta">${n.toLocaleString()} placed</span>${badge(g.rule, 'b-sensor')}</div>`);
  });
  if (S.sensor_ids !== null) out.push(`<div class="msg info">lumen numbers <b>${S.sensor_ids.toLocaleString()}</b> sensor ids for this recipe (one per placed copy inside the detector volume).</div>`);
  return out.join('');
}
function reportHtml() {
  const m = S.meta; const c = m.counts || {};
  const kv = (k, v) => `<div class="k">${esc(k)}</div><div class="v">${v}</div>`;
  let h = `<div class="group-title">FILE</div><div class="kv">${kv('file', esc(m.file))}${kv('sha256', esc((m.sha256 || '').slice(0, 16)) + '…')}
    ${kv('lumen', esc(m.lumen))}${kv('recipe', m.recipe ? esc(m.recipe) : '<i>none</i>')}${kv('world', esc(m.world))}
    ${kv('detector volume', m.detector_volume ? `<a class="lnk" data-lvname="${esc(m.detector_volume)}">${esc(m.detector_volume)}</a>${m.detector_guessed ? ' <span class="badge b-warn" title="no recipe: the largest volume of a transparent material, as lumen\'s auto: rule picks it">GUESSED</span>' : ''}` : '<i>unknown (no recipe)</i>')}
    ${kv('wavelengths', m.wavelength_range ? `${m.wavelength_range[0]}–${m.wavelength_range[1]} nm` : '–')}
    ${kv('counts', Object.entries(c).map(([k, v]) => `${k} ${Number(v).toLocaleString()}`).join(', '))}
    ${kv('built in', `${m.seconds} s`)}</div>`;
  if (S.errors.length) h += `<div class="group-title">REFUSED / FAILED</div>` + S.errors.map((e) => `<div class="msg err"><b>${esc(e.stage)}</b>: ${esc(e.text)}</div>`).join('');
  const refusedM = Object.values(S.materials).filter((x) => x.status === 'refused');
  const refusedS = Object.values(S.surfaces).filter((x) => x.status === 'refused');
  if (refusedM.length || refusedS.length) {
    h += `<div class="group-title">WHAT LUMEN REFUSES</div>`;
    refusedM.forEach((x) => { h += `<div class="msg err"><a class="lnk" data-mat="${esc(x.name)}">material ${esc(x.short)}</a>: ${esc(x.message)}</div>`; });
    refusedS.forEach((x) => { h += `<div class="msg err"><a class="lnk" data-surf="${esc(x.name)}">surface ${esc(x.short)}</a>: ${esc(x.message)}</div>`; });
  }
  if (S.notes.length) h += `<div class="group-title">WARNINGS</div>` + S.notes.map((n) => `<div class="msg warn"><b>${esc(n.kind)}</b> (${esc(n.stage)}): ${esc(n.text)}</div>`).join('');
  const approx = S.meshes.filter((x) => !x.exact);
  if (approx.length) h += `<div class="group-title">DRAWN APPROXIMATELY</div>` + approx.map((x) => `<div class="report-line">${esc(x.solid)}: ${esc(x.note)}</div>`).join('');
  if (m.instances_cut) h += `<div class="msg warn">${m.instances_cut.toLocaleString()} placed copies were left out (more than the viewer draws).</div>`;
  if (S.report.length) h += `<div class="group-title">IMPORT REPORT</div>` + S.report.map((l) => `<div class="report-line">${esc(l)}</div>`).join('');
  return h;
}

function lvLink(name) {
  const i = S.volumes.findIndex((v) => v.name === name);
  return i < 0 ? esc(name) : `<a class="lnk" data-lv="${i}">${esc(S.volumes[i].short)}</a>`;
}
function matLink(name) { return name && S.materials[name] ? `<a class="lnk" data-mat="${esc(name)}">${esc(S.materials[name].short)}</a>` : esc(name || '–'); }
function surfLink(name) { return S.surfaces[name] ? `<a class="lnk" data-surf="${esc(name)}">${esc(S.surfaces[name].short)}</a>` : esc(name); }

function instancePath(gid) {
  const parts = [];
  for (let g = gid; g >= 0; g = I.parent[g]) {
    const p = I.pv[g];
    const v = S.volumes[I.lv[g]];
    parts.push(p < 0 ? `<b>${esc(v.short)}</b>` : `<b>${esc(S.placements[p].short)}</b>${S.placements[p].copy !== null && S.placements[p].copy !== undefined ? '#' + S.placements[p].copy : ''}`);
  }
  return parts.reverse().join(' / ');
}

function renderDetails() {
  const el = $('detailsBody');
  if (!selection) { el.innerHTML = '<div class="empty">Select a volume, a material or a surface.</div>'; plotState = null; return; }
  if (selection.kind === 'volume') el.innerHTML = volumeHtml(selection.name, selection.gid);
  else if (selection.kind === 'material') el.innerHTML = materialHtml(S.materials[selection.name]);
  else if (selection.kind === 'surface') el.innerHTML = surfaceHtml(S.surfaces[selection.name]);
  else el.innerHTML = groupHtml(selection.name);
  drawPlot();
}

function volumeHtml(i, gid) {
  const v = S.volumes[i]; const role = lvRole(v); const m = material(v.material);
  const kv = (k, val) => `<div class="k">${esc(k)}</div><div class="v">${val}</div>`;
  let h = `<div class="d-kind">VOLUME</div><div class="d-title">${esc(v.short)}</div><div class="d-full">${esc(v.name)}</div>
    <div class="d-badges">${badge(ROLE[role][1], 'b-' + (role === 'bound' || role === 'none' ? 'dim' : role))}
    ${v.sensor ? badge('group ' + v.sensor, 'b-sensor') : ''}${S.meta.detector_volume === v.name ? badge(S.meta.detector_guessed ? 'DETECTOR VOLUME (GUESSED)' : 'DETECTOR VOLUME', S.meta.detector_guessed ? 'b-warn' : 'b-ok') : ''}</div>`;
  if (gid !== undefined && gid !== null) h += `<div class="section">PICKED COPY</div><div class="path">${instancePath(gid)}</div>`;
  h += `<div class="section">VOLUME</div><div class="kv">${kv('material', matLink(v.material))}
    ${kv('placed', `${v.instances.toLocaleString()} ×${S.meta.detector_volume ? ` (${v.inside.toLocaleString()} in the detector volume)` : ''}`)}
    ${kv('daughters', v.daughters)}${kv('depth', v.depth)}
    ${kv('skin', v.skins.length ? v.skins.map(surfLink).join(', ') : '–')}
    ${kv('borders', v.borders.length ? v.borders.map(surfLink).join(', ') : '–')}</div>`;
  if (m) {
    const msg = m.status === 'refused' ? `<div class="msg err">${esc(m.message)}</div>`
      : (m.status === 'opaque' ? `<div class="msg info">No RINDEX: lumen stops a photon at this volume's boundary (Geant4's NoRINDEX), unless a surface says otherwise.</div>` : '');
    h += msg;
  }
  if (v.crossings) {
    const cls = { none: 'b-dim', pass: 'b-dim', absorb: 'b-opaque', fresnel: 'b-medium', surface: 'b-ok', sensor: 'b-sensor', refused: 'b-refused' };
    const line = (c, dir) => `<div class="kv" style="margin:4px 0">${kv(dir, `${badge(c.outcome.toUpperCase(), cls[c.outcome] || 'b-dim')}
      ${c.surface ? surfLink(c.surface) + ` <span class="badge b-dim">${esc(c.via)}</span>` : ''}<br><span style="color:#999">${esc(c.text)}</span>`)}</div>`;
    h += `<div class="section">AT ITS BOUNDARY</div>${v.crossings.map((c) => line(c, (c.dir === 'in' ? 'entering from ' : 'leaving into ') + c.partner)).join('')}
      <div style="color:#555;font-size:10px">lumen's boundary rule (border, then skins, then the materials) for the first copy: with its mother, and across every border surface it takes part in.</div>`;
  }
  if (v.solid_info) {
    const s = v.solid_info; const mesh = S.meshes[v.mesh];
    h += `<div class="section">SOLID</div><div class="kv">${kv('name', esc(v.solid))}${kv('type', esc(s.type))}
      ${Object.entries(s.params).map(([k, [x, u]]) => kv(k, `${fmt(x, 6)} ${esc(u)}`)).join('')}
      ${s.refs.length ? kv('operands', s.refs.map(esc).join(', ')) : ''}
      ${s.planes ? kv('planes (mm)', s.planes.map((p) => '(' + p.map((x) => fmt(x, 5)).join(', ') + ')').join(' ')) : ''}
      ${mesh ? kv('drawn as', `${mesh.nt.toLocaleString()} triangles${mesh.exact ? '' : ' (approximate)'}`) : ''}</div>
      ${mesh && !mesh.exact ? `<div class="msg warn">${esc(mesh.note)}</div>` : ''}`;
  }
  return h;
}

function readLabels(reads) {
  // one label per parameter: a table read as its energies and values, or derived from them, is one parameter
  const by = new Map();
  reads.forEach((r) => {
    const name = r.param === 'source' ? 'scintillation source' : (PARAM_NAMES[r.param.split('/')[0]] || r.param.split('/')[0]);
    const tag = r.column === 'derived' ? ' (derived)' : (r.column === 'spectrum' ? ' (spectrum)' : '');
    if (!by.has(name) || tag) by.set(name, tag || by.get(name) || '');
  });
  return [...by].map(([name, tag]) => name + tag);
}
function readsText(p, owner) {
  if (p.reads.length) {
    const ps = readLabels(p.reads);
    const keys = [...new Set(p.reads.map((r) => r.key).filter(Boolean))];
    return `<td class="read" title="${esc(keys.join('\n'))}">${esc(ps.join(', '))}</td>`;
  }
  let why = p.why || ((owner && owner.geant4) ? 'not read by lumen' : (NOT_READ[p.key] || 'not read by lumen'));
  if (owner && owner.status === 'opaque') why = 'not read: no RINDEX, the material is opaque';
  if (owner && owner.status === 'refused') why = 'not read: lumen refuses this material';
  return `<td class="read no">${esc(why)}</td>`;
}
function propRows(props, owner, idPrefix) {
  return props.map((p, k) => {
    const rng = p.kind === 'table' ? `${fmt(p.min)}–${fmt(p.max)}${p.unit ? ' ' + p.unit : ''}<br>${fmt(p.lam[0], 4)}–${fmt(p.lam[1], 4)} nm, ${p.n} knots` : `${fmt(p.value, 6)}${p.unit ? ' ' + p.unit : ''}`;
    const cov = p.covers === false ? ' ' + badge('CLAMPED', 'b-warn', "the table's knots do not cover the recipe's wavelength range: Geant4 and lumen keep the end values outside") : '';
    const org = p.origin && p.origin !== 'gdml' ? ' ' + badge(p.origin.toUpperCase(), 'b-dim', p.note || '') : '';
    const active = plotState && plotState.id === `${idPrefix}:${k}`;
    return `<tr class="${p.kind === 'table' ? 'plot' : ''} ${active ? 'active' : ''}" ${p.kind === 'table' ? `data-plot="${idPrefix}:${k}"` : ''}>
      <td class="key">${esc(p.key)}${cov}${org}</td>${readsText(p, owner)}<td class="rng">${rng}</td></tr>`;
  }).join('');
}
function materialHtml(m) {
  const kv = (k, val) => `<div class="k">${esc(k)}</div><div class="v">${val}</div>`;
  const vols = [...lvsOfMaterial(m.name)];
  let h = `<div class="d-kind">MATERIAL</div><div class="d-title">${esc(m.short)}</div><div class="d-full">${esc(m.name)}</div>
    <div class="d-badges">${materialBadges(m)}</div>`;
  if (m.status === 'refused') h += `<div class="msg err"><b>lumen refuses this material.</b> ${esc(m.message)}</div>`;
  if (m.status === 'opaque') h += `<div class="msg info">No RINDEX: lumen (as Geant4) stops a photon at the boundary of a volume of this material. Its other optical properties are never read.</div>`;
  m.warnings.forEach((w) => { h += `<div class="msg warn">${esc(w)}</div>`; });
  h += `<div class="section">LUMEN</div><div class="kv">${kv('optics model', esc(m.optics.model) + (m.optics.origin !== 'gdml' ? ` <span class="badge b-dim">${esc(m.optics.origin)}</span>` : ''))}
    ${m.optics.note ? kv('why', esc(m.optics.note)) : ''}${m.target && m.target !== m.name ? kv('medium', esc(m.target)) : ''}
    ${kv('channels', m.channels.length ? esc(m.channels.join(', ')) : '–')}
    ${Object.entries(m.emission || {}).map(([c, prof]) => kv(`${c} delay`, EMISSION[prof] || esc(prof))).join('')}
    ${m.scintillation ? kv('scintillation', `${fmt(m.scintillation.yield_per_MeV)} /MeV, read by the scintillation source`) : ''}
    ${kv('volumes', vols.map((i) => lvLink(S.volumes[i].name)).join(', ') || '–')}</div>`;
  if (m.properties.length || m.derived.length) {
    h += `<div class="section">PROPERTIES</div><table class="props"><tr><th>KEY</th><th>READ BY LUMEN AS</th><th>RANGE</th></tr>
      ${propRows(m.properties, m, 'm')}${propRows(m.derived, m, 'd')}</table>`;
    h += `<div id="plotWrap"><canvas id="plot"></canvas><div id="plotTip"></div></div>`;
  } else {
    h += `<div class="msg info">No optical properties in the GDML.</div>`;
  }
  if (m.inert.length) h += `<div class="section">KEPT INERT</div><div class="msg info">Keys no Geant4 optical process reads, kept verbatim and never read: ${m.inert.map(esc).join(', ')}.</div>`;
  h += `<div class="section">COMPOSITION</div><div class="kv">${kv('state', esc(m.state || '–'))}${kv('density', m.density ? fmt(m.density) + ' g/cm³' : '–')}
    ${m.temperature ? kv('temperature', fmt(m.temperature) + ' K') : ''}${m.pressure ? kv('pressure', fmt(m.pressure / 1e5) + ' bar') : ''}
    ${kv('components', m.composition.map((c) => `${esc(c.ref)} ${c.fraction !== null ? fmt(c.fraction, 3) : (c.atoms + ' atoms')}`).join(', ') || '–')}</div>`;
  return h;
}
function surfaceHtml(s) {
  const kv = (k, val) => `<div class="k">${esc(k)}</div><div class="v">${val}</div>`;
  let h = `<div class="d-kind">OPTICAL SURFACE</div><div class="d-title">${esc(s.short)}</div><div class="d-full">${esc(s.name)}</div>
    <div class="d-badges">${badge(surfaceLabel(s), surfaceStatusClass(s))}</div>`;
  if (s.status === 'refused') h += `<div class="msg err"><b>lumen refuses this surface.</b> ${esc(s.message)}</div>`;
  if (s.status === 'sensor') h += `<div class="msg info">A detecting surface on a sensor: lumen reads it with the group's surface reader as <b>${esc(s.sensor_type)}</b>.</div>`;
  h += `<div class="section">GEANT4</div><div class="kv">${kv('type', esc(s.geant4.type))}${kv('model', esc(s.geant4.model))}${kv('finish', esc(s.geant4.finish))}
    ${kv(s.geant4.value_name || 'value', fmt(s.geant4.value, 6))}${s.optics.model !== 'geant4' ? kv('optics model', esc(s.optics.model)) : ''}</div>`;
  h += `<div class="section">ATTACHED</div><div class="kv">${kv('skin on', s.skins.length ? s.skins.map(lvLink).join(', ') : '–')}
    ${kv('borders', s.n_borders ? `${s.n_borders.toLocaleString()} (${s.borders.slice(0, 6).map((b) => esc(b[0].split('0x')[0]) + ' → ' + esc(b[1].split('0x')[0])).join('; ')}${s.n_borders > 6 ? '; …' : ''})` : '–')}</div>`;
  if (s.params.length) {
    h += `<div class="section">LUMEN PARAMETERS</div><table class="props"><tr><th>PARAM</th><th>FROM</th><th>NOTE</th></tr>` +
      s.params.map((p) => `<tr><td class="key">${esc(p.param)}</td><td class="read ${p.key ? '' : 'no'}">${p.key ? esc(p.key.split('/').pop()) + ` <span class="badge b-dim">${esc(p.column)}</span>` : esc(p.column)}</td><td class="read">${esc(p.note)}</td></tr>`).join('') + '</table>';
  }
  if (s.properties.length) {
    h += `<div class="section">PROPERTIES</div><table class="props"><tr><th>KEY</th><th>READ BY LUMEN AS</th><th>RANGE</th></tr>${propRows(s.properties, s, 's')}</table>
      <div id="plotWrap"><canvas id="plot"></canvas><div id="plotTip"></div></div>`;
  }
  return h;
}
function groupHtml(name) {
  const g = S.sensor_groups.find((x) => x.name === name);
  const kv = (k, val) => `<div class="k">${esc(k)}</div><div class="v">${val}</div>`;
  const lvs = [...lvsOfGroup(name)];
  return `<div class="d-kind">SENSOR GROUP</div><div class="d-title">${esc(name)}</div><div class="d-badges">${badge(g.rule, 'b-sensor')}</div>
    <div class="kv">${kv('volumes', lvs.map((i) => lvLink(S.volumes[i].name)).join(', '))}${kv('selector', esc(g.lv || g.lv_pattern))}
    ${kv('counting', esc(g.rule))}${kv('values', esc(g.values || '–'))}${g.where ? kv('where', esc(JSON.stringify(g.where))) : ''}
    ${Object.entries(g.settings).map(([k, v]) => kv(k, esc(v))).join('')}
    ${kv('placed', lvs.reduce((a, i) => a + S.volumes[i].inside, 0).toLocaleString())}</div>`;
}

function choosePlot(id) {
  const [kind, k] = id.split(':');
  let p = null; let owner = null;
  if (selection.kind === 'material') { owner = S.materials[selection.name]; p = (kind === 'd' ? owner.derived : owner.properties)[+k]; }
  if (selection.kind === 'surface') { owner = S.surfaces[selection.name]; p = owner.properties[+k]; }
  if (!p || p.kind !== 'table') return;
  plotState = { id, title: `${owner.short} · ${p.key}`, e: p.energy, v: p.value, unit: p.unit };
  renderDetails();
}
function drawPlot() {
  const c = $('plot'); if (!c) return;
  if (!plotState || !selection || !plotState.id) {  // the first table of the selection, by default
    const owner = selection.kind === 'material' ? S.materials[selection.name] : (selection.kind === 'surface' ? S.surfaces[selection.name] : null);
    const k = owner ? owner.properties.findIndex((p) => p.kind === 'table') : -1;
    if (k < 0) { c.parentElement.style.display = 'none'; return; }
    const p = owner.properties[k];
    plotState = { id: `${selection.kind === 'material' ? 'm' : 's'}:${k}`, title: `${owner.short} · ${p.key}`, e: p.energy, v: p.value, unit: p.unit, auto: true };
    const row = document.querySelector(`tr[data-plot="${plotState.id}"]`); if (row) row.classList.add('active');
  }
  const dpr = window.devicePixelRatio || 1; const W = c.clientWidth, H = c.clientHeight;
  c.width = W * dpr; c.height = H * dpr; const g = c.getContext('2d'); g.scale(dpr, dpr);
  g.fillStyle = '#0c0c0c'; g.fillRect(0, 0, W, H);
  const xs = plotState.e.map((e) => energyX ? e : HC / e); const ys = plotState.v;
  const band = S.meta.wavelength_range ? (energyX ? [HC / S.meta.wavelength_range[1], HC / S.meta.wavelength_range[0]] : S.meta.wavelength_range) : null;
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  if (band) { x0 = Math.min(x0, band[0]); x1 = Math.max(x1, band[1]); }
  const pad = (x1 - x0) * 0.04 || 1; x0 -= pad; x1 += pad;
  const pos = ys.filter((y) => y > 0);
  const useLog = logY && pos.length > 0;
  let y0 = useLog ? Math.min(...pos) : Math.min(0, ...ys), y1 = Math.max(...ys);
  if (useLog) { y0 = Math.log10(y0); y1 = Math.log10(y1); if (y1 - y0 < 1e-9) { y0 -= 0.5; y1 += 0.5; } }
  if (y1 - y0 < 1e-30) { y1 = y0 + 1; }
  const yr = (y1 - y0); y1 += yr * 0.08; if (!useLog && y0 < 0) y0 -= yr * 0.08;
  const L = 58, R = 12, T = 22, B = 30;
  const X = (x) => L + (x - x0) / (x1 - x0) * (W - L - R);
  const Y = (y) => { const q = useLog ? Math.log10(Math.max(y, 1e-300)) : y; return T + (1 - (q - y0) / (y1 - y0)) * (H - T - B); };
  if (band) { g.fillStyle = 'rgba(41,182,246,0.07)'; g.fillRect(X(band[0]), T, X(band[1]) - X(band[0]), H - T - B); }
  g.strokeStyle = '#222'; g.lineWidth = 1; g.font = '10px monospace'; g.fillStyle = '#666';
  for (let k = 0; k <= 4; k++) {
    const q = y0 + (y1 - y0) * k / 4; const yy = T + (1 - k / 4) * (H - T - B);
    g.beginPath(); g.moveTo(L, yy); g.lineTo(W - R, yy); g.stroke();
    g.fillText(fmt(useLog ? 10 ** q : q, 3), 4, yy + 3);
  }
  for (let k = 0; k <= 5; k++) { const xx = x0 + (x1 - x0) * k / 5; g.fillText(fmt(xx, 4), X(xx) - 12, H - B + 14); }
  g.fillText(energyX ? 'photon energy (eV)' : 'wavelength (nm)', W / 2 - 40, H - 4);
  g.fillStyle = '#9fd3ee'; g.fillText(plotState.title + (plotState.unit ? ` (${plotState.unit})` : ''), L, 13);
  const order = xs.map((x, i) => i).sort((a, b) => xs[a] - xs[b]);
  g.strokeStyle = '#29b6f6'; g.lineWidth = 1.6; g.beginPath();
  order.forEach((i, k) => { const px = X(xs[i]), py = Y(ys[i]); if (k) g.lineTo(px, py); else g.moveTo(px, py); });
  g.stroke();
  g.setLineDash([4, 4]); g.strokeStyle = 'rgba(41,182,246,0.45)'; g.beginPath();  // kept flat outside the knots
  const first = order[0], last = order[order.length - 1];
  g.moveTo(L, Y(ys[first])); g.lineTo(X(xs[first]), Y(ys[first])); g.moveTo(X(xs[last]), Y(ys[last])); g.lineTo(W - R, Y(ys[last]));
  g.stroke(); g.setLineDash([]);
  g.fillStyle = '#29b6f6'; if (xs.length < 120) order.forEach((i) => { g.beginPath(); g.arc(X(xs[i]), Y(ys[i]), 2, 0, 6.3); g.fill(); });
  c.onmousemove = (ev) => {
    const r = c.getBoundingClientRect(); const mx = ev.clientX - r.left;
    let best = order[0]; order.forEach((i) => { if (Math.abs(X(xs[i]) - mx) < Math.abs(X(xs[best]) - mx)) best = i; });
    const tip = $('plotTip'); tip.style.display = 'block';
    tip.textContent = `${fmt(xs[best], 5)} ${energyX ? 'eV' : 'nm'}: ${fmt(ys[best], 5)}${plotState.unit ? ' ' + plotState.unit : ''}`;
    tip.style.left = Math.min(W - 150, X(xs[best]) + 8) + 'px'; tip.style.top = (Y(ys[best]) - 22) + 'px';
  };
  c.onmouseleave = () => { $('plotTip').style.display = 'none'; };
}

function renderLegend() {
  const el = $('legend'); let items = []; let title = '';
  if (colorMode === 'role') {
    title = 'HOW LUMEN READS IT';
    const present = new Set(S.volumes.map((v, i) => (lvObjects[i] && lvObjects[i].mesh.count ? lvRole(v) : null)));
    items = Object.entries(ROLE).filter(([k]) => present.has(k)).map(([k, [c, t]]) => [c, t]);
  } else if (colorMode === 'material') {
    title = 'MATERIAL';
    const present = new Set(S.volumes.map((v, i) => (lvObjects[i] && lvObjects[i].mesh.count ? v.material : null)));
    items = Object.keys(S.materials).filter((m) => present.has(m)).map((m) => [matColors[m], S.materials[m].short]);
  } else if (colorMode === 'surface') {
    title = 'SURFACE ON THE VOLUME';
    const present = new Set(); S.volumes.forEach((v, i) => { if (lvObjects[i] && lvObjects[i].mesh.count) lvSurfaces(v).slice(0, 1).forEach((s) => present.add(s)); });
    items = [...present].map((s) => [surfaceColor(s), `${S.surfaces[s].short}: ${surfaceLabel(S.surfaces[s])}`]);
    items.push(['#3a4247', 'no optical surface']);
  } else {
    title = 'DEPTH'; items = [0, 0.25, 0.5, 0.75, 1].map((t) => [viridis(t), String(Math.round(t * maxDepth))]);
  }
  el.innerHTML = `<div class="lt">${title}</div>` + items.map(([c, t]) => `<div class="li"><span class="sw" style="background:${c}"></span>${esc(t)}</div>`).join('');
}

function group(id, fn) {
  $(id).addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    [...$(id).children].forEach((x) => x.classList.toggle('active', x === b)); fn(b.dataset.v);
  });
}
function setActive(id, v) { [...$(id).children].forEach((x) => x.classList.toggle('active', x.dataset.v === v)); }

function wire() {
  group('colorGrp', (v) => { colorMode = v; refreshMaterials(); renderList(); });
  group('showGrp', (v) => { showMode = v; refreshInstances(); refreshMaterials(); renderList(); });
  group('upGrp', setUp);
  group('clipGrp', (v) => { clipAxis = v; updateClip(); refreshMaterials(); });
  $('clipPos').addEventListener('input', updateClip);
  $('clipFlip').addEventListener('click', () => { clipFlip = !clipFlip; updateClip(); });
  $('opacity').addEventListener('input', (e) => { mediaOpacity = parseFloat(e.target.value); refreshMaterials(); });
  $('resetBtn').addEventListener('click', () => { select(null); fitCamera(); });
  $('settingsBtn').addEventListener('click', () => $('settingsPanel').classList.toggle('visible'));
  $('settingsClose').addEventListener('click', () => $('settingsPanel').classList.remove('visible'));
  $('edgesChk').addEventListener('change', (e) => { edgesOn = e.target.checked; if (selection && selection.kind === 'volume') showPicked(selection.gid, selection.name); });
  $('isolateChk').addEventListener('change', (e) => { isolate = e.target.checked; refreshMaterials(); });
  $('opaqueSolidChk').addEventListener('change', (e) => { opaqueSolid = e.target.checked; refreshMaterials(); });
  $('solidOp').addEventListener('input', (e) => { solidOpacity = parseFloat(e.target.value); $('solidOpVal').textContent = solidOpacity.toFixed(2); refreshMaterials(); });
  $('depthMax').max = String(maxDepth); $('depthMax').value = String(maxDepth); depthMax = maxDepth; $('depthVal').textContent = 'all';
  $('depthMax').addEventListener('input', (e) => { depthMax = parseInt(e.target.value, 10); $('depthVal').textContent = depthMax >= maxDepth ? 'all' : depthMax; refreshInstances(); refreshMaterials(); });
  $('bulkMax').addEventListener('input', (e) => { bulkMax = parseInt(e.target.value, 10); $('bulkVal').textContent = bulkMax.toLocaleString(); refreshInstances(); refreshMaterials(); renderList(); });
  $('logYChk').addEventListener('change', (e) => { logY = e.target.checked; drawPlot(); });
  $('energyXChk').addEventListener('change', (e) => { energyX = e.target.checked; drawPlot(); });
  $('tabs').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return; tab = b.dataset.t;
    [...$('tabs').children].forEach((x) => x.classList.toggle('active', x === b)); renderList();
  });
  $('search').addEventListener('input', renderList);
  $('save3d').addEventListener('click', () => {
    const a = document.createElement('a'); a.download = `${S.meta.name || 'lumen'}_3d.png`; a.href = renderer.domElement.toDataURL('image/png'); a.click();
  });
  const onLink = (e) => {
    const t = e.target.closest('[data-lv],[data-mat],[data-surf],[data-group],[data-plot],[data-lvname]');
    if (!t || e.target.matches('input[type=checkbox]')) return;
    if (t.dataset.plot) { choosePlot(t.dataset.plot); return; }
    if (t.dataset.lv !== undefined) select({ kind: 'volume', name: +t.dataset.lv, gid: lvGids[+t.dataset.lv][0] });
    else if (t.dataset.lvname) { const i = S.volumes.findIndex((v) => v.name === t.dataset.lvname); if (i >= 0) select({ kind: 'volume', name: i, gid: lvGids[i][0] }); }
    else if (t.dataset.mat) select({ kind: 'material', name: t.dataset.mat });
    else if (t.dataset.surf) select({ kind: 'surface', name: t.dataset.surf });
    else if (t.dataset.group) select({ kind: 'group', name: t.dataset.group });
    plotState = null; renderDetails();
  };
  $('list').addEventListener('click', onLink);
  $('details').addEventListener('click', onLink);
  $('list').addEventListener('change', (e) => {
    const t = e.target; if (t.dataset.vis === undefined) return;
    lvVisible[+t.dataset.vis] = t.checked ? 'on' : 'off'; refreshInstances(); refreshMaterials(); renderList();
  });
  window.addEventListener('resize', drawPlot);
}

async function main() {
  try {
    await load();
  } catch (err) {
    $('loadStatus').textContent = 'could not load the scene: ' + err.message; console.error('[viewer]', err); return;
  }
  $('loading').style.display = 'none'; $('app').style.display = 'flex';
  $('detName').textContent = `${S.meta.name}${S.meta.recipe ? '' : ' · no recipe'}`;
  $('detName').title = `${S.meta.file}${S.meta.recipe ? ' read with the recipe ' + S.meta.recipe : ' read without a recipe'}`;
  document.title = `${S.meta.name} · lumen viewer`;
  setActive('showGrp', showMode);
  if (AXES[S.meta.up]) { upAxis = S.meta.up; setActive('upGrp', upAxis); }
  initThree(); wire(); buildObjects(); fitCamera(); renderList();
  const nRef = Object.values(S.materials).filter((m) => m.status === 'refused').length + Object.values(S.surfaces).filter((s) => s.status === 'refused').length;
  setStatus(`${S.meta.counts.instances.toLocaleString()} placed volumes · ${Object.keys(S.materials).length} materials · ${Object.keys(S.surfaces).length} surfaces${nRef ? ` · ${nRef} refused by lumen` : ''}${S.errors.length ? ' · import failed: see REPORT' : ''}`);
  if (S.errors.length || nRef) toast(`lumen refuses ${nRef} material(s)/surface(s)${S.errors.length ? ' and the import failed' : ''}: see REPORT`);
}
main();
