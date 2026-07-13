const kineticOrbitSculptureSource = [
  "const scene = new THREE.Scene();",
  "scene.background = new THREE.Color(0x030712);",
  "const cameraRig = new THREE.Group();",
  "cameraRig.position.set(0, 0.25, 0);",
  "scene.add(cameraRig);",
  "const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);",
  "camera.position.set(0, 1.15, 8.2);",
  "cameraRig.add(camera);",
  "const sculptureRig = new THREE.Group();",
  "sculptureRig.position.set(0, 0.15, 0);",
  "sculptureRig.rotation.z = 0.16;",
  "scene.add(sculptureRig);",
  "const sculpture = new THREE.Mesh(",
  "  new THREE.TorusKnotGeometry(1.35, 0.34, 180, 28),",
  "  new THREE.MeshStandardMaterial({ color: 0xffc857, emissive: 0x5c2508, metalness: 0.82, roughness: 0.24 })",
  ");",
  "sculpture.scale.set(1.05, 1.05, 1.05);",
  "sculptureRig.add(sculpture);",
  "const orbitRig = new THREE.Group();",
  "orbitRig.rotation.x = 0.68;",
  "sculptureRig.add(orbitRig);",
  "const orbitMaterial = new THREE.MeshBasicMaterial({ color: 0x70b7ff, transparent: true, opacity: 0.78 });",
  "const rings = [2.1, 2.55, 3.0].map((radius, index) => {",
  "  const ring = new THREE.Mesh(new THREE.TorusGeometry(radius, 0.025 + index * 0.008, 10, 128), orbitMaterial);",
  "  ring.rotation.x = index * 0.52;",
  "  ring.rotation.y = index * 0.37;",
  "  orbitRig.add(ring);",
  "  return ring;",
  "});",
  "const ambient = new THREE.AmbientLight(0x16213f, 1.8);",
  "const key = new THREE.PointLight(0xffd27a, 75, 24);",
  "key.position.set(3.8, 4.6, 5.2);",
  "const rim = new THREE.PointLight(0x3f8cff, 95, 24);",
  "rim.position.set(-4.5, 1.8, 2.4);",
  "scene.add(ambient, key, rim);",
  "const renderer = new THREE.WebGLRenderer({ antialias: true });",
  "renderer.setSize(width, height, false);",
  "renderer.outputColorSpace = THREE.SRGBColorSpace;",
  "renderer.toneMapping = THREE.ACESFilmicToneMapping;",
  "renderer.toneMappingExposure = 1.15;",
  "function animate(time) {",
  "  const seconds = time * 0.001;",
  "  sculpture.rotation.x = seconds * 0.18;",
  "  sculpture.rotation.y = seconds * 0.31;",
  "  orbitRig.rotation.z = seconds * -0.12;",
  "  rings.forEach((ring, index) => { ring.rotation.y += 0.0015 * (index + 1); });",
  "  cameraRig.rotation.y = Math.sin(seconds * 0.22) * 0.2;",
  "  camera.position.y = 1.15 + Math.sin(seconds * 0.35) * 0.24;",
  "  camera.lookAt(0, 0.15, 0);",
  "  renderer.render(scene, camera);",
  "  requestAnimationFrame(animate);",
  "}",
  "requestAnimationFrame(animate);"
].join("\n");

const polyrhythmicConstellationSource = [
  "const bell = new Tone.FMSynth().toDestination();",
  "const pulse = new Tone.MembraneSynth().toDestination();",
  "new Tone.Sequence((time, note) => bell.triggerAttackRelease(note, '8n', time), ['C4', 'G4', 'E4', 'B4'], '8n').start(0);",
  "new Tone.Sequence((time, note) => pulse.triggerAttackRelease(note, '16n', time), ['C2', 'C2', 'G2', 'D2'], '16n').start(0);",
  "Tone.Transport.bpm.value = 108;",
  "Tone.Transport.start();"
].join("\n");

const recursiveAuroraGardenSource = [
  "let phase = 0;",
  "function setup() {",
  "  createCanvas(windowWidth, windowHeight);",
  "  pixelDensity(1);",
  "  colorMode(HSL, 360, 100, 100, 1);",
  "  noStroke();",
  "}",
  "function draw() {",
  "  phase += 0.008;",
  "  background(232, 54, 6, 0.2);",
  "  translate(width * 0.5, height * 0.52);",
  "  for (let seed = 0; seed < 160; seed += 1) {",
  "    const goldenAngle = seed * 2.399963;",
  "    const radius = 7.2 * sqrt(seed);",
  "    const breath = 1 + 0.08 * sin(phase * 2 + seed * 0.11);",
  "    const x = cos(goldenAngle + phase) * radius * breath + (mouseX - width * 0.5) * 0.025;",
  "    const y = sin(goldenAngle + phase) * radius * 0.62 * breath + (mouseY - height * 0.5) * 0.018;",
  "    fill((178 + seed * 0.72) % 360, 82, 64, 0.54);",
  "    circle(x, y, 4 + 5 * sin(seed * 0.17 + phase));",
  "  }",
  "}"
].join("\n");

const fractalSolarBloomSource = [
  "uniform float u_time;",
  "uniform vec2 u_resolution;",
  "void main() {",
  "  vec2 uv = (2.0 * gl_FragCoord.xy - u_resolution.xy) / min(u_resolution.x, u_resolution.y);",
  "  float angle = atan(uv.y, uv.x);",
  "  float radius = length(uv);",
  "  float fold = abs(sin(angle * 8.0 + sin(angle * 3.0) * 1.8));",
  "  float petals = exp(-5.0 * abs(radius - 0.32 - fold * 0.13));",
  "  float corona = 0.025 / max(abs(radius - 0.62 + 0.05 * sin(angle * 16.0 + u_time)), 0.012);",
  "  vec3 indigo = vec3(0.015, 0.02, 0.11);",
  "  vec3 cyan = vec3(0.08, 0.82, 1.0);",
  "  vec3 gold = vec3(1.0, 0.57, 0.12);",
  "  vec3 color = indigo + cyan * petals + gold * corona * (0.32 + 0.18 * sin(u_time));",
  "  gl_FragColor = vec4(color, 1.0);",
  "}"
].join("\n");

const showcaseSmokeCases = [
  {
    id: "polyrhythmic-constellation",
    title: "Polyrhythmic constellation",
    artifact: {
      id: "showcase-polyrhythmic-constellation",
      title: "polyrhythmic-constellation.tone.js",
      type: "code",
      language: "javascript + tone.js",
      domain: "tone_js",
      runtime: "tone",
      renderer_id: "surface.tone",
      preview_eligible: true,
      preview_target: "browser_sandbox",
      status: "Generated",
      summary:
        "Local deterministic showcase fixture for browser smoke; this is not provider-backed generation evidence.",
      content: polyrhythmicConstellationSource
    },
    requestTokens: [
      "polyrhythmic-constellation.tone.js",
      "Tone.MembraneSynth",
      "Tone.Transport.bpm.value = 108"
    ],
    qualityTokens: [
      "Tone.FMSynth",
      "Tone.MembraneSynth",
      "'8n'",
      "'16n'",
      "Tone.Transport.bpm.value = 108"
    ],
    runtimeLabel: "Tone.js",
    expectedRuntimeState: "ready",
    followUp: "Slow the constellation tempo while preserving both complementary voices."
  },
  {
    id: "recursive-aurora-garden",
    title: "Recursive aurora garden",
    artifact: {
      id: "showcase-recursive-aurora-garden",
      title: "recursive-aurora-garden.p5.js",
      type: "code",
      language: "p5.js",
      domain: "p5_js",
      runtime: "p5",
      renderer_id: "surface.p5",
      preview_eligible: true,
      preview_target: "browser_sandbox",
      status: "Generated",
      summary:
        "Local deterministic showcase fixture for browser smoke; this is not provider-backed generation evidence.",
      content: recursiveAuroraGardenSource
    },
    requestTokens: [
      "recursive-aurora-garden.p5.js",
      "160 golden-angle seeds",
      "pointer parallax"
    ],
    qualityTokens: [
      "goldenAngle",
      "seed < 160",
      "mouseX",
      "mouseY",
      "circle("
    ],
    runtimeLabel: "p5.js",
    expectedRuntimeState: "running",
    followUp: "Shift the aurora garden toward a colder cyan palette without removing pointer parallax."
  },
  {
    id: "kinetic-orbit-capstone",
    title: "Kinetic orbit sculpture",
    artifact: {
      id: "showcase-kinetic-orbit-capstone",
      title: "kinetic-orbit-capstone.three.js",
      type: "code",
      language: "javascript",
      domain: "three_js",
      runtime: "three",
      renderer_id: "surface.three",
      preview_eligible: true,
      preview_target: "browser_sandbox",
      status: "Generated",
      summary:
        "Local deterministic showcase fixture for browser smoke; this is not provider-backed generation evidence.",
      content: kineticOrbitSculptureSource
    },
    requestTokens: [
      "kinetic-orbit-capstone.three.js",
      "TorusKnotGeometry",
      "cameraRig"
    ],
    qualityTokens: [
      "new THREE.TorusKnotGeometry",
      "cameraRig.add(camera)",
      "sculptureRig.add(orbitRig)",
      "new THREE.PointLight",
      "camera.lookAt"
    ],
    runtimeLabel: "Three.js",
    expectedRuntimeState: "running",
    followUp: "Slow the orbit motion while preserving the nested camera and sculpture rigs."
  },
  {
    id: "fractal-solar-bloom",
    title: "Fractal solar bloom",
    artifact: {
      id: "showcase-fractal-solar-bloom",
      title: "fractal-solar-bloom.frag",
      type: "code",
      language: "glsl",
      domain: "glsl",
      runtime: "glsl",
      renderer_id: "surface.glsl",
      preview_eligible: true,
      preview_target: "browser_sandbox",
      status: "Generated",
      summary:
        "Local deterministic showcase fixture for browser smoke; this is not provider-backed generation evidence.",
      content: fractalSolarBloomSource
    },
    requestTokens: [
      "fractal-solar-bloom.frag",
      "analytic polar folds",
      "gl_FragColor"
    ],
    qualityTokens: [
      "uniform float u_time",
      "uniform vec2 u_resolution",
      "atan(uv.y, uv.x)",
      "sin(angle * 8.0",
      "gl_FragColor"
    ],
    runtimeLabel: "GLSL",
    expectedRuntimeState: "running",
    followUp: "Increase cyan and gold contrast while preserving the bounded WebGL 1 source contract."
  }
];

function getShowcaseSmokeCase(id) {
  return showcaseSmokeCases.find((showcase) => showcase.id === id) ?? null;
}

module.exports = {
  getShowcaseSmokeCase,
  kineticOrbitSculptureSource,
  showcaseSmokeCases
};
