// Public V8 golden artifact: Three.js audio-reactive scene module.
// Boundary: requires a page that provides THREE and a user gesture for audio.

export function createAudioReactiveScene({ THREE, canvas, audioElement }) {
  if (!THREE) {
    throw new Error("THREE is required.");
  }

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x08121c);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  camera.position.set(0, 1.25, 7);

  const geometry = new THREE.IcosahedronGeometry(1.4, 4);
  const material = new THREE.MeshStandardMaterial({
    color: 0x42f5d7,
    emissive: 0x123a4a,
    roughness: 0.38,
    metalness: 0.18,
  });
  const core = new THREE.Mesh(geometry, material);
  scene.add(core);

  const ringMaterial = new THREE.MeshBasicMaterial({
    color: 0xf5c542,
    transparent: true,
    opacity: 0.34,
    wireframe: true,
  });
  const rings = new THREE.Group();
  for (let i = 0; i < 5; i += 1) {
    const torus = new THREE.Mesh(new THREE.TorusGeometry(2 + i * 0.26, 0.009, 8, 160), ringMaterial);
    torus.rotation.x = Math.PI / 2 + i * 0.12;
    torus.rotation.y = i * 0.41;
    rings.add(torus);
  }
  scene.add(rings);

  const key = new THREE.PointLight(0x80fff1, 2.2, 18);
  key.position.set(2.5, 2.5, 4);
  scene.add(key);
  scene.add(new THREE.AmbientLight(0x445566, 0.48));

  let analyser = null;
  let frequencyData = null;

  async function activateAudio() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass || !audioElement) {
      return false;
    }

    const audioContext = new AudioContextClass();
    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }

    const source = audioContext.createMediaElementSource(audioElement);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    frequencyData = new Uint8Array(analyser.frequencyBinCount);
    source.connect(analyser);
    analyser.connect(audioContext.destination);
    return true;
  }

  function resize(width, height) {
    const safeWidth = Math.max(1, width);
    const safeHeight = Math.max(1, height);
    renderer.setSize(safeWidth, safeHeight, false);
    camera.aspect = safeWidth / safeHeight;
    camera.updateProjectionMatrix();
  }

  function frame(now = 0) {
    const seconds = now * 0.001;
    let energy = 0.18 + Math.sin(seconds * 0.8) * 0.08;

    if (analyser && frequencyData) {
      analyser.getByteFrequencyData(frequencyData);
      const sum = frequencyData.reduce((total, value) => total + value, 0);
      energy = sum / frequencyData.length / 255;
    }

    core.rotation.x = seconds * 0.18;
    core.rotation.y = seconds * 0.31;
    core.scale.setScalar(1 + energy * 0.32);
    material.emissiveIntensity = 0.45 + energy * 1.8;
    key.intensity = 1.4 + energy * 3.2;
    rings.rotation.z = seconds * 0.09;
    rings.children.forEach((ring, index) => {
      ring.rotation.z = seconds * (0.08 + index * 0.018);
      ring.scale.setScalar(1 + energy * (0.04 + index * 0.012));
    });

    renderer.render(scene, camera);
  }

  return {
    activateAudio,
    frame,
    resize,
    scene,
    camera,
    renderer,
  };
}
