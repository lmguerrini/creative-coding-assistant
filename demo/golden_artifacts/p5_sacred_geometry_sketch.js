// Public V8 golden artifact: p5.js sacred geometry sketch.
// Boundary: aesthetic/symbolic composition only, not authority or truth claim.

let rotationPhase = 0;
let ringCount = 7;
let pulseAmount = 0.35;

function setup() {
  createCanvas(720, 720);
  colorMode(HSB, 360, 100, 100, 1);
  angleMode(RADIANS);
  noFill();
}

function draw() {
  background(220, 18, 8, 1);
  translate(width / 2, height / 2);

  const t = frameCount * 0.012;
  const baseRadius = min(width, height) * 0.075;
  rotationPhase += 0.003;

  drawLuminousGrid(baseRadius, t);
  drawOrbitalNodes(baseRadius * 2.4, t);
  drawBreathingMandala(baseRadius * 3.8, t);
}

function drawLuminousGrid(radius, t) {
  strokeWeight(1.4);
  for (let ring = 1; ring <= ringCount; ring += 1) {
    const hue = (185 + ring * 18 + sin(t + ring) * 18) % 360;
    stroke(hue, 72, 94, 0.58);
    circle(0, 0, radius * ring * (1 + sin(t * 1.7 + ring) * 0.018));
  }

  for (let spoke = 0; spoke < 24; spoke += 1) {
    const angle = (TWO_PI * spoke) / 24 + rotationPhase;
    const inner = radius * 0.9;
    const outer = radius * 7.25;
    stroke(42, 64, 96, 0.28);
    line(cos(angle) * inner, sin(angle) * inner, cos(angle) * outer, sin(angle) * outer);
  }
}

function drawOrbitalNodes(radius, t) {
  for (let orbit = 0; orbit < 3; orbit += 1) {
    const nodeCount = 6 + orbit * 6;
    const orbitRadius = radius + orbit * 58;
    for (let index = 0; index < nodeCount; index += 1) {
      const angle = (TWO_PI * index) / nodeCount + t * (0.18 + orbit * 0.05);
      const x = cos(angle) * orbitRadius;
      const y = sin(angle) * orbitRadius;
      const size = 9 + sin(t * 2 + index) * 3;
      stroke((300 + index * 8) % 360, 62, 98, 0.48);
      circle(x, y, size);
    }
  }
}

function drawBreathingMandala(radius, t) {
  beginShape();
  strokeWeight(2.2);
  stroke(168, 78, 100, 0.78);
  for (let i = 0; i <= 360; i += 1) {
    const angle = radians(i);
    const sector = sin(angle * 6 + t) * 0.5 + sin(angle * 12 - t * 0.7) * 0.25;
    const r = radius * (1 + sector * pulseAmount * 0.16);
    vertex(cos(angle) * r, sin(angle) * r);
  }
  endShape(CLOSE);
}

function mousePressed() {
  ringCount = ringCount === 7 ? 9 : 7;
  pulseAmount = pulseAmount === 0.35 ? 0.52 : 0.35;
}
