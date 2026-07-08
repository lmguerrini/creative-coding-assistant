// Public V8 golden artifact: GLSL kaleidoscope fragment shader.
// Boundary: shader code only; host integration must provide uniforms.

#ifdef GL_ES
precision highp float;
#endif

uniform vec2 u_resolution;
uniform float u_time;

const float PI = 3.14159265359;

mat2 rotate2d(float angle) {
  float s = sin(angle);
  float c = cos(angle);
  return mat2(c, -s, s, c);
}

vec3 palette(float t) {
  vec3 a = vec3(0.45, 0.42, 0.38);
  vec3 b = vec3(0.55, 0.46, 0.44);
  vec3 c = vec3(1.00, 0.86, 0.68);
  vec3 d = vec3(0.08, 0.23, 0.41);
  return a + b * cos(6.28318 * (c * t + d));
}

void main() {
  vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / min(u_resolution.x, u_resolution.y);
  uv *= rotate2d(0.08 * u_time);

  float angle = atan(uv.y, uv.x);
  float radius = length(uv);
  float sectors = 10.0;
  float folded = abs(mod(angle + PI / sectors, (2.0 * PI) / sectors) - PI / sectors);

  vec2 kaleido = vec2(cos(folded), sin(folded)) * radius;
  float ripple = sin(18.0 * kaleido.x - u_time * 1.4) + cos(16.0 * kaleido.y + u_time * 1.1);
  float rings = sin(34.0 * radius - u_time * 2.0);
  float field = smoothstep(0.18, 0.96, 0.5 + 0.25 * ripple + 0.22 * rings);

  vec3 color = palette(field + radius * 0.38 + u_time * 0.035);
  color += vec3(0.05, 0.26, 0.34) / max(0.28, radius + 0.15);
  color *= 1.0 - smoothstep(0.92, 1.42, radius);

  gl_FragColor = vec4(color, 1.0);
}
