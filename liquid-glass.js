// Liquid Glass WebGL2 Engine
// Extracted from liquid-glass-studio by iyinchao
// 4-pass pipeline: bgPass -> vBlurPass -> hBlurPass -> mainPass

// ============ GLSL Shaders ============

const VERTEX_SHADER = `#version 300 es
in vec4 a_position;
out vec2 v_uv;
void main() {
  v_uv = (a_position.xy + 1.0) * 0.5;
  gl_Position = a_position;
}`;

const FRAGMENT_BG = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;
uniform vec2 u_resolution;
uniform float u_dpr;
uniform vec2 u_mouseSpring;
uniform float u_mergeRate;
uniform float u_shapeWidth;
uniform float u_shapeHeight;
uniform float u_shapeRadius;
uniform float u_shapeRoundness;
uniform float u_shadowExpand;
uniform float u_shadowFactor;
uniform vec2 u_shadowPosition;
uniform int u_bgType;
uniform sampler2D u_bgTexture;
uniform float u_bgTextureRatio;
uniform int u_bgTextureReady;
uniform int u_showShape1;

float sdCircle(vec2 p, float r) { return length(p) - r; }

float superellipseCornerSDF(vec2 p, float r, float n) {
  p = abs(p);
  return pow(pow(p.x, n) + pow(p.y, n), 1.0 / n) - r;
}

float roundedRectSDF(vec2 p, vec2 center, float width, float height, float cornerRadius, float n) {
  p -= center;
  float cr = cornerRadius * u_dpr;
  vec2 d = abs(p) - vec2(width * u_dpr, height * u_dpr) * 0.5;
  float dist;
  if (d.x > -cr && d.y > -cr) {
    vec2 cornerCenter = sign(p) * (vec2(width * u_dpr, height * u_dpr) * 0.5 - vec2(cr));
    vec2 cornerP = p - cornerCenter;
    dist = superellipseCornerSDF(cornerP, cr, n);
  } else {
    dist = min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
  }
  return dist;
}

float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

float mainSDF(vec2 p1, vec2 p2, vec2 p) {
  vec2 p1n = p1 + p / u_resolution.y;
  vec2 p2n = p2 + p / u_resolution.y;
  float d1 = u_showShape1 == 1 ? sdCircle(p1n, 100.0 * u_dpr / u_resolution.y) : 1.0;
  float d2 = roundedRectSDF(p2n, vec2(0.0), u_shapeWidth / u_resolution.y, u_shapeHeight / u_resolution.y, u_shapeRadius / u_resolution.y, u_shapeRoundness);
  return smin(d1, d2, u_mergeRate);
}

vec2 getCoverUV(vec2 uv, float canvasAspect, float textureAspect) {
  if (canvasAspect > textureAspect) {
    float scale = textureAspect / canvasAspect;
    uv.y = uv.y * scale + 0.5 - 0.5 * scale;
  } else {
    float scale = canvasAspect / textureAspect;
    uv.x = uv.x * scale + 0.5 - 0.5 * scale;
  }
  return uv;
}

void main() {
  vec2 u_resolution1x = u_resolution.xy / u_dpr;
  vec3 bgColor = vec3(1.0);

  if (u_bgType <= 2) {
    // Solid or simple bg - use texture if ready
    if (u_bgTextureReady == 1) {
      vec2 uv = getCoverUV(v_uv, u_resolution.x / u_resolution.y, u_bgTextureRatio);
      bgColor = texture(u_bgTexture, uv).rgb;
    }
  } else {
    if (u_bgTextureReady == 1) {
      vec2 uv = getCoverUV(v_uv, u_resolution.x / u_resolution.y, u_bgTextureRatio);
      bgColor = texture(u_bgTexture, uv).rgb;
    }
  }

  // Shadow
  vec2 p1 = (vec2(0, 0) - u_resolution.xy * 0.5 + vec2(u_shadowPosition.x * u_dpr, u_shadowPosition.y * u_dpr)) / u_resolution.y;
  vec2 p2 = (vec2(0, 0) - u_mouseSpring + vec2(u_shadowPosition.x * u_dpr, u_shadowPosition.y * u_dpr)) / u_resolution.y;
  float merged = mainSDF(p1, p2, gl_FragCoord.xy);
  float shadow = exp(-1.0 / u_shadowExpand * abs(merged) * u_resolution1x.y) * 0.6 * u_shadowFactor;

  fragColor = vec4(bgColor - vec3(shadow), 1.0);
}`;

const FRAGMENT_VBLUR = `#version 300 es
precision highp float;
#define MAX_BLUR_RADIUS (200)
in vec2 v_uv;
uniform sampler2D u_prevPassTexture;
uniform vec2 u_resolution;
uniform int u_blurRadius;
uniform float u_blurWeights[MAX_BLUR_RADIUS + 1];
out vec4 fragColor;
void main() {
  vec2 texelSize = 1.0 / u_resolution;
  vec4 color = texture(u_prevPassTexture, v_uv) * u_blurWeights[0];
  for (int i = 1; i <= u_blurRadius; ++i) {
    float w = u_blurWeights[i];
    vec2 offset = vec2(float(i)) * texelSize;
    color += texture(u_prevPassTexture, v_uv + vec2(offset.x, 0.0)) * w;
    color += texture(u_prevPassTexture, v_uv - vec2(offset.x, 0.0)) * w;
  }
  fragColor = color;
}`;

const FRAGMENT_HBLUR = `#version 300 es
precision highp float;
#define MAX_BLUR_RADIUS (200)
in vec2 v_uv;
uniform sampler2D u_prevPassTexture;
uniform vec2 u_resolution;
uniform int u_blurRadius;
uniform float u_blurWeights[MAX_BLUR_RADIUS + 1];
out vec4 fragColor;
void main() {
  vec2 texelSize = 1.0 / u_resolution;
  vec4 color = texture(u_prevPassTexture, v_uv) * u_blurWeights[0];
  for (int i = 1; i <= u_blurRadius; ++i) {
    float w = u_blurWeights[i];
    vec2 offset = vec2(float(i)) * texelSize;
    color += texture(u_prevPassTexture, v_uv + vec2(0.0, offset.y)) * w;
    color += texture(u_prevPassTexture, v_uv - vec2(0.0, offset.y)) * w;
  }
  fragColor = color;
}`;

const FRAGMENT_MAIN = `#version 300 es
precision highp float;
#define PI (3.14159265359)
const float N_R = 1.0 - 0.02;
const float N_G = 1.0;
const float N_B = 1.0 + 0.02;

in vec2 v_uv;
uniform sampler2D u_blurredBg;
uniform sampler2D u_bg;
uniform vec2 u_resolution;
uniform float u_dpr;
uniform vec2 u_mouseSpring;
uniform float u_mergeRate;
uniform float u_shapeWidth;
uniform float u_shapeHeight;
uniform float u_shapeRadius;
uniform float u_shapeRoundness;
uniform vec4 u_tint;
uniform float u_refThickness;
uniform float u_refFactor;
uniform float u_refDispersion;
uniform float u_refFresnelRange;
uniform float u_refFresnelFactor;
uniform float u_refFresnelHardness;
uniform float u_glareRange;
uniform float u_glareConvergence;
uniform float u_glareOppositeFactor;
uniform float u_glareFactor;
uniform float u_glareHardness;
uniform float u_glareAngle;
uniform int u_blurEdge;
uniform int u_showShape1;
uniform vec2 u_mouseCursor;

out vec4 fragColor;

float sdCircle(vec2 p, float r) { return length(p) - r; }

float superellipseCornerSDF(vec2 p, float r, float n) {
  p = abs(p);
  return pow(pow(p.x, n) + pow(p.y, n), 1.0 / n) - r;
}

float roundedRectSDF(vec2 p, vec2 center, float width, float height, float cornerRadius, float n) {
  p -= center;
  float cr = cornerRadius * u_dpr;
  vec2 d = abs(p) - vec2(width * u_dpr, height * u_dpr) * 0.5;
  float dist;
  if (d.x > -cr && d.y > -cr) {
    vec2 cornerCenter = sign(p) * (vec2(width * u_dpr, height * u_dpr) * 0.5 - vec2(cr));
    vec2 cornerP = p - cornerCenter;
    dist = superellipseCornerSDF(cornerP, cr, n);
  } else {
    dist = min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
  }
  return dist;
}

float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

float mainSDF(vec2 p1, vec2 p2, vec2 p) {
  vec2 p1n = p1 + p / u_resolution.y;
  vec2 p2n = p2 + p / u_resolution.y;
  float d1 = u_showShape1 == 1 ? sdCircle(p1n, 100.0 * u_dpr / u_resolution.y) : 1.0;
  float d2 = roundedRectSDF(p2n, vec2(0.0), u_shapeWidth / u_resolution.y, u_shapeHeight / u_resolution.y, u_shapeRadius / u_resolution.y, u_shapeRoundness);
  return smin(d1, d2, u_mergeRate);
}

vec2 getNormal(vec2 p1, vec2 p2, vec2 p) {
  vec2 h = vec2(max(abs(dFdx(p.x)), 0.0001), max(abs(dFdy(p.y)), 0.0001));
  vec2 grad = vec2(
    mainSDF(p1, p2, p + vec2(h.x, 0.0)) - mainSDF(p1, p2, p - vec2(h.x, 0.0)),
    mainSDF(p1, p2, p + vec2(0.0, h.y)) - mainSDF(p1, p2, p - vec2(0.0, h.y))
  ) / (2.0 * h);
  return grad * 1.414213562 * 1000.0;
}

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Color space conversions
const vec3 D65_WHITE = vec3(0.95045592705, 1.0, 1.08905775076);
vec3 WHITE = D65_WHITE;
const mat3 RGB_TO_XYZ_M = mat3(0.4124, 0.3576, 0.1805, 0.2126, 0.7152, 0.0722, 0.0193, 0.1192, 0.9505);
const mat3 XYZ_TO_RGB_M = mat3(3.2406255, -1.537208, -0.4986286, -0.9689307, 1.8757561, 0.0415175, 0.0557101, -0.2040211, 1.0569959);

float UNCOMPAND_SRGB(float a) { return a > 0.04045 ? pow((a + 0.055) / 1.055, 2.4) : a / 12.92; }
float COMPAND_RGB(float a) { return a <= 0.0031308 ? 12.92 * a : 1.055 * pow(a, 0.41666666666) - 0.055; }
vec3 RGB_TO_XYZ(vec3 rgb) { return rgb * RGB_TO_XYZ_M; }
vec3 SRGB_TO_RGB(vec3 srgb) { return vec3(UNCOMPAND_SRGB(srgb.x), UNCOMPAND_SRGB(srgb.y), UNCOMPAND_SRGB(srgb.z)); }
vec3 RGB_TO_SRGB(vec3 rgb) { return vec3(COMPAND_RGB(rgb.x), COMPAND_RGB(rgb.y), COMPAND_RGB(rgb.z)); }
vec3 SRGB_TO_XYZ(vec3 srgb) { return RGB_TO_XYZ(SRGB_TO_RGB(srgb)); }
float XYZ_TO_LAB_F(float x) { return x > 0.00885645167 ? pow(x, 0.333333333) : 7.78703703704 * x + 0.13793103448; }
vec3 XYZ_TO_LAB(vec3 xyz) {
  vec3 s = xyz / WHITE;
  s = vec3(XYZ_TO_LAB_F(s.x), XYZ_TO_LAB_F(s.y), XYZ_TO_LAB_F(s.z));
  return vec3(116.0 * s.y - 16.0, 500.0 * (s.x - s.y), 200.0 * (s.y - s.z));
}
vec3 SRGB_TO_LAB(vec3 srgb) { return XYZ_TO_LAB(SRGB_TO_XYZ(srgb)); }
vec3 LAB_TO_LCH(vec3 Lab) { return vec3(Lab.x, sqrt(dot(Lab.yz, Lab.yz)), atan(Lab.z, Lab.y) * 57.2957795131); }
vec3 SRGB_TO_LCH(vec3 srgb) { return LAB_TO_LCH(SRGB_TO_LAB(srgb)); }
vec3 XYZ_TO_RGB(vec3 xyz) { return xyz * XYZ_TO_RGB_M; }
vec3 XYZ_TO_SRGB(vec3 xyz) { return RGB_TO_SRGB(XYZ_TO_RGB(xyz)); }
float LAB_TO_XYZ_F(float x) { return x > 0.206897 ? x * x * x : 0.12841854934 * (x - 0.137931034); }
vec3 LAB_TO_XYZ(vec3 Lab) {
  float w = (Lab.x + 16.0) / 116.0;
  return WHITE * vec3(LAB_TO_XYZ_F(w + Lab.y / 500.0), LAB_TO_XYZ_F(w), LAB_TO_XYZ_F(w - Lab.z / 200.0));
}
vec3 LAB_TO_SRGB(vec3 lab) { return XYZ_TO_SRGB(LAB_TO_XYZ(lab)); }
vec3 LCH_TO_LAB(vec3 LCh) { return vec3(LCh.x, LCh.y * cos(LCh.z * 0.01745329251), LCh.y * sin(LCh.z * 0.01745329251)); }
vec3 LCH_TO_SRGB(vec3 lch) { return LAB_TO_SRGB(LCH_TO_LAB(lch)); }

float vec2ToAngle(vec2 v) {
  float angle = atan(v.y, v.x);
  if (angle < 0.0) angle += 2.0 * PI;
  return angle;
}

uniform float u_refBlur;

vec4 sampleDispersionAt(sampler2D tex1, sampler2D tex2, float mixRate, vec2 offset, float factor, vec2 jitter) {
  vec4 pixel = vec4(1.0);
  vec2 texelSize = 1.0 / u_resolution;
  vec2 j = jitter * texelSize * u_refBlur;
  float bgR = texture(tex1, v_uv + j + offset * (1.0 - (N_R - 1.0) * factor)).r;
  float bgG = texture(tex1, v_uv + j + offset * (1.0 - (N_G - 1.0) * factor)).g;
  float bgB = texture(tex1, v_uv + j + offset * (1.0 - (N_B - 1.0) * factor)).b;
  float blurR = texture(tex2, v_uv + j + offset * (1.0 - (N_R - 1.0) * factor)).r;
  float blurG = texture(tex2, v_uv + j + offset * (1.0 - (N_G - 1.0) * factor)).g;
  float blurB = texture(tex2, v_uv + j + offset * (1.0 - (N_B - 1.0) * factor)).b;
  pixel.r = mix(bgR, blurR, mixRate);
  pixel.g = mix(bgG, blurG, mixRate);
  pixel.b = mix(bgB, blurB, mixRate);
  return pixel;
}

vec4 getTextureDispersion(sampler2D tex1, sampler2D tex2, float mixRate, vec2 offset, float factor) {
  if (u_refBlur < 0.5) {
    return sampleDispersionAt(tex1, tex2, mixRate, offset, factor, vec2(0.0));
  }
  // Multi-sample blur: 9-tap 3x3 grid
  vec4 acc = vec4(0.0);
  acc += sampleDispersionAt(tex1, tex2, mixRate, offset, factor, vec2(-1.0, -1.0));
  acc += sampleDispersionAt(tex1, tex2, mixRate, offset, factor, vec2( 0.0, -1.0));
  acc += sampleDispersionAt(tex1, tex2, mixRate, offset, factor, vec2( 1.0, -1.0));
  acc += sampleDispersionAt(tex1, tex2, mixRate, offset, factor, vec2(-1.0,  0.0));
  acc += sampleDispersionAt(tex1, tex2, mixRate, offset, factor, vec2( 0.0,  0.0));
  acc += sampleDispersionAt(tex1, tex2, mixRate, offset, factor, vec2( 1.0,  0.0));
  acc += sampleDispersionAt(tex1, tex2, mixRate, offset, factor, vec2(-1.0,  1.0));
  acc += sampleDispersionAt(tex1, tex2, mixRate, offset, factor, vec2( 0.0,  1.0));
  acc += sampleDispersionAt(tex1, tex2, mixRate, offset, factor, vec2( 1.0,  1.0));
  return acc / 9.0;
}

void main() {
  vec2 u_resolution1x = u_resolution.xy / u_dpr;
  vec2 p1 = (vec2(0, 0) - u_mouseCursor) / u_resolution.y;
  vec2 p2 = (vec2(0, 0) - u_mouseSpring) / u_resolution.y;
  float merged = mainSDF(p1, p2, gl_FragCoord.xy);

  vec4 outColor;

  // Production output (STEP=9 equivalent)
  if (merged < 0.005) {
    float nmerged = -1.0 * (merged * u_resolution1x.y);
    float x_R_ratio = 1.0 - nmerged / u_refThickness;
    float thetaI = asin(pow(x_R_ratio, 2.0));
    float thetaT = asin(1.0 / u_refFactor * sin(thetaI));
    float edgeFactor = -1.0 * tan(thetaT - thetaI);
    if (nmerged >= u_refThickness) { edgeFactor = 0.0; }

    if (edgeFactor <= 0.0) {
      outColor = texture(u_blurredBg, v_uv);
      outColor = mix(outColor, vec4(u_tint.r, u_tint.g, u_tint.b, 1.0), u_tint.a * 0.8);
    } else {
      float edgeH = nmerged / u_refThickness;
      vec2 normal = getNormal(p1, p2, gl_FragCoord.xy);
      vec4 blurredPixel = getTextureDispersion(
        u_bg, u_blurredBg,
        u_blurEdge > 0 ? 1.0 : edgeH,
        -normal * edgeFactor * 0.05 * u_dpr * vec2(u_resolution.y / (u_resolution1x.x * u_dpr), 1.0),
        u_refDispersion
      );

      outColor = mix(blurredPixel, vec4(u_tint.r, u_tint.g, u_tint.b, 1.0), u_tint.a * 0.8);

      // Fresnel
      float fresnelFactor = clamp(pow(1.0 + merged * u_resolution1x.y / 1500.0 * pow(500.0 / u_refFresnelRange, 2.0) + u_refFresnelHardness, 5.0), 0.0, 1.0);
      vec3 fresnelTintLCH = SRGB_TO_LCH(mix(vec3(1.0), vec3(u_tint.r, u_tint.g, u_tint.b), u_tint.a * 0.5));
      fresnelTintLCH.x += 20.0 * fresnelFactor * u_refFresnelFactor;
      fresnelTintLCH.x = clamp(fresnelTintLCH.x, 0.0, 100.0);
      outColor = mix(outColor, vec4(LCH_TO_SRGB(fresnelTintLCH), 1.0), fresnelFactor * u_refFresnelFactor * 0.7 * length(normal));

      // Glare
      float glareGeoFactor = clamp(pow(1.0 + merged * u_resolution1x.y / 1500.0 * pow(500.0 / u_glareRange, 2.0) + u_glareHardness, 5.0), 0.0, 1.0);
      float glareAngle = (vec2ToAngle(normalize(normal)) - PI / 4.0 + u_glareAngle) * 2.0;
      int glareFarside = 0;
      if (glareAngle > PI * (2.0 - 0.5) && glareAngle < PI * (4.0 - 0.5) || glareAngle < PI * (0.0 - 0.5)) { glareFarside = 1; }
      float glareAngleFactor = (0.5 + sin(glareAngle) * 0.5) * (glareFarside == 1 ? 1.2 * u_glareOppositeFactor : 1.2) * u_glareFactor;
      glareAngleFactor = clamp(pow(glareAngleFactor, 0.1 + u_glareConvergence * 2.0), 0.0, 1.0);

      vec3 glareTintLCH = SRGB_TO_LCH(mix(blurredPixel.rgb, vec3(u_tint.r, u_tint.g, u_tint.b), u_tint.a * 0.5));
      glareTintLCH.x += 150.0 * glareAngleFactor * glareGeoFactor;
      glareTintLCH.y += 30.0 * glareAngleFactor * glareGeoFactor;
      glareTintLCH.x = clamp(glareTintLCH.x, 0.0, 120.0);
      outColor = mix(outColor, vec4(LCH_TO_SRGB(glareTintLCH), 1.0), glareAngleFactor * glareGeoFactor * length(normal));
    }
  } else {
    outColor = texture(u_bg, v_uv);
  }

  // Smooth edge
  outColor = mix(outColor, texture(u_bg, v_uv), smoothstep(-0.001, 0.001, merged));
  fragColor = outColor;
}`;

// ============ Utility Functions ============

function computeGaussianKernel(radius) {
  const sigma = radius / 3.0;
  const kernel = [];
  let sum = 0;
  for (let i = 0; i <= radius; i++) {
    const weight = Math.exp(-0.5 * (i * i) / (sigma * sigma));
    kernel.push(weight);
    sum += i === 0 ? weight : weight * 2;
  }
  return kernel.map(w => w / sum);
}

// ============ GL Classes ============

class ShaderProgram {
  constructor(gl, source) {
    this.gl = gl;
    this.uniforms = new Map();
    this.attributes = new Map();
    this.program = this._createProgram(source);
    this._detectAttributes();
    this._detectUniforms();
  }

  _createShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error('Shader compile error: ' + info);
    }
    return shader;
  }

  _createProgram(source) {
    const gl = this.gl;
    const program = gl.createProgram();
    const vs = this._createShader(gl.VERTEX_SHADER, source.vertex);
    const fs = this._createShader(gl.FRAGMENT_SHADER, source.fragment);
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error('Program link error: ' + info);
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return program;
  }

  _detectAttributes() {
    const gl = this.gl;
    const n = gl.getProgramParameter(this.program, gl.ACTIVE_ATTRIBUTES);
    for (let i = 0; i < n; i++) {
      const info = gl.getActiveAttrib(this.program, i);
      if (!info) continue;
      this.attributes.set(info.name, {
        location: gl.getAttribLocation(this.program, info.name),
        size: info.size, type: info.type,
      });
    }
  }

  _detectUniforms() {
    const gl = this.gl;
    const n = gl.getProgramParameter(this.program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const info = gl.getActiveUniform(this.program, i);
      if (!info) continue;
      const loc = gl.getUniformLocation(this.program, info.name);
      if (!loc) continue;
      const arrayRegex = /\[\d+\]$/;
      if (arrayRegex.test(info.name)) {
        const baseName = info.name.replace(arrayRegex, '');
        this.uniforms.set(baseName, { location: loc, type: info.type, isArray: { size: info.size } });
      } else {
        this.uniforms.set(info.name, { location: loc, type: info.type, isArray: false });
      }
    }
  }

  use() { this.gl.useProgram(this.program); }

  setUniform(name, value) {
    const gl = this.gl;
    const u = this.uniforms.get(name);
    if (!u) return;
    if (u.isArray && Array.isArray(value)) {
      switch (u.type) {
        case gl.FLOAT: gl.uniform1fv(u.location, value); break;
        case gl.FLOAT_VEC2: gl.uniform2fv(u.location, value); break;
        case gl.FLOAT_VEC3: gl.uniform3fv(u.location, value); break;
        case gl.FLOAT_VEC4: gl.uniform4fv(u.location, value); break;
      }
    } else {
      switch (u.type) {
        case gl.FLOAT: gl.uniform1f(u.location, value); break;
        case gl.FLOAT_VEC2: gl.uniform2fv(u.location, value); break;
        case gl.FLOAT_VEC3: gl.uniform3fv(u.location, value); break;
        case gl.FLOAT_VEC4: gl.uniform4fv(u.location, value); break;
        case gl.INT: gl.uniform1i(u.location, value); break;
        case gl.SAMPLER_2D: gl.uniform1i(u.location, value); break;
        case gl.FLOAT_MAT3: gl.uniformMatrix3fv(u.location, false, value); break;
        case gl.FLOAT_MAT4: gl.uniformMatrix4fv(u.location, false, value); break;
      }
    }
  }

  getAttributeLocation(name) {
    const a = this.attributes.get(name);
    return a ? a.location : -1;
  }
}

class FrameBuffer {
  constructor(gl, width, height) {
    this.gl = gl;
    this.width = width;
    this.height = height;
    this._create();
  }

  _create() {
    const gl = this.gl;
    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);

    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, this.width, this.height, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texture, 0);

    this.depthTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.depthTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, this.width, this.height, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.depthTexture, 0);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  bind() { this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbo); }
  unbind() { this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null); }
  getTexture() { return this.texture; }

  resize(w, h) {
    this.width = w; this.height = h;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.FLOAT, null);
    gl.bindTexture(gl.TEXTURE_2D, this.depthTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, w, h, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }
}

class RenderPass {
  constructor(gl, shaderSource, outputToScreen) {
    this.gl = gl;
    this.config = { name: '', shader: shaderSource };
    this.program = new ShaderProgram(gl, shaderSource);
    this.frameBuffer = !outputToScreen ? new FrameBuffer(gl, gl.canvas.width, gl.canvas.height) : null;
    this._createVAO();
  }

  _createVAO() {
    const gl = this.gl;
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
    const loc = this.program.getAttributeLocation('a_position');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  render(uniforms) {
    const gl = this.gl;
    if (this.frameBuffer) { this.frameBuffer.bind(); } else { gl.bindFramebuffer(gl.FRAMEBUFFER, null); }
    this.program.use();
    if (uniforms) {
      let texUnit = 0;
      for (const [name, value] of Object.entries(uniforms)) {
        if (value instanceof WebGLTexture) {
          gl.activeTexture(gl.TEXTURE0 + texUnit);
          gl.bindTexture(gl.TEXTURE_2D, value);
          this.program.setUniform(name, texUnit);
          texUnit++;
        } else {
          this.program.setUniform(name, value);
        }
      }
    }
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
    if (this.frameBuffer) { this.frameBuffer.unbind(); }
  }

  getOutputTexture() { return this.frameBuffer ? this.frameBuffer.getTexture() : null; }
  resize(w, h) { if (this.frameBuffer) this.frameBuffer.resize(w, h); }
}

// ============ Main LiquidGlass Class ============

class LiquidGlass {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { premultipliedAlpha: false, alpha: true });
    if (!gl) throw new Error('WebGL2 not supported');
    const ext = gl.getExtension('EXT_color_buffer_float');
    if (!ext) throw new Error('EXT_color_buffer_float not supported');
    this.gl = gl;

    // Create passes
    const v = VERTEX_SHADER;
    this.bgPass = new RenderPass(gl, { vertex: v, fragment: FRAGMENT_BG }, false);
    this.bgPass.config.name = 'bgPass';
    this.vBlurPass = new RenderPass(gl, { vertex: v, fragment: FRAGMENT_VBLUR }, false);
    this.vBlurPass.config.name = 'vBlurPass';
    this.hBlurPass = new RenderPass(gl, { vertex: v, fragment: FRAGMENT_HBLUR }, false);
    this.hBlurPass.config.name = 'hBlurPass';
    this.mainPass = new RenderPass(gl, { vertex: v, fragment: FRAGMENT_MAIN }, true);
    this.mainPass.config.name = 'mainPass';

    this.passes = [this.bgPass, this.vBlurPass, this.hBlurPass, this.mainPass];
    this.globalUniforms = {};
    this.bgTexture = null;
    this.bgTextureRatio = 1;
    this.bgTextureReady = false;

    // Default settings from user's JSON
    this.settings = {
      refThickness: 2000,
      refFactor: 3.0,
      refDispersion: 1.6,
      refFresnelRange: 48.1,
      refFresnelHardness: 28.51,
      refFresnelFactor: 38.65,
      glareRange: 54.18,
      glareHardness: 24.34,
      glareFactor: 0,
      glareConvergence: 56.36,
      glareOppositeFactor: 80,
      glareAngle: -42.09,
      blurRadius: 1,
      blurEdge: true,
      tint: { r: 255, g: 255, b: 255, a: 0 },
      shadowExpand: 25,
      shadowFactor: 5,
      shadowPosition: { x: 0, y: -10 },
      shapeWidth: 212,
      shapeHeight: 212,
      shapeRadius: 100,
      shapeRoundness: 2,
      refBlur: 0,
      mergeRate: 0.05,
      showShape1: false,
      ...options,
    };

    // Shape position (in screen pixels, center of canvas = default)
    this.shapeX = 0;
    this.shapeY = 0;

    // Mouse cursor tracking with spring
    this.mouseX = canvas.width / 2;
    this.mouseY = canvas.height / 2;
    this.mouseSpringX = canvas.width / 2;
    this.mouseSpringY = canvas.height / 2;
    this.mouseActive = false;
  }

  setSettings(s) {
    Object.assign(this.settings, s);
  }

  setShapePosition(x, y) {
    this.shapeX = x;
    this.shapeY = y;
  }

  setMousePosition(clientX, clientY) {
    const rect = this.gl.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.mouseX = (clientX - rect.left) * dpr;
    this.mouseY = (rect.height - (clientY - rect.top)) * dpr;
    this.mouseActive = true;
  }

  clearMouse() {
    this.mouseActive = false;
  }

  setShapeSize(w, h) {
    this.settings.shapeWidth = w;
    this.settings.shapeHeight = h;
  }

  setBgTexture(texture, ratio) {
    this.bgTexture = texture;
    this.bgTextureRatio = ratio;
    this.bgTextureReady = true;
  }

  // Render background from a canvas element (e.g. rasterized HTML content)
  setBgFromCanvas(sourceCanvas) {
    const gl = this.gl;
    if (!this.bgTexture) {
      this.bgTexture = gl.createTexture();
    }
    gl.bindTexture(gl.TEXTURE_2D, this.bgTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, sourceCanvas);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    this.bgTextureRatio = sourceCanvas.width / sourceCanvas.height;
    this.bgTextureReady = true;
  }

  resize(w, h) {
    this.canvas.width = w;
    this.canvas.height = h;
    this.gl.viewport(0, 0, w, h);
    this.passes.forEach(p => p.resize(w, h));
  }

  render() {
    const gl = this.gl;
    const s = this.settings;
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.width;
    const h = this.canvas.height;

    gl.viewport(0, 0, w, h);

    // Compute blur kernel
    const blurRadius = Math.max(1, Math.min(200, s.blurRadius));
    const kernel = computeGaussianKernel(blurRadius);
    // Pad to 201 length
    const weights = new Float32Array(201);
    for (let i = 0; i < kernel.length; i++) weights[i] = kernel[i];

    // Shape position: u_mouseSpring is in canvas pixel coords (center of shape 2)
    // Convert from our screen coordinates to gl coordinates
    const mouseSpringX = w / 2 - this.shapeX * dpr;
    const mouseSpringY = h / 2 + this.shapeY * dpr;

    // Mouse cursor spring animation (lerp toward target)
    const springFactor = 0.12;
    if (this.mouseActive) {
      this.mouseSpringX += (this.mouseX - this.mouseSpringX) * springFactor;
      this.mouseSpringY += (this.mouseY - this.mouseSpringY) * springFactor;
    } else {
      // When mouse leaves, spring back to center
      this.mouseSpringX += (w / 2 - this.mouseSpringX) * 0.05;
      this.mouseSpringY += (h / 2 - this.mouseSpringY) * 0.05;
    }

    // Compute actual shapeRadius: (min(w,h)/2 * slider%) / 100
    const minDim = Math.min(s.shapeWidth, s.shapeHeight);
    const actualShapeRadius = (minDim / 2 * s.shapeRadius) / 100;

    // Common uniforms
    const common = {
      u_resolution: [w, h],
      u_dpr: dpr,
      u_mouseSpring: [mouseSpringX, mouseSpringY],
      u_mouseCursor: [this.mouseSpringX, this.mouseSpringY],
      u_mergeRate: s.mergeRate,
      u_shapeWidth: s.shapeWidth,
      u_shapeHeight: s.shapeHeight,
      u_shapeRadius: actualShapeRadius,
      u_shapeRoundness: s.shapeRoundness,
      u_showShape1: s.showShape1 ? 1 : 0,
    };

    // 1. Background pass
    // NOTE: shadowFactor is /100, shadowPosition is negated per original App.tsx
    const bgUniforms = {
      ...common,
      u_shadowExpand: s.shadowExpand,
      u_shadowFactor: s.shadowFactor / 100,
      u_shadowPosition: [-s.shadowPosition.x, -s.shadowPosition.y],
      u_bgType: 5, // texture mode
      u_bgTextureReady: this.bgTextureReady ? 1 : 0,
      u_bgTextureRatio: this.bgTextureRatio,
    };
    if (this.bgTexture) bgUniforms.u_bgTexture = this.bgTexture;
    this.bgPass.render(bgUniforms);

    // 2. Vertical blur pass
    const bgOutputTex = this.bgPass.getOutputTexture();
    this.vBlurPass.render({
      u_prevPassTexture: bgOutputTex,
      u_resolution: [w, h],
      u_blurRadius: blurRadius,
      u_blurWeights: Array.from(weights),
    });

    // 3. Horizontal blur pass
    const vBlurTex = this.vBlurPass.getOutputTexture();
    this.hBlurPass.render({
      u_prevPassTexture: vBlurTex,
      u_resolution: [w, h],
      u_blurRadius: blurRadius,
      u_blurWeights: Array.from(weights),
    });

    // 4. Main composite pass
    // NOTE: several values are /100, glareAngle is in radians, tint alpha is raw (not /255)
    const hBlurTex = this.hBlurPass.getOutputTexture();
    const tint = s.tint;
    this.mainPass.render({
      ...common,
      u_bg: bgOutputTex,
      u_blurredBg: hBlurTex,
      u_tint: [tint.r / 255, tint.g / 255, tint.b / 255, tint.a],
      u_refThickness: s.refThickness,
      u_refFactor: s.refFactor,
      u_refDispersion: s.refDispersion,
      u_refBlur: s.refBlur,
      u_refFresnelRange: s.refFresnelRange,
      u_refFresnelFactor: s.refFresnelFactor / 100,
      u_refFresnelHardness: s.refFresnelHardness / 100,
      u_glareRange: s.glareRange,
      u_glareConvergence: s.glareConvergence / 100,
      u_glareOppositeFactor: s.glareOppositeFactor / 100,
      u_glareFactor: s.glareFactor / 100,
      u_glareHardness: s.glareHardness / 100,
      u_glareAngle: s.glareAngle * Math.PI / 180,
      u_blurEdge: s.blurEdge ? 1 : 0,
    });
  }
}

// Export globally
window.LiquidGlass = LiquidGlass;
