"""
Standalone HTML5 / WebGL Interactive 3D Molecular Viewer Exporter.

Generates self-contained, single-file HTML pages that render molecules and
optimization trajectories in full 3D WebGL without requiring internet access,
external server processes, or installing Python.
"""

import json
import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .elements import ELEMENT_DATA, estimate_bond_length, get_cov_radius
from .forcefield import detect_connectivity

CPK_HEX_COLORS = {
    "H": "#f8fafc",   # Near white
    "C": "#475569",   # Slate gray
    "N": "#3b82f6",   # Blue
    "O": "#ef4444",   # Red
    "F": "#86efac",   # Pale green
    "CL": "#22c55e",  # Green
    "BR": "#b91c1c",  # Dark red/brown
    "I": "#8b5cf6",   # Purple
    "S": "#eab308",   # Yellow
    "P": "#f97316",   # Orange
    "B": "#fb923c",   # Light orange
    "SI": "#94a3b8",  # Gray
    "FE": "#ea580c",  # Rust orange
    "CU": "#d97706",  # Copper
    "ZN": "#64748b",  # Zinc gray
    "NA": "#6366f1",  # Violet
    "K": "#7c3aed",   # Deep violet
    "CA": "#475569",  # Dark gray
    "MG": "#15803d",  # Forest green
}


def get_element_hex_color(symbol: str) -> str:
    sym_u = symbol.upper()
    return CPK_HEX_COLORS.get(sym_u, "#ec4899")


def generate_interactive_html(
    symbols: List[str],
    coords: np.ndarray,
    title: str = "AgyMolden 3D Molecule",
    bonds: Optional[List[Tuple[int, int]]] = None,
    trajectory: Optional[List[np.ndarray]] = None,
    trajectory_energies: Optional[List[float]] = None,
    qm_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate self-contained standalone HTML with an interactive WebGL 3D viewer."""
    if bonds is None:
        bonds = detect_connectivity(symbols, coords)

    n_atoms = len(symbols)
    coords_list = coords.tolist()

    # Pre-calculate center and radius for default camera distance
    center = np.mean(coords, axis=0)
    shifted = coords - center
    max_r = float(np.max(np.linalg.norm(shifted, axis=1))) if len(shifted) > 0 else 2.0
    cam_dist = max(3.5, max_r * 2.8)

    # Process trajectory
    traj_frames = []
    if trajectory and len(trajectory) > 1:
        for frame in trajectory:
            traj_frames.append(np.array(frame, dtype=np.float64).tolist())
    else:
        traj_frames = [coords_list]

    # Pre-serialize data to JSON
    atom_data = []
    for i, s in enumerate(symbols):
        cov_r = get_cov_radius(s)
        atom_data.append({
            "idx": i,
            "sym": s,
            "color": get_element_hex_color(s),
            "radius": max(0.25, cov_r * 0.45),
        })

    data_payload = {
        "title": title,
        "symbols": symbols,
        "atoms": atom_data,
        "bonds": [[int(b[0]), int(b[1])] for b in bonds],
        "trajectory": traj_frames,
        "energies": trajectory_energies or [],
        "qm_summary": qm_summary or {},
        "cam_dist": cam_dist,
    }
    payload_json = json.dumps(data_payload)

    # Summary Card HTML content
    summary_html = ""
    if qm_summary:
        summary_rows = []
        if "program" in qm_summary:
            summary_rows.append(f"<div class='prop-row'><span>Program:</span><b>{qm_summary['program']}</b></div>")
        if "energy_ha" in qm_summary:
            summary_rows.append(f"<div class='prop-row'><span>Total Energy:</span><b>{qm_summary['energy_ha']:.6f} Ha</b></div>")
        if "dipole_debye" in qm_summary:
            summary_rows.append(f"<div class='prop-row'><span>Dipole Moment:</span><b>{qm_summary['dipole_debye']:.3f} Debye</b></div>")
        if "homo_ev" in qm_summary and "lumo_ev" in qm_summary:
            gap = qm_summary["lumo_ev"] - qm_summary["homo_ev"]
            summary_rows.append(f"<div class='prop-row'><span>HOMO / LUMO:</span><b>{qm_summary['homo_ev']:.2f} / {qm_summary['lumo_ev']:.2f} eV</b></div>")
            summary_rows.append(f"<div class='prop-row'><span>Band Gap:</span><b style='color:#38bdf8;'>{gap:.2f} eV</b></div>")
        if "frequencies_count" in qm_summary:
            summary_rows.append(f"<div class='prop-row'><span>Normal Modes:</span><b>{qm_summary['frequencies_count']} modes</b></div>")

        if summary_rows:
            summary_html = f"""
            <div class="card">
                <h3>Calculation Summary</h3>
                {''.join(summary_rows)}
            </div>
            """

    traj_ui_html = ""
    if len(traj_frames) > 1:
        traj_ui_html = f"""
        <div class="card">
            <h3>Optimization Trajectory</h3>
            <div class="controls-row">
                <button id="btnPlay" class="btn btn-primary">▶ Play</button>
                <input type="range" id="sliderStep" min="0" max="{len(traj_frames) - 1}" value="0" style="flex:1;">
                <span id="lblStep" style="font-family:monospace; min-width:65px;">Step 1/{len(traj_frames)}</span>
            </div>
            <div id="lblEnergy" style="color:#a7f3d0; font-family:monospace; font-size:12px; margin-top:6px;"></div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>{title} - AgyMolden 3D</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background-color: #0f172a;
    color: #f8fafc;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    overflow: hidden;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }}
  header {{
    background: #1e293b;
    padding: 10px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #334155;
    z-index: 10;
  }}
  h1 {{ font-size: 1.15rem; font-weight: bold; color: #38bdf8; display: flex; align-items: center; gap: 8px; }}
  .badge {{ background: #0284c7; font-size: 0.72rem; padding: 2px 7px; border-radius: 9999px; text-transform: uppercase; color: #fff; }}
  #main {{ flex: 1; position: relative; width: 100%; height: 100%; }}
  canvas {{ width: 100%; height: 100%; display: block; }}
  .panel {{
    position: absolute;
    top: 14px;
    right: 14px;
    width: 290px;
    max-height: calc(100vh - 90px);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
    z-index: 20;
    pointer-events: none;
  }}
  .card {{
    background: rgba(30, 41, 59, 0.88);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    padding: 12px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.4);
    pointer-events: auto;
  }}
  .card h3 {{ font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.05em; }}
  .prop-row {{ display: flex; justify-content: space-between; font-size: 0.85rem; padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }}
  .prop-row span {{ color: #94a3b8; }}
  .controls-row {{ display: flex; align-items: center; gap: 8px; }}
  .btn {{
    border: none;
    border-radius: 5px;
    padding: 6px 12px;
    font-size: 0.82rem;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .btn-primary {{ background: #2563eb; color: #fff; }}
  .btn-primary:hover {{ background: #1d4ed8; }}
  .btn-secondary {{ background: #334155; color: #f8fafc; }}
  .btn-secondary:hover {{ background: #475569; }}
  .btn-action {{ background: #059669; color: #fff; }}
  .btn-action:hover {{ background: #047857; }}
  .bottom-bar {{
    position: absolute;
    bottom: 14px;
    left: 14px;
    display: flex;
    gap: 8px;
    z-index: 20;
  }}
  #tooltip {{
    position: absolute;
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid #38bdf8;
    color: #fff;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: bold;
    pointer-events: none;
    display: none;
    z-index: 30;
  }}
</style>
</head>
<body>
<header>
  <h1><span>⚛️ {title}</span> <span class="badge">{n_atoms} Atoms</span></h1>
  <div style="display:flex; gap:8px;">
    <button class="btn btn-action" id="btnDownload">📥 Download .XYZ</button>
  </div>
</header>

<div id="main">
  <canvas id="glcanvas"></canvas>
  <div id="tooltip"></div>
  <div class="panel">
    {summary_html}
    {traj_ui_html}
    <div class="card">
      <h3>Viewer Controls</h3>
      <div style="display:flex; flex-direction:column; gap:6px;">
        <button class="btn btn-secondary" id="btnCenter">🎯 Reset Camera</button>
        <button class="btn btn-secondary" id="btnToggleLabels">🏷️ Toggle Labels</button>
      </div>
      <div style="font-size: 11px; color: #64748b; margin-top: 8px;">
        Rotate: Left click & drag<br>
        Zoom: Mouse scroll / Pinch<br>
        Pan: Right click & drag
      </div>
    </div>
  </div>
</div>

<script>
const DATA = {payload_json};

const canvas = document.getElementById('glcanvas');
const gl = canvas.getContext('webgl', {{ antialias: true }});
if (!gl) {{
  alert('WebGL not supported on this browser.');
}}

// Scene state
let currentFrame = 0;
let showLabels = true;
let isPlaying = false;
let playInterval = null;

// Camera
let camRotX = 0.3;
let camRotY = -0.4;
let camDist = DATA.cam_dist;
let panX = 0.0;
let panY = 0.0;

function resizeCanvas() {{
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth * dpr;
  const h = canvas.clientHeight * dpr;
  if (canvas.width !== w || canvas.height !== h) {{
    canvas.width = w;
    canvas.height = h;
    gl.viewport(0, 0, w, h);
  }}
}}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Shaders for Sphere Impostors (Ray-traced perfect spheres on camera-aligned quads)
const vsSource = `
  attribute vec3 aOffset;
  attribute vec2 aQuad;
  attribute vec3 aColor;
  attribute float aRadius;
  
  uniform mat4 uProjection;
  uniform mat4 uModelView;
  
  varying vec2 vQuad;
  varying vec3 vColor;
  varying float vRadius;
  varying vec4 vEyePos;
  
  void main() {{
    vQuad = aQuad;
    vColor = aColor;
    vRadius = aRadius;
    vec4 eyeCenter = uModelView * vec4(aOffset, 1.0);
    vec4 eyePos = eyeCenter + vec4(aQuad * aRadius * 1.3, 0.0, 0.0);
    vEyePos = eyePos;
    gl_Position = uProjection * eyePos;
  }}
`;

const fsSource = `
  precision mediump float;
  varying vec2 vQuad;
  varying vec3 vColor;
  varying float vRadius;
  varying vec4 vEyePos;
  uniform mat4 uProjection;

  void main() {{
    float r2 = dot(vQuad, vQuad);
    if (r2 > 1.0) discard;
    float z = sqrt(1.0 - r2);
    vec3 N = vec3(vQuad, z);
    
    // Lighting
    vec3 L1 = normalize(vec3(0.5, 0.8, 1.0));
    vec3 L2 = normalize(vec3(-0.4, -0.6, 0.7));
    float diff1 = max(dot(N, L1), 0.0);
    float diff2 = max(dot(N, L2), 0.0) * 0.4;
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 H = normalize(L1 + V);
    float spec = pow(max(dot(N, H), 0.0), 32.0) * 0.7;
    
    vec3 finalColor = vColor * (0.28 + diff1 + diff2) + vec3(spec);
    gl_FragColor = vec4(finalColor, 1.0);
  }}
`;

// Simple Line Shader for Bonds
const vsLineSource = `
  attribute vec3 aPosition;
  attribute vec3 aColor;
  uniform mat4 uProjection;
  uniform mat4 uModelView;
  varying vec3 vColor;
  void main() {{
    vColor = aColor;
    gl_Position = uProjection * uModelView * vec4(aPosition, 1.0);
  }}
`;
const fsLineSource = `
  precision mediump float;
  varying vec3 vColor;
  void main() {{
    gl_FragColor = vec4(vColor, 1.0);
  }}
`;

function createShader(gl, type, source) {{
  const s = gl.createShader(type);
  gl.shaderSource(s, source);
  gl.compileShader(s);
  return s;
}}
function createProgram(gl, vs, fs) {{
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  return p;
}}

const sphereProg = createProgram(gl, createShader(gl, gl.VERTEX_SHADER, vsSource), createShader(gl, gl.FRAGMENT_SHADER, fsSource));
const lineProg = createProgram(gl, createShader(gl, gl.VERTEX_SHADER, vsLineSource), createShader(gl, gl.FRAGMENT_SHADER, fsLineSource));

// Matrix math helpers
function mat4Perspective(fovRad, aspect, near, far) {{
  const f = 1.0 / Math.tan(fovRad / 2);
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) / (near - far), -1,
    0, 0, (2 * far * near) / (near - far), 0
  ];
}}

function mat4LookAt(eye, center, up) {{
  let z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
  let len = 1 / Math.hypot(z0, z1, z2);
  z0 *= len; z1 *= len; z2 *= len;
  let x0 = up[1] * z2 - up[2] * z1, x1 = up[2] * z0 - up[0] * z2, x2 = up[0] * z1 - up[1] * z0;
  len = 1 / Math.hypot(x0, x1, x2);
  x0 *= len; x1 *= len; x2 *= len;
  let y0 = z1 * x2 - z2 * x1, y1 = z2 * x0 - z0 * x2, y2 = z0 * x1 - z1 * x0;
  return [
    x0, y0, z0, 0,
    x1, y1, z1, 0,
    x2, y2, z2, 0,
    -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]),
    -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]),
    -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]),
    1
  ];
}}

function hexToRgb(hex) {{
  const bigint = parseInt(hex.replace('#',''), 16);
  return [(bigint >> 16 & 255)/255, (bigint >> 8 & 255)/255, (bigint & 255)/255];
}}

function render() {{
  resizeCanvas();
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0.06, 0.09, 0.16, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const coords = DATA.trajectory[currentFrame];
  
  // Calculate centroid
  let cx = 0, cy = 0, cz = 0;
  for (let c of coords) {{ cx += c[0]; cy += c[1]; cz += c[2]; }}
  cx /= coords.length; cy /= coords.length; cz /= coords.length;

  // View matrix
  const eyeX = cx + camDist * Math.sin(camRotY) * Math.cos(camRotX) + panX;
  const eyeY = cy + camDist * Math.sin(camRotX) + panY;
  const eyeZ = cz + camDist * Math.cos(camRotY) * Math.cos(camRotX);
  const uModelView = mat4LookAt([eyeX, eyeY, eyeZ], [cx + panX, cy + panY, cz], [0, 1, 0]);
  const uProjection = mat4Perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 100.0);

  // 1. Draw Bonds (Lines)
  gl.useProgram(lineProg);
  gl.uniformMatrix4fv(gl.getUniformLocation(lineProg, 'uProjection'), false, new Float32Array(uProjection));
  gl.uniformMatrix4fv(gl.getUniformLocation(lineProg, 'uModelView'), false, new Float32Array(uModelView));

  const lineVerts = [];
  const lineCols = [];
  for (let b of DATA.bonds) {{
    const p1 = coords[b[0]], p2 = coords[b[1]];
    const c1 = hexToRgb(DATA.atoms[b[0]].color);
    const c2 = hexToRgb(DATA.atoms[b[1]].color);
    const mid = [(p1[0]+p2[0])*0.5, (p1[1]+p2[1])*0.5, (p1[2]+p2[2])*0.5];
    lineVerts.push(...p1, ...mid, ...mid, ...p2);
    lineCols.push(...c1, ...c1, ...c2, ...c2);
  }}

  if (lineVerts.length > 0) {{
    const lineBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lineVerts), gl.STATIC_DRAW);
    const posLoc = gl.getAttribLocation(lineProg, 'aPosition');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

    const colBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lineCols), gl.STATIC_DRAW);
    const colLoc = gl.getAttribLocation(lineProg, 'aColor');
    gl.enableVertexAttribArray(colLoc);
    gl.vertexAttribPointer(colLoc, 3, gl.FLOAT, false, 0, 0);

    gl.lineWidth(3);
    gl.drawArrays(gl.LINES, 0, lineVerts.length / 3);

    gl.disableVertexAttribArray(posLoc);
    gl.disableVertexAttribArray(colLoc);
    gl.deleteBuffer(lineBuf);
    gl.deleteBuffer(colBuf);
  }}

  // 2. Draw Atoms (Sphere Quads)
  gl.useProgram(sphereProg);
  gl.uniformMatrix4fv(gl.getUniformLocation(sphereProg, 'uProjection'), false, new Float32Array(uProjection));
  gl.uniformMatrix4fv(gl.getUniformLocation(sphereProg, 'uModelView'), false, new Float32Array(uModelView));

  const atomData = [];
  const qVerts = [
    [-1, -1], [1, -1], [-1, 1],
    [-1, 1], [1, -1], [1, 1]
  ];
  for (let i = 0; i < coords.length; i++) {{
    const p = coords[i];
    const a = DATA.atoms[i];
    const col = hexToRgb(a.color);
    for (let k = 0; k < 6; k++) {{
      atomData.push(
        p[0], p[1], p[2],
        qVerts[k][0], qVerts[k][1],
        col[0], col[1], col[2],
        a.radius
      );
    }}
  }}

  const atomBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, atomBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(atomData), gl.STATIC_DRAW);

  const stride = 9 * 4;
  const offLoc = gl.getAttribLocation(sphereProg, 'aOffset');
  const qLoc = gl.getAttribLocation(sphereProg, 'aQuad');
  const cLoc = gl.getAttribLocation(sphereProg, 'aColor');
  const rLoc = gl.getAttribLocation(sphereProg, 'aRadius');

  gl.enableVertexAttribArray(offLoc);
  gl.vertexAttribPointer(offLoc, 3, gl.FLOAT, false, stride, 0);

  gl.enableVertexAttribArray(qLoc);
  gl.vertexAttribPointer(qLoc, 2, gl.FLOAT, false, stride, 12);

  gl.enableVertexAttribArray(cLoc);
  gl.vertexAttribPointer(cLoc, 3, gl.FLOAT, false, stride, 20);

  gl.enableVertexAttribArray(rLoc);
  gl.vertexAttribPointer(rLoc, 1, gl.FLOAT, false, stride, 32);

  gl.drawArrays(gl.TRIANGLES, 0, coords.length * 6);

  gl.disableVertexAttribArray(offLoc);
  gl.disableVertexAttribArray(qLoc);
  gl.disableVertexAttribArray(cLoc);
  gl.disableVertexAttribArray(rLoc);
  gl.deleteBuffer(atomBuf);
}}

// Mouse / Touch Controls
let isDragging = false;
let isPanning = false;
let lastX = 0, lastY = 0;

canvas.addEventListener('mousedown', e => {{
  isDragging = true;
  isPanning = (e.button === 2);
  lastX = e.clientX; lastY = e.clientY;
}});
window.addEventListener('mouseup', () => {{ isDragging = false; isPanning = false; }});
canvas.addEventListener('contextmenu', e => e.preventDefault());

canvas.addEventListener('mousemove', e => {{
  if (!isDragging) return;
  const dx = e.clientX - lastX;
  const dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  if (isPanning) {{
    panX += dx * 0.008 * (camDist / 5.0);
    panY -= dy * 0.008 * (camDist / 5.0);
  }} else {{
    camRotY += dx * 0.01;
    camRotX += dy * 0.01;
    camRotX = Math.max(-1.5, Math.min(1.5, camRotX));
  }}
  render();
}});

canvas.addEventListener('wheel', e => {{
  camDist *= (e.deltaY > 0 ? 1.08 : 0.92);
  camDist = Math.max(1.0, Math.min(80.0, camDist));
  render();
  e.preventDefault();
}}, {{ passive: false }});

// Touch support for phones & tablets
let touchStartDist = 0;
canvas.addEventListener('touchstart', e => {{
  if (e.touches.length === 1) {{
    isDragging = true; isPanning = false;
    lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
  }} else if (e.touches.length === 2) {{
    touchStartDist = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
  }}
}});
canvas.addEventListener('touchmove', e => {{
  if (e.touches.length === 1 && isDragging) {{
    const dx = e.touches[0].clientX - lastX;
    const dy = e.touches[0].clientY - lastY;
    lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
    camRotY += dx * 0.01;
    camRotX += dy * 0.01;
    render();
  }} else if (e.touches.length === 2) {{
    const dist = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
    const factor = touchStartDist / (dist + 1e-5);
    camDist *= (factor > 1.0 ? 1.04 : 0.96);
    touchStartDist = dist;
    render();
  }}
  e.preventDefault();
}}, {{ passive: false }});
window.addEventListener('touchend', () => {{ isDragging = false; }});

// Trajectory Controls
const btnPlay = document.getElementById('btnPlay');
const sliderStep = document.getElementById('sliderStep');
const lblStep = document.getElementById('lblStep');
const lblEnergy = document.getElementById('lblEnergy');

function updateStep(step) {{
  currentFrame = step;
  if (sliderStep) sliderStep.value = step;
  if (lblStep) lblStep.innerText = `Step ${{step + 1}}/${{DATA.trajectory.length}}`;
  if (lblEnergy && DATA.energies && DATA.energies[step] !== undefined) {{
    const eHa = DATA.energies[step] / 27.211386;
    lblEnergy.innerText = `E = ${{eHa.toFixed(6)}} Ha`;
  }}
  render();
}}

if (sliderStep) {{
  sliderStep.addEventListener('input', e => updateStep(parseInt(e.target.value)));
}}

if (btnPlay) {{
  btnPlay.addEventListener('click', () => {{
    if (isPlaying) {{
      clearInterval(playInterval);
      btnPlay.innerText = '▶ Play';
      isPlaying = false;
    }} else {{
      btnPlay.innerText = '⏸ Pause';
      isPlaying = true;
      playInterval = setInterval(() => {{
        let next = (currentFrame + 1) % DATA.trajectory.length;
        updateStep(next);
      }}, 120);
    }}
  }});
}}

// Buttons
document.getElementById('btnCenter').addEventListener('click', () => {{
  camRotX = 0.3; camRotY = -0.4;
  camDist = DATA.cam_dist;
  panX = 0; panY = 0;
  render();
}});

document.getElementById('btnDownload').addEventListener('click', () => {{
  const coords = DATA.trajectory[currentFrame];
  let xyz = `${{DATA.symbols.length}}\\n${{DATA.title}} - Frame ${{currentFrame + 1}}\\n`;
  for (let i = 0; i < DATA.symbols.length; i++) {{
    xyz += `${{DATA.symbols[i]}}  ${{coords[i][0].toFixed(6)}}  ${{coords[i][1].toFixed(6)}}  ${{coords[i][2].toFixed(6)}}\\n`;
  }}
  const blob = new Blob([xyz], {{ type: 'text/plain' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${{DATA.title.replace(/\\s+/g, '_')}}.xyz`;
  a.click();
}});

updateStep(0);
</script>
</body>
</html>
"""
    return html
