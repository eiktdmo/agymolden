"""3D procedural mesh generation and GLSL shader sources for MoleculeViewer."""

import math
from typing import Tuple
import numpy as np

def create_sphere_mesh(stacks: int = 14, sectors: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """Generate vertex positions and normals for a unit sphere."""
    vertices = []
    normals = []

    for i in range(stacks + 1):
        lat = math.pi * (-0.5 + float(i) / stacks)
        z = math.sin(lat)
        zr = math.cos(lat)

        for j in range(sectors + 1):
            lng = 2.0 * math.pi * float(j) / sectors
            x = zr * math.cos(lng)
            y = zr * math.sin(lng)

            vertices.extend([x, y, z])
            normals.extend([x, y, z])

    indices = []
    for i in range(stacks):
        k1 = i * (sectors + 1)
        k2 = k1 + sectors + 1
        for j in range(sectors):
            if i != 0:
                indices.extend([k1 + j, k2 + j, k1 + j + 1])
            if i != (stacks - 1):
                indices.extend([k1 + j + 1, k2 + j, k2 + j + 1])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


def create_cylinder_mesh(sectors: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Generate vertex positions and normals for a unit cylinder along Z axis (0 to 1)."""
    vertices = []
    indices = []

    # Side vertices: 2 rings
    for j in range(sectors + 1):
        angle = 2.0 * math.pi * float(j) / sectors
        x = math.cos(angle)
        y = math.sin(angle)
        # Ring 0 at z=0, Ring 1 at z=1
        vertices.extend([x, y, 0.0,  x, y, 0.0]) # pos, normal
        vertices.extend([x, y, 1.0,  x, y, 0.0])

    stride = 2
    for j in range(sectors):
        p0 = j * stride
        p1 = p0 + 1
        p2 = (j + 1) * stride
        p3 = p2 + 1
        indices.extend([p0, p1, p2])
        indices.extend([p2, p1, p3])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


def create_cone_mesh(sectors: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Generate vertex positions and normals for a unit cone along Z axis (base at z=0, apex at z=1)."""
    vertices = []
    indices = []

    # Base center: vertex 0
    vertices.extend([0.0, 0.0, 0.0,  0.0, 0.0, -1.0])
    # Apex: vertex 1
    vertices.extend([0.0, 0.0, 1.0,  0.0, 0.0, 1.0])

    # Base ring vertices
    for j in range(sectors + 1):
        angle = 2.0 * math.pi * float(j) / sectors
        x = math.cos(angle)
        y = math.sin(angle)
        # Side normal pointing outwards
        vertices.extend([x, y, 0.0,  x, y, 0.5])

    for j in range(sectors):
        p_ring = 2 + j
        p_ring_next = 2 + j + 1
        # Side triangle: apex -> p_ring -> p_ring_next
        indices.extend([1, p_ring, p_ring_next])
        # Base triangle: center -> p_ring_next -> p_ring
        indices.extend([0, p_ring_next, p_ring])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

// Optional per-instance attributes (used during instanced draw calls)
layout(location = 2) in vec3 in_instance_pos;
layout(location = 3) in float in_instance_radius;
layout(location = 4) in vec3 in_instance_color;
layout(location = 5) in float in_instance_hl;
layout(location = 6) in vec3 in_instance_end;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normal_mat;
uniform int u_use_instancing;

out vec3 v_normal;
out vec3 v_frag_pos;
flat out vec3 v_inst_color;
flat out int v_inst_hl;

void main() {
    vec4 world_pos;
    if (u_use_instancing == 1) {
        vec3 scaled_pos = in_position * in_instance_radius + in_instance_pos;
        world_pos = vec4(scaled_pos, 1.0);
        v_normal = in_normal; // Unit sphere normal equals vertex position
        v_inst_color = in_instance_color;
        v_inst_hl = int(in_instance_hl + 0.5);
    } else if (u_use_instancing == 2) {
        vec3 delta = in_instance_end - in_instance_pos;
        float len = length(delta);
        vec3 u = (len > 1e-6) ? (delta / len) : vec3(0.0, 0.0, 1.0);
        vec3 helper = (abs(u.z) < 0.999) ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
        vec3 t = normalize(cross(helper, u));
        vec3 b = cross(u, t);
        vec3 local_pos = (t * (in_position.x * in_instance_radius))
                       + (b * (in_position.y * in_instance_radius))
                       + (u * (in_position.z * len))
                       + in_instance_pos;
        world_pos = vec4(local_pos, 1.0);
        v_normal = normalize(t * in_normal.x + b * in_normal.y + u * in_normal.z);
        v_inst_color = in_instance_color;
        v_inst_hl = int(in_instance_hl + 0.5);
    } else {
        world_pos = u_model * vec4(in_position, 1.0);
        v_normal = normalize(u_normal_mat * in_normal);
        v_inst_color = vec3(0.0);
        v_inst_hl = -1;
    }
    v_frag_pos = world_pos.xyz;
    gl_Position = u_proj * u_view * world_pos;
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec3 v_normal;
in vec3 v_frag_pos;
flat in vec3 v_inst_color;
flat in int v_inst_hl;

uniform vec3 u_color;
uniform vec3 u_light_pos;
uniform vec3 u_view_pos;
uniform int u_is_highlighted;
uniform int u_use_instancing;
uniform int u_publication_outline;
uniform int u_modern_glossy;
uniform int u_depth_cue;

out vec4 frag_color;

void main() {
    vec3 base_col = (u_use_instancing == 1) ? v_inst_color : u_color;
    int hl_flag = (u_use_instancing == 1) ? v_inst_hl : u_is_highlighted;

    vec3 norm = normalize(v_normal);
    vec3 light_dir = normalize(u_light_pos - v_frag_pos);
    vec3 view_dir = normalize(u_view_pos - v_frag_pos);

    vec3 result;
    if (u_publication_outline == 1) {
        // IboView / QuteMol style hemisphere lighting + crisp ink silhouette rim
        float hemi = 0.5 * norm.y + 0.5;
        vec3 ambient = mix(0.28 * base_col, 0.45 * base_col, hemi);
        float diff = max(dot(norm, light_dir), 0.0);
        float soft_diff = smoothstep(0.0, 0.85, diff);
        vec3 diffuse = soft_diff * base_col * 0.72;
        vec3 half_dir = normalize(light_dir + view_dir);
        float spec = pow(max(dot(norm, half_dir), 0.0), 48.0);
        vec3 specular = 0.38 * spec * vec3(1.0, 1.0, 1.0);
        result = ambient + diffuse + specular;
    } else if (u_modern_glossy == 1) {
        // ChimeraX / Avogadro 2 style dual-source Blinn-Phong glossy shading
        float hemi = 0.4 * norm.y + 0.6;
        vec3 ambient = 0.32 * hemi * base_col;

        // Key light
        float diff1 = max(dot(norm, light_dir), 0.0);
        vec3 diffuse1 = diff1 * base_col * 0.70;
        vec3 half_dir1 = normalize(light_dir + view_dir);
        float spec1 = pow(max(dot(norm, half_dir1), 0.0), 64.0);
        vec3 specular1 = 0.55 * spec1 * vec3(1.0, 1.0, 1.0);

        // Fill light (camera opposite side) for depth softness
        vec3 fill_dir = normalize(vec3(-light_dir.x, light_dir.y * 0.5, light_dir.z));
        float diff2 = max(dot(norm, fill_dir), 0.0);
        vec3 diffuse2 = diff2 * base_col * 0.20;

        // Subtle Fresnel rim glow
        float fresnel = pow(1.0 - max(dot(norm, view_dir), 0.0), 3.0);
        vec3 rim = 0.18 * fresnel * mix(base_col, vec3(1.0), 0.5);

        // Cavity ambient occlusion / contact obscurance
        float cavity_ao = clamp(0.70 + 0.30 * dot(norm, view_dir), 0.60, 1.0);

        result = (ambient * cavity_ao) + diffuse1 + diffuse2 + specular1 + rim;

        // Screen-space depth cueing (atmospheric depth falloff)
        if (u_depth_cue == 1) {
            float depth_falloff = 1.0 - (gl_FragCoord.z * 0.22);
            result *= depth_falloff;
        }
    } else {
        // Legacy Ambient
        float ambient_strength = 0.35;
        vec3 ambient = ambient_strength * base_col;

        // Diffuse
        float diff = max(dot(norm, light_dir), 0.0);
        vec3 diffuse = diff * base_col * 0.75;

        // Specular
        float specular_strength = 0.45;
        vec3 reflect_dir = reflect(-light_dir, norm);
        float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
        vec3 specular = specular_strength * spec * vec3(1.0, 1.0, 1.0);

        result = ambient + diffuse + specular;
    }
    // Tiny ordered noise (dither) removes visible 8-bit banding rings on smooth spheres
    float dither = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
    result += vec3(dither / 255.0);

    if (hl_flag == 1) {
        // Primary selected atom: bright vibrant golden illumination
        result = result * 1.35 + vec3(0.48, 0.40, 0.05);
    } else if (hl_flag == 2) {
        // Multi-selected atoms: bright vibrant sky-blue / cyan illumination
        result = result * 1.30 + vec3(0.05, 0.35, 0.50);
    } else if (hl_flag == 3) {
        // Chosen connectivity reference (green)
        result = mix(result, vec3(0.15, 0.95, 0.35), 0.6);
    } else if (hl_flag == 4) {
        // Target atom in active feature (vibrant gold)
        result = mix(result, vec3(1.0, 0.84, 0.0), 0.65);
    } else if (hl_flag == 5) {
        // Bond reference atom (vibrant cyan)
        result = mix(result, vec3(0.0, 0.9, 1.0), 0.65);
    } else if (hl_flag == 6) {
        // Angle reference atom (bright purple/magenta)
        result = mix(result, vec3(0.9, 0.3, 1.0), 0.65);
    } else if (hl_flag == 7) {
        // Dihedral reference atom (bright orange)
        result = mix(result, vec3(1.0, 0.5, 0.1), 0.65);
    }

    if (u_publication_outline == 1) {
        float ndotv = abs(dot(norm, view_dir));
        float edge = smoothstep(0.24, 0.34, ndotv);
        result = mix(vec3(0.06, 0.08, 0.12), result, edge);
    }

    frag_color = vec4(result, 1.0);
}
"""

SURFACE_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;

uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 v_normal;
out vec3 v_frag_pos;
out vec3 v_color;

void main() {
    v_frag_pos = in_position;
    v_normal = in_normal;
    v_color = in_color;
    gl_Position = u_proj * u_view * vec4(in_position, 1.0);
}
"""

SURFACE_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_normal;
in vec3 v_frag_pos;
in vec3 v_color;

uniform vec3 u_color;
uniform int u_use_vertex_color;
uniform vec3 u_light_pos;
uniform vec3 u_view_pos;
uniform float u_opacity;

out vec4 frag_color;

void main() {
    // Two-sided normal for smooth isosurface shading
    vec3 norm = normalize(v_normal);
    vec3 view_dir = normalize(u_view_pos - v_frag_pos);
    if (dot(norm, view_dir) < 0.0) {
        norm = -norm;
    }

    vec3 base_col = (u_use_vertex_color == 1) ? v_color : u_color;

    // Key light (primary directional source)
    vec3 light_dir = normalize(u_light_pos - v_frag_pos);
    float diff1 = max(dot(norm, light_dir), 0.0);
    vec3 diffuse1 = diff1 * base_col * 0.70;

    // Fill light from opposite quadrant to soften shadows and reveal 3D curvature
    vec3 fill_dir = normalize(vec3(-light_dir.x, -light_dir.y * 0.5, light_dir.z));
    float diff2 = max(dot(norm, fill_dir), 0.0);
    vec3 diffuse2 = diff2 * base_col * 0.25;

    // Subtle hemisphere ambient
    vec3 ambient = 0.32 * base_col;

    // Blinn-Phong specular highlight for realistic glossy surface sheen
    vec3 half_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(norm, half_dir), 0.0), 32.0);
    vec3 specular = 0.35 * spec * vec3(1.0, 1.0, 1.0);

    // Fresnel rim-lighting edge enhancement (subtle translucent depth)
    float fresnel = pow(1.0 - max(dot(norm, view_dir), 0.0), 3.0);
    vec3 rim = 0.18 * fresnel * base_col;

    vec3 result = ambient + diffuse1 + diffuse2 + specular + rim;
    frag_color = vec4(result, u_opacity);
}
"""


