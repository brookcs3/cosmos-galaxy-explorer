#version 410 core

in float vFamilyId;
in float vAlpha;
in float vIsHighlight;

out vec4 FragColor;

// 15 family colors (index 14 = unknown)
uniform vec3 uFamilyColors[15];

void main() {
    // Circular point shape
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = dot(coord, coord);
    if (dist > 0.25) discard;

    int fid = clamp(int(vFamilyId + 0.5), 0, 14);
    vec3 color = uFamilyColors[fid];

    // Soft edge
    float edge = smoothstep(0.25, 0.18, dist);
    float alpha = vAlpha * edge;

    // Glow effect for highlighted points
    if (vIsHighlight > 0.5) {
        float glow = smoothstep(0.25, 0.0, dist);
        alpha = mix(alpha, 1.0, glow * 0.3);
    }

    FragColor = vec4(color, alpha);
}
