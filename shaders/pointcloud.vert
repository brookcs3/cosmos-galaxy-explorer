#version 410 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in float aFamilyId;
layout(location = 2) in float aCosmosId;
layout(location = 3) in float aConfidence;

uniform mat4 uProjection;
uniform float uPointSize;
uniform int uHoveredId;
uniform int uSelectedId;

out float vFamilyId;
out float vAlpha;
out float vIsHighlight;

void main() {
    gl_Position = uProjection * vec4(aPos, 0.0, 1.0);

    int cid = int(aCosmosId);
    float highlight = 0.0;
    if (cid == uHoveredId || cid == uSelectedId) {
        highlight = 1.0;
    }
    vIsHighlight = highlight;

    float size = uPointSize;
    if (highlight > 0.5) {
        size *= 2.5;
    }
    gl_PointSize = size;

    vFamilyId = aFamilyId;

    float conf = aConfidence;
    float alpha = 0.25 + abs(conf) * 0.15;
    alpha = clamp(alpha, 0.15, 0.85);
    if (highlight > 0.5) alpha = 1.0;
    vAlpha = alpha;
}
