#version 410 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in float aFamilyId;
layout(location = 2) in float aCosmosId;
layout(location = 3) in float aConfidence;

uniform mat4 uProjection;
uniform float uPointSize;

flat out int vCosmosId;

void main() {
    gl_Position = uProjection * vec4(aPos, 0.0, 1.0);
    gl_PointSize = uPointSize * 1.5;  // slightly larger for easier picking
    vCosmosId = int(aCosmosId);
}
