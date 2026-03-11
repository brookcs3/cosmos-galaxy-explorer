#version 410 core

flat in int vCosmosId;

out vec4 FragColor;

void main() {
    // Circular point shape for picking too
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (dot(coord, coord) > 0.25) discard;

    // Encode cosmos_id as RGB (supports up to 16,777,215 IDs)
    int id = vCosmosId;
    float r = float(id & 0xFF) / 255.0;
    float g = float((id >> 8) & 0xFF) / 255.0;
    float b = float((id >> 16) & 0xFF) / 255.0;
    FragColor = vec4(r, g, b, 1.0);
}
