// COSMOS Galaxy Explorer — C++/OpenGL Port
// Renders ~44K audio sample points as a 2D point cloud with GPU color picking,
// drag-to-scrub audio, custom cursor effect, and ImGui UI overlays.

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "lib/imgui.h"
#include "lib/imgui_impl_glfw.h"
#include "lib/imgui_impl_opengl3.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>

#define MINIAUDIO_IMPLEMENTATION
#include "lib/miniaudio.h"
#include "lib/json.hpp"

using json = nlohmann::json;

// ─── Data Structures ─────────────────────────────────────────────────

struct GalaxyPoint {
    int cosmosId;
    std::string name;
    std::string instrumentName;
    std::string familyName;
    int type;  // 1=oneshot, 2=loop

    // 5 VAE heads: crest, wetdry, satclean, centroid, instrument
    float headX[5];
    float headY[5];
    float headC[5];

    std::string wavPath;
    int familyId;  // 0-14 index for coloring

    std::vector<int> tagIndices;  // indices into gTagNames
    int instrumentId;  // index into gInstrumentNames
};

// Head indices
enum Head { HEAD_CREST=0, HEAD_WETDRY=1, HEAD_SATCLEAN=2, HEAD_CENTROID=3, HEAD_INSTRUMENT=4 };
static const char* HEAD_NAMES[] = { "Crest", "Wet / Dry", "Sat / Clean", "Centroid", "Instrument" };
static const char* HEAD_DESCS[] = {
    "Transient Character: punchy <-> sustained",
    "Effect Amount: dry <-> ambient",
    "Saturation: warm <-> clean",
    "Spectral Brightness: dark <-> bright",
    "Instrument Family Embedding"
};

// Family color palette (from the original JS, extended)
struct FamilyInfo {
    const char* name;
    float r, g, b;
};

static const FamilyInfo FAMILIES[] = {
    { "Bass",           0.216f, 0.271f, 0.525f },  // #374586
    { "Brass & Wind",   0.153f, 0.337f, 0.380f },  // #275661
    { "Cymbals",        0.933f, 0.663f, 0.047f },  // #EEA90C
    { "Drums",          0.651f, 0.275f, 0.337f },  // #A64656
    { "FX",             0.616f, 0.318f, 0.569f },  // #9D5191
    { "Guitar",         0.282f, 0.612f, 0.749f },  // #489CBF
    { "Keys",           0.165f, 0.631f, 0.455f },  // #2AA174
    { "Mallet",         0.357f, 0.537f, 0.435f },  // #5B896F
    { "Percussion",     0.365f, 0.706f, 0.063f },  // #5DB410
    { "Strings",        0.376f, 0.733f, 0.678f },  // #60BBAD
    { "Synth",          0.502f, 0.431f, 0.616f },  // #806E9D
    { "Vocal",          0.686f, 0.314f, 0.471f },  // #AF5078
    { "__Pipeline__",   1.000f, 0.000f, 1.000f },  // Magenta #FF00FF
    { "__GroundTruth__",0.300f, 0.300f, 0.350f },  // Dark gray
    { "Unknown",        0.165f, 0.180f, 0.220f },  // #2a2e38
};
static const int NUM_FAMILIES = 15;

// ─── Globals ─────────────────────────────────────────────────────────

static std::vector<GalaxyPoint> gPoints;
static std::unordered_map<int, size_t> gIdToIndex;

// Family counts for legend
struct FamilyStats {
    int familyId;
    int count;
};
static std::vector<FamilyStats> gFamilyStats;

// Current view
static int gCurrentHead = HEAD_CREST;

// Camera
static float gCamX = 0.0f, gCamY = 0.0f, gCamZoom = 1.0f;

// Drag state
enum DragMode { DRAG_NONE=0, DRAG_PAN=1, DRAG_SCRUB=2 };
static DragMode gDragMode = DRAG_NONE;
static double gDragStartX, gDragStartY;
static float gDragCamX, gDragCamY;
static double gDragStartMouseX, gDragStartMouseY;
static int gDragLastPlayedId = -1;

// Selection
static int gHoveredId = -1;
static int gSelectedId = -1;

// Picking state
static double gLastHoverPickTime = 0.0;
static const double HOVER_PICK_INTERVAL = 0.016;

// Mouse position
static double gLastMouseX = 0.0, gLastMouseY = 0.0;

// Custom cursor spring physics
static float gCursorSpringX = 0.0f, gCursorSpringY = 0.0f;
static GLuint gCursorProgram = 0;
static GLuint gCursorVAO = 0, gCursorVBO = 0;

// Window
static GLFWwindow* gWindow = nullptr;
static int gWinWidth = 1600, gWinHeight = 1000;
static float gPixelRatio = 1.0f;

// OpenGL objects
static GLuint gVAO = 0, gVBO = 0;
static GLuint gVisibleProgram = 0, gPickingProgram = 0;
static GLuint gPickFBO = 0, gPickColorTex = 0, gPickDepthRB = 0;

// Vertex data layout
struct PointVertex {
    float x, y;
    float familyId;
    float cosmosId;
    float confidence;
};
static std::vector<PointVertex> gVertices;

// Audio
static ma_engine gAudioEngine;
static bool gAudioInitialized = false;
static ma_sound gCurrentSound;
static bool gSoundLoaded = false;

// Visibility / filtering
static bool gShowOneshots = true;
static bool gShowLoops = false;
static bool gFamilyHidden[NUM_FAMILIES] = {};

// Instrument filter (-1 = show all)
static std::vector<std::string> gInstrumentNames;
static std::unordered_map<std::string, int> gInstrumentNameToId;
static int gInstrumentFilter = -1;

// Tags filter
static std::vector<std::string> gTagNames;
static std::vector<int> gTagCounts;
static std::vector<bool> gTagActive;  // which tags are enabled for filtering
static int gActiveTagCount = 0;

// Animation state for head transitions
struct AnimPoint {
    float curX, curY;
    float tgtX, tgtY;
};
static std::vector<AnimPoint> gAnimPoints;
static bool gAnimating = false;
static int gPrevHead = -1;

// Data bounds per head
struct HeadBounds {
    float minX, maxX, minY, maxY;
};
static HeadBounds gHeadBounds[5];

// Performance
static double gLastFrameTime = 0.0;
static int gFrameCount = 0;
static double gFpsTimer = 0.0;
static float gFps = 0.0f;
static int gVisibleCount = 0;

// ImGui search
static char gSearchBuf[256] = {};

// Injected sample state
static int gNextInjectedId = 100000;  // synthetic IDs above real cosmos_id range (~50K max)
static bool gAnalyzeRunning = false;
static std::string gAnalyzeStatus;
static const char* ANALYZE_SCRIPT =
    "/Users/cameronbrooks/Server/AI-STEM-Separator-Mad-Scientist-Edition/"
    "ml-ops/galaxy-semantic-backbone/analyze_single.py";

// ─── Shader Utilities ────────────────────────────────────────────────

static std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: Cannot open file: %s\n", path.c_str());
        return "";
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static GLuint compileShader(GLenum type, const std::string& source, const std::string& name) {
    GLuint shader = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        fprintf(stderr, "ERROR compiling %s:\n%s\n", name.c_str(), log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint createProgram(const std::string& vertPath, const std::string& fragPath) {
    std::string vertSrc = readFile(vertPath);
    std::string fragSrc = readFile(fragPath);
    if (vertSrc.empty() || fragSrc.empty()) return 0;

    GLuint vert = compileShader(GL_VERTEX_SHADER, vertSrc, vertPath);
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragSrc, fragPath);
    if (!vert || !frag) return 0;

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);

    GLint success;
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if (!success) {
        char log[1024];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        fprintf(stderr, "ERROR linking program (%s + %s):\n%s\n",
                vertPath.c_str(), fragPath.c_str(), log);
        glDeleteProgram(prog);
        return 0;
    }

    glDeleteShader(vert);
    glDeleteShader(frag);
    return prog;
}

static GLuint createProgramFromSource(const std::string& vertSrc, const std::string& fragSrc) {
    GLuint vert = compileShader(GL_VERTEX_SHADER, vertSrc, "cursor.vert");
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragSrc, "cursor.frag");
    if (!vert || !frag) return 0;

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);

    GLint success;
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if (!success) {
        char log[1024];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        fprintf(stderr, "ERROR linking cursor program:\n%s\n", log);
        glDeleteProgram(prog);
        return 0;
    }

    glDeleteShader(vert);
    glDeleteShader(frag);
    return prog;
}

// ─── Forward Declarations ────────────────────────────────────────────
static void resetCamera();
static bool gNeedsRebuild = false;

// ─── Family ID Resolution ────────────────────────────────────────────

static int resolveFamilyId(const std::string& family, const std::string& instrument) {
    std::string fam = family;
    if (fam.empty()) fam = instrument;

    for (int i = 0; i < NUM_FAMILIES; i++) {
        if (fam == FAMILIES[i].name) return i;
    }
    return NUM_FAMILIES - 1;  // Unknown
}

// ─── Data Loading ────────────────────────────────────────────────────

static bool loadGalaxyData(const std::string& path) {
    fprintf(stderr, "Loading galaxy data from %s...\n", path.c_str());
    auto t0 = std::chrono::steady_clock::now();

    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: Cannot open data file: %s\n", path.c_str());
        return false;
    }

    json data;
    try {
        data = json::parse(f);
    } catch (const json::parse_error& e) {
        fprintf(stderr, "ERROR: JSON parse error: %s\n", e.what());
        return false;
    }

    if (!data.is_array()) {
        fprintf(stderr, "ERROR: Expected JSON array\n");
        return false;
    }

    gPoints.reserve(data.size());

    for (int h = 0; h < 5; h++) {
        gHeadBounds[h] = { 1e9f, -1e9f, 1e9f, -1e9f };
    }

    int familyCounts[NUM_FAMILIES] = {};

    for (auto& item : data) {
        if (!item.is_array() || item.size() < 20) continue;

        GalaxyPoint p;
        p.cosmosId = item[0].get<int>();
        p.name = item[1].is_string() ? item[1].get<std::string>() : "";
        p.instrumentName = item[2].is_string() ? item[2].get<std::string>() : "";
        p.familyName = item[3].is_string() ? item[3].get<std::string>() : "";
        p.type = item[4].is_number() ? item[4].get<int>() : 1;

        int offsets[] = { 5, 8, 11, 14, 17 };
        for (int h = 0; h < 5; h++) {
            int base = offsets[h];
            p.headX[h] = item[base].is_number() ? item[base].get<float>() : 0.0f;
            p.headY[h] = item[base+1].is_number() ? item[base+1].get<float>() : 0.0f;
            if (h < 4) {
                p.headC[h] = item[base+2].is_number() ? item[base+2].get<float>() : 0.0f;
            } else {
                p.headC[h] = 0.5f;
            }

            gHeadBounds[h].minX = std::min(gHeadBounds[h].minX, p.headX[h]);
            gHeadBounds[h].maxX = std::max(gHeadBounds[h].maxX, p.headX[h]);
            gHeadBounds[h].minY = std::min(gHeadBounds[h].minY, p.headY[h]);
            gHeadBounds[h].maxY = std::max(gHeadBounds[h].maxY, p.headY[h]);
        }

        p.wavPath = item[19].is_string() ? item[19].get<std::string>() : "";
        p.familyId = resolveFamilyId(p.familyName, p.instrumentName);

        // Assign instrument ID (build lookup on the fly)
        auto instIt = gInstrumentNameToId.find(p.instrumentName);
        if (instIt != gInstrumentNameToId.end()) {
            p.instrumentId = instIt->second;
        } else {
            p.instrumentId = (int)gInstrumentNames.size();
            gInstrumentNameToId[p.instrumentName] = p.instrumentId;
            gInstrumentNames.push_back(p.instrumentName);
        }

        familyCounts[p.familyId]++;
        size_t idx = gPoints.size();
        gPoints.push_back(std::move(p));
        gIdToIndex[gPoints[idx].cosmosId] = idx;
    }

    // Build sorted family stats
    gFamilyStats.clear();
    for (int i = 0; i < NUM_FAMILIES; i++) {
        if (familyCounts[i] > 0) {
            gFamilyStats.push_back({ i, familyCounts[i] });
        }
    }
    std::sort(gFamilyStats.begin(), gFamilyStats.end(),
              [](const FamilyStats& a, const FamilyStats& b) { return a.count > b.count; });

    // Sort instrument names alphabetically, rebuild IDs
    std::sort(gInstrumentNames.begin(), gInstrumentNames.end());
    gInstrumentNameToId.clear();
    for (int i = 0; i < (int)gInstrumentNames.size(); i++) {
        gInstrumentNameToId[gInstrumentNames[i]] = i;
    }
    for (auto& pt : gPoints) {
        pt.instrumentId = gInstrumentNameToId[pt.instrumentName];
    }

    // Initialize animation points to current head positions
    gAnimPoints.resize(gPoints.size());
    for (size_t i = 0; i < gPoints.size(); i++) {
        gAnimPoints[i].curX = gPoints[i].headX[gCurrentHead];
        gAnimPoints[i].curY = gPoints[i].headY[gCurrentHead];
        gAnimPoints[i].tgtX = gAnimPoints[i].curX;
        gAnimPoints[i].tgtY = gAnimPoints[i].curY;
    }
    gPrevHead = gCurrentHead;

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "Loaded %zu points (%zu instruments) in %.1f ms\n",
            gPoints.size(), gInstrumentNames.size(), ms);

    return true;
}

// ─── Tag Data Loading ────────────────────────────────────────────────

static bool loadTagData(const std::string& path) {
    fprintf(stderr, "Loading tag data from %s...\n", path.c_str());

    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "WARNING: No tag data file found (tags filter disabled)\n");
        return false;
    }

    json data;
    try {
        data = json::parse(f);
    } catch (const json::parse_error& e) {
        fprintf(stderr, "WARNING: Tag JSON parse error: %s\n", e.what());
        return false;
    }

    // Load tag names
    if (data.contains("tag_index") && data["tag_index"].is_array()) {
        for (auto& t : data["tag_index"]) {
            gTagNames.push_back(t.get<std::string>());
        }
    }

    // Load tag counts
    if (data.contains("tag_counts") && data["tag_counts"].is_array()) {
        for (auto& c : data["tag_counts"]) {
            gTagCounts.push_back(c.get<int>());
        }
    }

    // Initialize all tags as inactive
    gTagActive.resize(gTagNames.size(), false);

    // Load per-sample tags
    int tagged = 0;
    if (data.contains("sample_tags") && data["sample_tags"].is_object()) {
        for (auto& [key, val] : data["sample_tags"].items()) {
            int cosmosId = std::stoi(key);
            auto it = gIdToIndex.find(cosmosId);
            if (it != gIdToIndex.end() && val.is_array()) {
                for (auto& idx : val) {
                    gPoints[it->second].tagIndices.push_back(idx.get<int>());
                }
                tagged++;
            }
        }
    }

    fprintf(stderr, "Loaded %zu tags for %d samples\n", gTagNames.size(), tagged);
    return true;
}

// ─── Animation Update ────────────────────────────────────────────────

static void startHeadAnimation(int newHead) {
    for (size_t i = 0; i < gPoints.size(); i++) {
        gAnimPoints[i].tgtX = gPoints[i].headX[newHead];
        gAnimPoints[i].tgtY = gPoints[i].headY[newHead];
    }
    gAnimating = true;
}

static void switchToHead(int newHead) {
    if (newHead == gCurrentHead) return;
    gCurrentHead = newHead;
    startHeadAnimation(newHead);
    resetCamera();
    gNeedsRebuild = true;
}

static bool updateAnimation(float dt) {
    if (!gAnimating) return false;

    float decay = 1.0f - expf(-8.0f * dt);
    bool stillMoving = false;

    for (size_t i = 0; i < gAnimPoints.size(); i++) {
        float dx = gAnimPoints[i].tgtX - gAnimPoints[i].curX;
        float dy = gAnimPoints[i].tgtY - gAnimPoints[i].curY;
        if (fabsf(dx) > 0.0001f || fabsf(dy) > 0.0001f) {
            gAnimPoints[i].curX += dx * decay;
            gAnimPoints[i].curY += dy * decay;
            stillMoving = true;
        } else {
            gAnimPoints[i].curX = gAnimPoints[i].tgtX;
            gAnimPoints[i].curY = gAnimPoints[i].tgtY;
        }
    }

    if (!stillMoving) {
        gAnimating = false;
    }
    return stillMoving;
}

// ─── Vertex Buffer Building ──────────────────────────────────────────

static void buildVertexData() {
    gVertices.clear();
    gVertices.reserve(gPoints.size());

    int head = gCurrentHead;
    std::string searchLower;
    if (gSearchBuf[0]) {
        searchLower = gSearchBuf;
        std::transform(searchLower.begin(), searchLower.end(), searchLower.begin(), ::tolower);
    }

    // Count active tags for AND-logic filtering
    gActiveTagCount = 0;
    for (size_t t = 0; t < gTagActive.size(); t++) {
        if (gTagActive[t]) gActiveTagCount++;
    }

    for (size_t i = 0; i < gPoints.size(); i++) {
        const auto& p = gPoints[i];

        if (p.type == 1 && !gShowOneshots) continue;
        if (p.type == 2 && !gShowLoops) continue;
        if (p.familyId >= 0 && p.familyId < NUM_FAMILIES && gFamilyHidden[p.familyId]) continue;

        // Instrument filter
        if (gInstrumentFilter >= 0 && p.instrumentId != gInstrumentFilter) continue;

        // Tag filter (AND logic: sample must have ALL active tags)
        if (gActiveTagCount > 0) {
            bool hasAll = true;
            for (size_t t = 0; t < gTagActive.size() && hasAll; t++) {
                if (!gTagActive[t]) continue;
                bool found = false;
                for (int ti : p.tagIndices) {
                    if (ti == (int)t) { found = true; break; }
                }
                if (!found) hasAll = false;
            }
            if (!hasAll) continue;
        }

        // Search filter
        if (!searchLower.empty()) {
            std::string nameLower = p.name;
            std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
            std::string instLower = p.instrumentName;
            std::transform(instLower.begin(), instLower.end(), instLower.begin(), ::tolower);
            std::string famLower = p.familyName;
            std::transform(famLower.begin(), famLower.end(), famLower.begin(), ::tolower);

            if (nameLower.find(searchLower) == std::string::npos &&
                instLower.find(searchLower) == std::string::npos &&
                famLower.find(searchLower) == std::string::npos) {
                continue;
            }
        }

        PointVertex v;
        // Use animated positions during head transitions
        v.x = gAnimPoints[i].curX;
        v.y = gAnimPoints[i].curY;
        v.familyId = (float)p.familyId;
        v.cosmosId = (float)p.cosmosId;
        v.confidence = p.headC[head];
        gVertices.push_back(v);
    }

    gVisibleCount = (int)gVertices.size();
}

static void uploadVertexData() {
    glBindBuffer(GL_ARRAY_BUFFER, gVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 gVertices.size() * sizeof(PointVertex),
                 gVertices.data(),
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// ─── Camera ──────────────────────────────────────────────────────────

static void resetCamera() {
    int head = gCurrentHead;
    float minX = gHeadBounds[head].minX;
    float maxX = gHeadBounds[head].maxX;
    float minY = gHeadBounds[head].minY;
    float maxY = gHeadBounds[head].maxY;

    float rangeX = (maxX - minX);
    float rangeY = (maxY - minY);
    if (rangeX < 0.001f) rangeX = 1.0f;
    if (rangeY < 0.001f) rangeY = 1.0f;

    float margin = 0.1f;
    float scaleX = (float)gWinWidth / (rangeX * (1.0f + margin * 2.0f));
    float scaleY = (float)gWinHeight / (rangeY * (1.0f + margin * 2.0f));
    gCamZoom = std::min(scaleX, scaleY);

    gCamX = -(minX + rangeX * 0.5f) * gCamZoom + (float)gWinWidth * 0.5f;
    gCamY = -(minY + rangeY * 0.5f) * gCamZoom + (float)gWinHeight * 0.5f;
}

static glm::mat4 buildProjection() {
    glm::mat4 ortho = glm::ortho(0.0f, (float)gWinWidth, (float)gWinHeight, 0.0f, -1.0f, 1.0f);
    glm::mat4 cam = glm::mat4(1.0f);
    cam = glm::translate(cam, glm::vec3(gCamX, gCamY, 0.0f));
    cam = glm::scale(cam, glm::vec3(gCamZoom, gCamZoom, 1.0f));
    return ortho * cam;
}

// ─── FBO Setup for Picking ───────────────────────────────────────────

static void createPickingFBO(int w, int h) {
    if (gPickFBO) {
        glDeleteFramebuffers(1, &gPickFBO);
        glDeleteTextures(1, &gPickColorTex);
        glDeleteRenderbuffers(1, &gPickDepthRB);
    }

    glGenFramebuffers(1, &gPickFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, gPickFBO);

    glGenTextures(1, &gPickColorTex);
    glBindTexture(GL_TEXTURE_2D, gPickColorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gPickColorTex, 0);

    glGenRenderbuffers(1, &gPickDepthRB);
    glBindRenderbuffer(GL_RENDERBUFFER, gPickDepthRB);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, gPickDepthRB);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "ERROR: Picking FBO incomplete: 0x%x\n", status);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// ─── Picking ─────────────────────────────────────────────────────────

static void renderPickingPass() {
    int fbW, fbH;
    glfwGetFramebufferSize(gWindow, &fbW, &fbH);

    glm::mat4 proj = buildProjection();

    glBindFramebuffer(GL_FRAMEBUFFER, gPickFBO);
    glViewport(0, 0, fbW, fbH);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_BLEND);

    glUseProgram(gPickingProgram);
    glUniformMatrix4fv(glGetUniformLocation(gPickingProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(proj));

    float pointSize = std::max(3.0f, std::min(8.0f, 5.0f / powf(gCamZoom * 0.005f, 0.3f)));
    glUniform1f(glGetUniformLocation(gPickingProgram, "uPointSize"), pointSize * gPixelRatio);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glBindVertexArray(gVAO);
    glDrawArrays(GL_POINTS, 0, (GLsizei)gVertices.size());
    glBindVertexArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glEnable(GL_BLEND);
}

static int readPickAtScreenPos(double screenX, double screenY) {
    int fbW, fbH;
    glfwGetFramebufferSize(gWindow, &fbW, &fbH);

    int px = (int)(screenX * gPixelRatio);
    int py = (int)(screenY * gPixelRatio);
    py = fbH - 1 - py;

    if (px < 0 || px >= fbW || py < 0 || py >= fbH) return -1;

    glBindFramebuffer(GL_FRAMEBUFFER, gPickFBO);
    unsigned char pixel[4];
    glReadPixels(px, py, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0) return -1;
    int id = (int)pixel[0] | ((int)pixel[1] << 8) | ((int)pixel[2] << 16);
    return id;
}

// ─── Audio ───────────────────────────────────────────────────────────

static void initAudio() {
    ma_engine_config config = ma_engine_config_init();
    config.channels = 2;
    config.sampleRate = 44100;

    ma_result result = ma_engine_init(&config, &gAudioEngine);
    if (result != MA_SUCCESS) {
        fprintf(stderr, "WARNING: Failed to initialize audio engine (error %d)\n", result);
        return;
    }
    gAudioInitialized = true;
    fprintf(stderr, "Audio engine initialized\n");
}

static void playSound(const std::string& wavPath) {
    if (!gAudioInitialized) return;

    if (gSoundLoaded) {
        ma_sound_uninit(&gCurrentSound);
        gSoundLoaded = false;
    }

    ma_result result = ma_sound_init_from_file(&gAudioEngine, wavPath.c_str(),
        MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_ASYNC, nullptr, nullptr, &gCurrentSound);

    if (result != MA_SUCCESS) {
        fprintf(stderr, "WARNING: Cannot play: %s (error %d)\n", wavPath.c_str(), result);
        return;
    }
    gSoundLoaded = true;
    ma_sound_start(&gCurrentSound);
}

static void stopSound() {
    if (gSoundLoaded) {
        ma_sound_stop(&gCurrentSound);
        ma_sound_uninit(&gCurrentSound);
        gSoundLoaded = false;
    }
}

static void fadeOutSound(float durationSec) {
    if (!gSoundLoaded || !gAudioInitialized) return;
    ma_sound_set_fade_in_milliseconds(&gCurrentSound, -1, 0, (ma_uint64)(durationSec * 1000.0f));
}

// ─── Sample Injection (hot-load via analyze_single.py) ──────────────

static std::string openFileDialog() {
    FILE* fp = popen(
        "osascript -e 'set f to POSIX path of (choose file of type {\"wav\", \"WAV\", \"public.audio\"} "
        "with prompt \"Select audio file to inject\")' 2>/dev/null", "r");
    if (!fp) return "";
    char buf[4096];
    std::string result;
    while (fgets(buf, sizeof(buf), fp)) result += buf;
    pclose(fp);
    // Trim trailing newline
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();
    return result;
}

static void injectSampleFromFile(const std::string& wavPath) {
    gAnalyzeRunning = true;
    gAnalyzeStatus = "Analyzing: " + wavPath.substr(wavPath.rfind('/') + 1);
    fprintf(stderr, "Injecting sample: %s\n", wavPath.c_str());

    // Shell out to analyze_single.py
    char cmd[8192];
    snprintf(cmd, sizeof(cmd), "python3 \"%s\" \"%s\" 2>/dev/null",
             ANALYZE_SCRIPT, wavPath.c_str());

    FILE* fp = popen(cmd, "r");
    if (!fp) {
        gAnalyzeStatus = "ERROR: Failed to run analyze_single.py";
        gAnalyzeRunning = false;
        fprintf(stderr, "ERROR: popen failed for analyze_single.py\n");
        return;
    }

    std::string output;
    char buf[4096];
    while (fgets(buf, sizeof(buf), fp)) output += buf;
    int status = pclose(fp);

    if (status != 0 || output.empty()) {
        gAnalyzeStatus = "ERROR: analyze_single.py failed (exit " + std::to_string(status) + ")";
        gAnalyzeRunning = false;
        fprintf(stderr, "ERROR: analyze_single.py exited with %d\n", status);
        if (!output.empty()) fprintf(stderr, "Output: %s\n", output.c_str());
        return;
    }

    // Parse JSON output
    json result;
    try {
        result = json::parse(output);
    } catch (const json::parse_error& e) {
        gAnalyzeStatus = "ERROR: Bad JSON from analyze_single.py";
        gAnalyzeRunning = false;
        fprintf(stderr, "ERROR: JSON parse: %s\nRaw: %s\n", e.what(), output.c_str());
        return;
    }

    if (result.contains("error")) {
        gAnalyzeStatus = "ERROR: " + result["error"].get<std::string>();
        gAnalyzeRunning = false;
        return;
    }

    // Extract coordinates from galaxy dict
    auto& gal = result["galaxy"];
    GalaxyPoint p;
    p.cosmosId = gNextInjectedId++;
    p.wavPath = wavPath;
    p.type = 1;  // treat injected as oneshot

    // Instrument info
    if (result.contains("instrument")) {
        auto& inst = result["instrument"];
        p.instrumentName = inst.value("name", "Unknown");
        p.familyName = inst.value("family", "Unknown");
    } else {
        p.instrumentName = "Unknown";
        p.familyName = "Unknown";
    }

    // Extract filename for display name
    p.name = wavPath.substr(wavPath.rfind('/') + 1);
    if (p.name.size() > 4) p.name = p.name.substr(0, p.name.size() - 4);  // strip .wav

    // Map heads: crest=0, wet_dry=1, sat_clean=2, centroid=3, by_inst=4
    const char* headKeys[] = { "crest", "wet_dry", "sat_clean", "centroid", "by_inst" };
    for (int h = 0; h < 5; h++) {
        if (gal.contains(headKeys[h])) {
            auto& hd = gal[headKeys[h]];
            p.headX[h] = hd.value("x", 0.0f);
            p.headY[h] = hd.value("y", 0.0f);
            p.headC[h] = hd.value("c", 0.5f);  // by_inst has no c, defaults to 0.5
        } else {
            p.headX[h] = 0.0f;
            p.headY[h] = 0.0f;
            p.headC[h] = 0.5f;
        }
    }

    // Force __Pipeline__ family (index 12) for magenta marker
    p.familyId = 12;

    // Assign instrument ID (reuse existing or add new)
    auto instIt = gInstrumentNameToId.find(p.instrumentName);
    if (instIt != gInstrumentNameToId.end()) {
        p.instrumentId = instIt->second;
    } else {
        p.instrumentId = (int)gInstrumentNames.size();
        gInstrumentNameToId[p.instrumentName] = p.instrumentId;
        gInstrumentNames.push_back(p.instrumentName);
    }

    // Push into data arrays
    size_t idx = gPoints.size();
    gPoints.push_back(std::move(p));
    gIdToIndex[gPoints[idx].cosmosId] = idx;

    // Extend animation array
    AnimPoint ap;
    ap.curX = gPoints[idx].headX[gCurrentHead];
    ap.curY = gPoints[idx].headY[gCurrentHead];
    ap.tgtX = ap.curX;
    ap.tgtY = ap.curY;
    gAnimPoints.push_back(ap);

    // Update head bounds
    for (int h = 0; h < 5; h++) {
        gHeadBounds[h].minX = std::min(gHeadBounds[h].minX, gPoints[idx].headX[h]);
        gHeadBounds[h].maxX = std::max(gHeadBounds[h].maxX, gPoints[idx].headX[h]);
        gHeadBounds[h].minY = std::min(gHeadBounds[h].minY, gPoints[idx].headY[h]);
        gHeadBounds[h].maxY = std::max(gHeadBounds[h].maxY, gPoints[idx].headY[h]);
    }

    // Update family stats for __Pipeline__
    bool foundPipeline = false;
    for (auto& fs : gFamilyStats) {
        if (fs.familyId == 12) { fs.count++; foundPipeline = true; break; }
    }
    if (!foundPipeline) gFamilyStats.push_back({ 12, 1 });

    // Select it and trigger rebuild
    gSelectedId = gPoints[idx].cosmosId;
    gNeedsRebuild = true;
    gShowOneshots = true;  // ensure oneshots visible

    gAnalyzeStatus = "Injected: " + gPoints[idx].name +
                     " (" + gPoints[idx].instrumentName + " / " + gPoints[idx].familyName + ")";
    gAnalyzeRunning = false;

    fprintf(stderr, "Injected sample '%s' as cosmos_id=%d at head[%d] pos (%.2f, %.2f)\n",
            gPoints[idx].name.c_str(), gPoints[idx].cosmosId, gCurrentHead,
            gPoints[idx].headX[gCurrentHead], gPoints[idx].headY[gCurrentHead]);
    fprintf(stderr, "  Instrument: %s / Family: %s\n",
            gPoints[idx].instrumentName.c_str(), gPoints[idx].familyName.c_str());
    for (int h = 0; h < 5; h++) {
        fprintf(stderr, "  %s: (%.3f, %.3f) c=%.3f\n",
                HEAD_NAMES[h], gPoints[idx].headX[h], gPoints[idx].headY[h], gPoints[idx].headC[h]);
    }
}

// ─── Point Cloud Rendering ──────────────────────────────────────────

static void renderPointCloud() {
    int fbW, fbH;
    glfwGetFramebufferSize(gWindow, &fbW, &fbH);
    glViewport(0, 0, fbW, fbH);

    glClearColor(0.024f, 0.024f, 0.047f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    glm::mat4 proj = buildProjection();

    glUseProgram(gVisibleProgram);
    glUniformMatrix4fv(glGetUniformLocation(gVisibleProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(proj));

    float pointSize = std::max(1.5f, std::min(6.0f, 3.0f / powf(gCamZoom * 0.005f, 0.3f)));
    glUniform1f(glGetUniformLocation(gVisibleProgram, "uPointSize"), pointSize * gPixelRatio);

    glUniform1i(glGetUniformLocation(gVisibleProgram, "uHoveredId"), gHoveredId);
    glUniform1i(glGetUniformLocation(gVisibleProgram, "uSelectedId"), gSelectedId);

    GLint colorLoc = glGetUniformLocation(gVisibleProgram, "uFamilyColors");
    for (int i = 0; i < NUM_FAMILIES; i++) {
        glUniform3f(colorLoc + i, FAMILIES[i].r, FAMILIES[i].g, FAMILIES[i].b);
    }

    glBindVertexArray(gVAO);
    glDrawArrays(GL_POINTS, 0, (GLsizei)gVertices.size());
    glBindVertexArray(0);
    glUseProgram(0);
}

// ─── Custom Cursor Rendering ─────────────────────────────────────────

static void initCursorGL() {
    // Inline shaders for cursor glow
    const char* vertSrc = R"(
        #version 410 core
        layout(location = 0) in vec2 aPos;
        layout(location = 1) in vec2 aUV;
        out vec2 vUV;
        uniform mat4 uProjection;
        uniform vec2 uCenter;
        uniform float uSize;
        void main() {
            vec2 pos = uCenter + aPos * uSize;
            gl_Position = uProjection * vec4(pos, 0.0, 1.0);
            vUV = aUV;
        }
    )";

    const char* fragSrc = R"(
        #version 410 core
        in vec2 vUV;
        out vec4 FragColor;
        uniform vec3 uColor;
        uniform float uAlpha;
        uniform float uFalloff;
        void main() {
            vec2 d = vUV - vec2(0.5);
            float dist = length(d);
            float glow = exp(-dist * dist * uFalloff);
            FragColor = vec4(uColor, glow * uAlpha);
        }
    )";

    gCursorProgram = createProgramFromSource(vertSrc, fragSrc);

    // Fullscreen quad vertices: position (xy) + uv
    float verts[] = {
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
    };

    glGenVertexArrays(1, &gCursorVAO);
    glGenBuffers(1, &gCursorVBO);
    glBindVertexArray(gCursorVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gCursorVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);
}

static void renderCursor(double time) {
    if (!gCursorProgram) return;

    // Spring interpolation for ambient glow
    float springRate = 0.08f;
    gCursorSpringX += ((float)gLastMouseX - gCursorSpringX) * springRate;
    gCursorSpringY += ((float)gLastMouseY - gCursorSpringY) * springRate;

    float pulse = 1.0f + sinf((float)time * 2.0f) * 0.15f;

    glm::mat4 ortho = glm::ortho(0.0f, (float)gWinWidth, (float)gWinHeight, 0.0f, -1.0f, 1.0f);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);  // Additive blending for glow

    glUseProgram(gCursorProgram);
    glUniformMatrix4fv(glGetUniformLocation(gCursorProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(ortho));
    glBindVertexArray(gCursorVAO);

    // Layer 1: Large ambient glow (lagging, soft)
    glUniform2f(glGetUniformLocation(gCursorProgram, "uCenter"), gCursorSpringX, gCursorSpringY);
    glUniform1f(glGetUniformLocation(gCursorProgram, "uSize"), 80.0f * pulse);
    glUniform3f(glGetUniformLocation(gCursorProgram, "uColor"), 0.40f, 0.22f, 0.65f);  // Purple
    glUniform1f(glGetUniformLocation(gCursorProgram, "uAlpha"), 0.12f);
    glUniform1f(glGetUniformLocation(gCursorProgram, "uFalloff"), 6.0f);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // Layer 2: Medium focused beam (at cursor)
    glUniform2f(glGetUniformLocation(gCursorProgram, "uCenter"), (float)gLastMouseX, (float)gLastMouseY);
    glUniform1f(glGetUniformLocation(gCursorProgram, "uSize"), 30.0f * pulse);
    glUniform3f(glGetUniformLocation(gCursorProgram, "uColor"), 0.50f, 0.30f, 0.90f);  // Bright purple
    glUniform1f(glGetUniformLocation(gCursorProgram, "uAlpha"), 0.20f);
    glUniform1f(glGetUniformLocation(gCursorProgram, "uFalloff"), 10.0f);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // Layer 3: Tiny hotspot core
    glUniform2f(glGetUniformLocation(gCursorProgram, "uCenter"), (float)gLastMouseX, (float)gLastMouseY);
    glUniform1f(glGetUniformLocation(gCursorProgram, "uSize"), 6.0f);
    glUniform3f(glGetUniformLocation(gCursorProgram, "uColor"), 0.75f, 0.60f, 1.0f);  // Light purple-white
    glUniform1f(glGetUniformLocation(gCursorProgram, "uAlpha"), 0.7f);
    glUniform1f(glGetUniformLocation(gCursorProgram, "uFalloff"), 8.0f);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glBindVertexArray(0);
    glUseProgram(0);

    // Reset blending for subsequent passes
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

// ─── ImGui UI ────────────────────────────────────────────────────────

static void applyDarkCosmosTheme() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Window
    style.WindowRounding = 8.0f;
    style.FrameRounding = 5.0f;
    style.GrabRounding = 4.0f;
    style.PopupRounding = 6.0f;
    style.ScrollbarRounding = 4.0f;
    style.TabRounding = 4.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.WindowPadding = ImVec2(14, 12);
    style.FramePadding = ImVec2(8, 4);
    style.ItemSpacing = ImVec2(8, 6);

    // Deep space background
    colors[ImGuiCol_WindowBg]          = ImVec4(0.040f, 0.040f, 0.070f, 0.92f);
    colors[ImGuiCol_PopupBg]           = ImVec4(0.040f, 0.040f, 0.070f, 0.95f);
    colors[ImGuiCol_Border]            = ImVec4(1.000f, 1.000f, 1.000f, 0.06f);
    colors[ImGuiCol_FrameBg]           = ImVec4(1.000f, 1.000f, 1.000f, 0.04f);
    colors[ImGuiCol_FrameBgHovered]    = ImVec4(1.000f, 1.000f, 1.000f, 0.06f);
    colors[ImGuiCol_FrameBgActive]     = ImVec4(1.000f, 1.000f, 1.000f, 0.08f);
    colors[ImGuiCol_TitleBg]           = ImVec4(0.024f, 0.024f, 0.047f, 0.90f);
    colors[ImGuiCol_TitleBgActive]     = ImVec4(0.024f, 0.024f, 0.047f, 0.95f);
    colors[ImGuiCol_TitleBgCollapsed]  = ImVec4(0.024f, 0.024f, 0.047f, 0.60f);
    colors[ImGuiCol_ScrollbarBg]       = ImVec4(0.000f, 0.000f, 0.000f, 0.00f);
    colors[ImGuiCol_ScrollbarGrab]     = ImVec4(1.000f, 1.000f, 1.000f, 0.08f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(1.000f, 1.000f, 1.000f, 0.12f);
    colors[ImGuiCol_ScrollbarGrabActive]  = ImVec4(1.000f, 1.000f, 1.000f, 0.16f);
    colors[ImGuiCol_CheckMark]         = ImVec4(0.500f, 0.700f, 1.000f, 0.90f);
    colors[ImGuiCol_Button]            = ImVec4(1.000f, 1.000f, 1.000f, 0.06f);
    colors[ImGuiCol_ButtonHovered]     = ImVec4(1.000f, 1.000f, 1.000f, 0.10f);
    colors[ImGuiCol_ButtonActive]      = ImVec4(1.000f, 1.000f, 1.000f, 0.14f);
    colors[ImGuiCol_Header]            = ImVec4(1.000f, 1.000f, 1.000f, 0.06f);
    colors[ImGuiCol_HeaderHovered]     = ImVec4(1.000f, 1.000f, 1.000f, 0.08f);
    colors[ImGuiCol_HeaderActive]      = ImVec4(1.000f, 1.000f, 1.000f, 0.10f);
    colors[ImGuiCol_Tab]               = ImVec4(1.000f, 1.000f, 1.000f, 0.04f);
    colors[ImGuiCol_TabHovered]        = ImVec4(1.000f, 1.000f, 1.000f, 0.10f);
    colors[ImGuiCol_TabSelected]       = ImVec4(1.000f, 1.000f, 1.000f, 0.12f);
    colors[ImGuiCol_Text]              = ImVec4(0.780f, 0.800f, 0.830f, 1.00f);
    colors[ImGuiCol_TextDisabled]      = ImVec4(0.310f, 0.345f, 0.380f, 1.00f);
    colors[ImGuiCol_Separator]         = ImVec4(1.000f, 1.000f, 1.000f, 0.06f);
    colors[ImGuiCol_ResizeGrip]        = ImVec4(0.000f, 0.000f, 0.000f, 0.00f);
}

static void renderImGuiUI() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    float topH = 48.0f;
    float filterH = 42.0f;

    // ─── Top Bar ─────────────────────────────────────────────────
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2((float)gWinWidth, topH));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20, 12));
    ImGui::Begin("##topbar", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse);

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.53f, 0.56f, 0.63f, 1.0f));
    ImGui::Text("COSMOS GALAXY");
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.31f, 0.34f, 0.41f, 1.0f));
    ImGui::Text("/ %zu samples", gPoints.size());
    ImGui::PopStyleColor();

    // Add File button
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1.0f, 0.0f, 1.0f, 0.25f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.0f, 1.0f, 0.45f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.0f, 0.0f, 1.0f, 0.65f));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.6f, 1.0f, 1.0f));
    if (ImGui::Button("+ Add File##inject") && !gAnalyzeRunning) {
        std::string path = openFileDialog();
        if (!path.empty()) {
            injectSampleFromFile(path);
        }
    }
    ImGui::PopStyleColor(4);

    // Injection status
    if (!gAnalyzeStatus.empty()) {
        ImGui::SameLine();
        bool isError = gAnalyzeStatus.find("ERROR") != std::string::npos;
        if (isError)
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
        else if (gAnalyzeRunning)
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.4f, 1.0f));
        else
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 1.0f, 0.6f, 1.0f));
        ImGui::Text("%s", gAnalyzeStatus.c_str());
        ImGui::PopStyleColor();
    }

    // Search box (right-aligned)
    float searchWidth = 200.0f;
    ImGui::SameLine(ImGui::GetWindowWidth() - searchWidth - 220.0f);
    ImGui::PushItemWidth(searchWidth);
    if (ImGui::InputTextWithHint("##search", "Search samples...", gSearchBuf, sizeof(gSearchBuf))) {
        gNeedsRebuild = true;
    }
    ImGui::PopItemWidth();

    // Stats
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.31f, 0.34f, 0.41f, 1.0f));
    ImGui::Text("%d visible  %.0f FPS", gVisibleCount, gFps);
    ImGui::PopStyleColor();

    ImGui::End();
    ImGui::PopStyleVar();

    // ─── Filter / Head Bar ───────────────────────────────────────
    ImGui::SetNextWindowPos(ImVec2(0, topH));
    ImGui::SetNextWindowSize(ImVec2((float)gWinWidth, filterH));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20, 8));
    ImGui::Begin("##filterbar", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse);

    // One-shots toggle
    {
        if (gShowOneshots) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.165f, 0.631f, 0.455f, 0.3f));
        else ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1.0f, 1.0f, 1.0f, 0.04f));
        if (ImGui::Button("One-shots")) {
            gShowOneshots = !gShowOneshots;
            gNeedsRebuild = true;
        }
        ImGui::PopStyleColor();
    }
    ImGui::SameLine();

    // Loops toggle
    {
        if (gShowLoops) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.165f, 0.631f, 0.455f, 0.3f));
        else ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1.0f, 1.0f, 1.0f, 0.04f));
        if (ImGui::Button("Loops")) {
            gShowLoops = !gShowLoops;
            gNeedsRebuild = true;
        }
        ImGui::PopStyleColor();
    }

    // Head tabs (centered)
    float headTabsWidth = 5 * 110.0f;
    float centerX = ((float)gWinWidth - headTabsWidth) * 0.5f;
    ImGui::SameLine(centerX);

    for (int h = 0; h < 5; h++) {
        if (h > 0) ImGui::SameLine();
        bool active = (gCurrentHead == h);
        if (active) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1.0f, 1.0f, 1.0f, 0.12f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.88f, 0.90f, 0.93f, 1.0f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f, 0.48f, 0.55f, 1.0f));
        }
        char label[64];
        snprintf(label, sizeof(label), "%s##head%d", HEAD_NAMES[h], h);
        if (ImGui::Button(label, ImVec2(100.0f, 26.0f))) {
            switchToHead(h);
        }
        ImGui::PopStyleColor(2);
    }

    // Instrument dropdown (right of head tabs)
    ImGui::SameLine((float)gWinWidth - 360.0f);
    ImGui::PushItemWidth(140.0f);
    {
        const char* preview = (gInstrumentFilter < 0) ? "All Instruments" :
                              gInstrumentNames[gInstrumentFilter].c_str();
        if (preview[0] == '\0') preview = "(unnamed)";
        ImGui::SetNextWindowSizeConstraints(ImVec2(200, 0), ImVec2(300, 400));
        if (ImGui::BeginCombo("##instfilter", preview, ImGuiComboFlags_None)) {
            if (ImGui::Selectable("All Instruments##inst_all", gInstrumentFilter < 0)) {
                gInstrumentFilter = -1;
                gNeedsRebuild = true;
            }
            ImGui::Separator();
            for (int i = 0; i < (int)gInstrumentNames.size(); i++) {
                bool selected = (gInstrumentFilter == i);
                char instLabel[128];
                const char* iname = gInstrumentNames[i].c_str();
                if (iname[0] == '\0') iname = "(unnamed)";
                snprintf(instLabel, sizeof(instLabel), "%s##inst%d", iname, i);
                if (ImGui::Selectable(instLabel, selected)) {
                    gInstrumentFilter = i;
                    gNeedsRebuild = true;
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    }
    ImGui::PopItemWidth();

    // Tags dropdown (multi-select, AND logic)
    if (!gTagNames.empty()) {
        ImGui::SameLine();
        ImGui::PushItemWidth(140.0f);
        {
            char tagPreview[64];
            if (gActiveTagCount == 0) {
                snprintf(tagPreview, sizeof(tagPreview), "All Tags");
            } else {
                snprintf(tagPreview, sizeof(tagPreview), "%d tag%s active",
                         gActiveTagCount, gActiveTagCount > 1 ? "s" : "");
            }
            ImGui::SetNextWindowSizeConstraints(ImVec2(200, 0), ImVec2(300, 400));
            if (ImGui::BeginCombo("##tagfilter", tagPreview, ImGuiComboFlags_None)) {
                // Clear all button
                if (gActiveTagCount > 0) {
                    if (ImGui::Selectable("Clear All Tags", false)) {
                        for (size_t t = 0; t < gTagActive.size(); t++) gTagActive[t] = false;
                        gActiveTagCount = 0;
                        gNeedsRebuild = true;
                    }
                    ImGui::Separator();
                }
                // Build sorted indices once
                static std::vector<int> sortedTagIndices;
                if (sortedTagIndices.empty() && !gTagNames.empty()) {
                    sortedTagIndices.resize(gTagNames.size());
                    for (int t = 0; t < (int)gTagNames.size(); t++) sortedTagIndices[t] = t;
                    std::sort(sortedTagIndices.begin(), sortedTagIndices.end(),
                        [](int a, int b) {
                            int ca = (a < (int)gTagCounts.size()) ? gTagCounts[a] : 0;
                            int cb = (b < (int)gTagCounts.size()) ? gTagCounts[b] : 0;
                            return ca > cb;
                        });
                }
                // Render tags with clipper for performance
                ImGuiListClipper clipper;
                clipper.Begin((int)sortedTagIndices.size());
                while (clipper.Step()) {
                    for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                        int idx = sortedTagIndices[row];
                        if (idx < 0 || idx >= (int)gTagNames.size()) continue;
                        char tagLabel[128];
                        int cnt = (idx < (int)gTagCounts.size()) ? gTagCounts[idx] : 0;
                        const char* name = gTagNames[idx].c_str();
                        if (name[0] == '\0') name = "(unnamed)";
                        snprintf(tagLabel, sizeof(tagLabel), "%s (%d)##tag%d", name, cnt, idx);
                        bool active = (idx < (int)gTagActive.size()) && gTagActive[idx];
                        if (ImGui::Selectable(tagLabel, active)) {
                            if (idx < (int)gTagActive.size()) {
                                gTagActive[idx] = !gTagActive[idx];
                                gNeedsRebuild = true;
                            }
                        }
                    }
                }
                ImGui::EndCombo();
            }
        }
        ImGui::PopItemWidth();
    }

    ImGui::End();
    ImGui::PopStyleVar();

    // ─── Legend Panel (right side) ───────────────────────────────
    float legendW = 200.0f;
    float legendX = (float)gWinWidth - legendW - 16.0f;
    float legendY = topH + filterH + 16.0f;

    ImGui::SetNextWindowPos(ImVec2(legendX, legendY), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(legendW, 0.0f));  // auto height
    ImGui::Begin("##legend", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoScrollbar);

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.31f, 0.34f, 0.41f, 1.0f));
    ImGui::Text("FAMILIES");
    ImGui::PopStyleColor();
    ImGui::Spacing();

    for (auto& fs : gFamilyStats) {
        int fid = fs.familyId;
        const FamilyInfo& fi = FAMILIES[fid];
        bool hidden = gFamilyHidden[fid];

        float alpha = hidden ? 0.2f : 1.0f;

        // Colored dot
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImU32 dotColor = IM_COL32(
            (int)(fi.r * 255 * alpha), (int)(fi.g * 255 * alpha),
            (int)(fi.b * 255 * alpha), 255);
        dl->AddCircleFilled(ImVec2(p.x + 6.0f, p.y + 8.0f), 4.0f, dotColor);

        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 18.0f);

        // Clickable row
        char famLabel[128];
        snprintf(famLabel, sizeof(famLabel), "%-14s %6d##fam%d", fi.name, fs.count, fid);

        ImGui::PushStyleColor(ImGuiCol_Text,
            hidden ? ImVec4(0.31f, 0.34f, 0.41f, 0.3f) : ImVec4(0.53f, 0.56f, 0.63f, 1.0f));
        if (ImGui::Selectable(famLabel, false, 0, ImVec2(legendW - 40.0f, 0.0f))) {
            gFamilyHidden[fid] = !gFamilyHidden[fid];
            gNeedsRebuild = true;
        }
        ImGui::PopStyleColor();
    }

    ImGui::End();

    // ─── Detail Panel (bottom-left, shown when selected) ─────────
    if (gSelectedId > 0) {
        auto it = gIdToIndex.find(gSelectedId);
        if (it != gIdToIndex.end()) {
            const GalaxyPoint& p = gPoints[it->second];
            const FamilyInfo& fi = FAMILIES[p.familyId];

            float detailW = 380.0f;
            ImGui::SetNextWindowPos(ImVec2(16.0f, (float)gWinHeight - 16.0f), ImGuiCond_Always, ImVec2(0.0f, 1.0f));
            ImGui::SetNextWindowSize(ImVec2(detailW, 0.0f));
            ImGui::Begin("##detail", nullptr,
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing);

            // Close button
            float closeX = ImGui::GetWindowWidth() - 28.0f;
            ImGui::SameLine(closeX);
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
            if (ImGui::SmallButton("X##close")) {
                gSelectedId = -1;
                stopSound();
            }
            ImGui::PopStyleColor();

            // Name
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 18.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.88f, 0.90f, 0.93f, 1.0f));
            ImGui::TextWrapped("%s", p.name.c_str());
            ImGui::PopStyleColor();

            // Short path
            std::string shortPath = p.wavPath;
            size_t slashCount = 0;
            for (auto rit = shortPath.rbegin(); rit != shortPath.rend(); ++rit) {
                if (*rit == '/') {
                    slashCount++;
                    if (slashCount == 3) {
                        shortPath = ".../" + shortPath.substr(std::distance(shortPath.begin(), rit.base()));
                        break;
                    }
                }
            }
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.23f, 0.25f, 0.31f, 1.0f));
            ImGui::TextWrapped("%s", shortPath.c_str());
            ImGui::PopStyleColor();

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Info grid
            auto row = [](const char* key, const char* fmt, ...) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.31f, 0.34f, 0.41f, 1.0f));
                ImGui::Text("%-12s", key);
                ImGui::PopStyleColor();
                ImGui::SameLine(110.0f);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.53f, 0.56f, 0.63f, 1.0f));
                char buf[256];
                va_list args;
                va_start(args, fmt);
                vsnprintf(buf, sizeof(buf), fmt, args);
                va_end(args);
                ImGui::Text("%s", buf);
                ImGui::PopStyleColor();
            };

            // Instrument with colored indicator
            ImVec2 instPos = ImGui::GetCursorScreenPos();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddCircleFilled(ImVec2(instPos.x + 4.0f, instPos.y + 8.0f), 3.0f,
                IM_COL32((int)(fi.r*255), (int)(fi.g*255), (int)(fi.b*255), 255));
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 14.0f);

            row("Instrument", "%s", p.instrumentName.c_str());
            row("Family", "%s", p.familyName.empty() ? "Unknown" : p.familyName.c_str());
            row("Type", "%s", p.type == 1 ? "One-shot" : "Loop");
            row("Cosmos ID", "%d", p.cosmosId);

            ImGui::Spacing();

            // Head coordinates
            const char* shortHeadNames[] = { "Crest", "Wet/Dry", "Sat/Clean", "Centroid", "Instrument" };
            for (int h = 0; h < 5; h++) {
                char coordBuf[64];
                if (h < 4) {
                    snprintf(coordBuf, sizeof(coordBuf), "(%.3f, %.3f) c=%.3f", p.headX[h], p.headY[h], p.headC[h]);
                } else {
                    snprintf(coordBuf, sizeof(coordBuf), "(%.3f, %.3f)", p.headX[h], p.headY[h]);
                }
                row(shortHeadNames[h], "%s", coordBuf);
            }

            // Tags
            if (!p.tagIndices.empty() && !gTagNames.empty()) {
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();

                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.31f, 0.34f, 0.41f, 1.0f));
                ImGui::Text("Tags");
                ImGui::PopStyleColor();

                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f, 0.48f, 0.55f, 1.0f));
                std::string tagStr;
                for (size_t ti = 0; ti < p.tagIndices.size(); ti++) {
                    int idx = p.tagIndices[ti];
                    if (idx >= 0 && idx < (int)gTagNames.size()) {
                        if (!tagStr.empty()) tagStr += ", ";
                        tagStr += gTagNames[idx];
                    }
                }
                ImGui::TextWrapped("%s", tagStr.c_str());
                ImGui::PopStyleColor();
            }

            ImGui::End();
        }
    }

    // ─── Tooltip (near cursor when hovering) ─────────────────────
    if (gHoveredId > 0 && gDragMode == DRAG_NONE) {
        auto it = gIdToIndex.find(gHoveredId);
        if (it != gIdToIndex.end()) {
            const GalaxyPoint& p = gPoints[it->second];
            const FamilyInfo& fi = FAMILIES[p.familyId];

            ImGui::SetNextWindowPos(ImVec2((float)gLastMouseX + 14.0f, (float)gLastMouseY + 14.0f));
            ImGui::SetNextWindowSize(ImVec2(0, 0));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12, 10));
            ImGui::Begin("##tooltip", nullptr,
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing |
                ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoScrollbar);

            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.88f, 0.90f, 0.93f, 1.0f));
            ImGui::Text("%s", p.name.c_str());
            ImGui::PopStyleColor();

            // Instrument badge with color
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(fi.r, fi.g, fi.b, 1.0f));
            ImGui::Text("%s", p.instrumentName.c_str());
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.31f, 0.34f, 0.41f, 1.0f));
            ImGui::Text("/ %s", p.familyName.empty() ? "Unknown" : p.familyName.c_str());
            ImGui::PopStyleColor();

            // Coordinates
            int head = gCurrentHead;
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.31f, 0.34f, 0.41f, 1.0f));
            ImGui::Text("x: %.3f  y: %.3f  %s",
                         p.headX[head], p.headY[head],
                         p.type == 1 ? "one-shot" : "loop");
            ImGui::PopStyleColor();

            ImGui::End();
            ImGui::PopStyleVar();
        }
    }

    // ─── Audio indicator (bottom-left, above detail) ─────────────
    if (gSoundLoaded) {
        float indicatorY = gSelectedId > 0 ? (float)gWinHeight - 200.0f : (float)gWinHeight - 36.0f;
        ImGui::SetNextWindowPos(ImVec2(16.0f, indicatorY));
        ImGui::SetNextWindowSize(ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 5));
        ImGui::Begin("##audio", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoInputs);

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.165f, 0.631f, 0.455f, 1.0f));
        if (gDragMode == DRAG_SCRUB) {
            ImGui::Text(">> scrub");
        } else {
            ImGui::Text(">> playing");
        }
        ImGui::PopStyleColor();

        ImGui::End();
        ImGui::PopStyleVar();
    }

    // ─── Help hint (bottom-right) ────────────────────────────────
    {
        ImGui::SetNextWindowPos(ImVec2((float)gWinWidth - 16.0f, (float)gWinHeight - 16.0f), ImGuiCond_Always, ImVec2(1.0f, 1.0f));
        ImGui::SetNextWindowSize(ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 4));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
        ImGui::Begin("##help", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoScrollbar);

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.15f, 0.16f, 0.19f, 1.0f));
        ImGui::Text("Scroll zoom   Drag pan   Click play   1-5 heads   H help   Esc reset");
        ImGui::PopStyleColor();

        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);
    }

    // ─── Head description subtitle ──────────────────────────────
    {
        ImGui::SetNextWindowPos(ImVec2((float)gWinWidth * 0.5f, topH + filterH + 8.0f), ImGuiCond_Always, ImVec2(0.5f, 0.0f));
        ImGui::SetNextWindowSize(ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
        ImGui::Begin("##headlabel", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoScrollbar);

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.23f, 0.25f, 0.31f, 1.0f));
        ImGui::Text("%s", HEAD_DESCS[gCurrentHead]);
        ImGui::PopStyleColor();

        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

// ─── GLFW Callbacks ──────────────────────────────────────────────────

static void onMouseButton(GLFWwindow* window, int button, int action, int mods) {
    // Let ImGui handle if it wants the input
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);

        if (action == GLFW_PRESS) {
            gDragStartX = mx;
            gDragStartY = my;
            gDragStartMouseX = mx;
            gDragStartMouseY = my;
            gDragCamX = gCamX;
            gDragCamY = gCamY;

            // Check if clicking on a point -> scrub mode, else -> pan mode
            int id = readPickAtScreenPos(mx, my);
            if (id > 0) {
                gDragMode = DRAG_SCRUB;
                gDragLastPlayedId = id;
                gHoveredId = id;
                gSelectedId = id;

                auto it = gIdToIndex.find(id);
                if (it != gIdToIndex.end()) {
                    const GalaxyPoint& p = gPoints[it->second];
                    if (!p.wavPath.empty()) {
                        playSound(p.wavPath);
                    }
                }
            } else {
                gDragMode = DRAG_PAN;
            }
        } else if (action == GLFW_RELEASE) {
            double dist = fabs(mx - gDragStartMouseX) + fabs(my - gDragStartMouseY);

            if (gDragMode == DRAG_SCRUB) {
                fadeOutSound(0.25f);
            } else if (gDragMode == DRAG_PAN && dist < 5.0) {
                // Click on empty space -> deselect
                int id = readPickAtScreenPos(mx, my);
                if (id > 0) {
                    gSelectedId = id;
                    auto it = gIdToIndex.find(id);
                    if (it != gIdToIndex.end()) {
                        const GalaxyPoint& p = gPoints[it->second];
                        if (!p.wavPath.empty()) playSound(p.wavPath);
                    }
                } else {
                    gSelectedId = -1;
                    stopSound();
                }
            }

            gDragMode = DRAG_NONE;
            gDragLastPlayedId = -1;
        }
    }
}

static void onCursorPos(GLFWwindow* window, double mx, double my) {
    ImGuiIO& io = ImGui::GetIO();

    gLastMouseX = mx;
    gLastMouseY = my;

    if (io.WantCaptureMouse) return;

    if (gDragMode == DRAG_PAN) {
        gCamX = gDragCamX + (float)(mx - gDragStartX);
        gCamY = gDragCamY + (float)(my - gDragStartY);
        return;
    }

    if (gDragMode == DRAG_SCRUB) {
        // Scrub: play each new point the cursor crosses
        int id = readPickAtScreenPos(mx, my);
        if (id > 0 && id != gDragLastPlayedId) {
            gDragLastPlayedId = id;
            gHoveredId = id;
            gSelectedId = id;

            auto it = gIdToIndex.find(id);
            if (it != gIdToIndex.end()) {
                const GalaxyPoint& p = gPoints[it->second];
                if (!p.wavPath.empty()) {
                    playSound(p.wavPath);
                }
            }
        }
        return;
    }
}

static void onScroll(GLFWwindow* window, double xoff, double yoff) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    double mx, my;
    glfwGetCursorPos(window, &mx, &my);

    float factor = (yoff > 0) ? 1.1f : 0.9f;
    float newZoom = gCamZoom * factor;

    gCamX = (float)mx - ((float)mx - gCamX) * (newZoom / gCamZoom);
    gCamY = (float)my - ((float)my - gCamY) * (newZoom / gCamZoom);
    gCamZoom = newZoom;
}

static void onKey(GLFWwindow* window, int key, int scancode, int action, int mods) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard) return;

    if (action != GLFW_PRESS) return;

    if (key == GLFW_KEY_ESCAPE) {
        gSelectedId = -1;
        gHoveredId = -1;
        stopSound();
        gSearchBuf[0] = '\0';
        gInstrumentFilter = -1;
        for (size_t t = 0; t < gTagActive.size(); t++) gTagActive[t] = false;
        resetCamera();
        gNeedsRebuild = true;
        return;
    }

    if (key >= GLFW_KEY_1 && key <= GLFW_KEY_5) {
        switchToHead(key - GLFW_KEY_1);
        return;
    }

    if (key == GLFW_KEY_O) {
        gShowOneshots = !gShowOneshots;
        gNeedsRebuild = true;
        return;
    }
    if (key == GLFW_KEY_L) {
        gShowLoops = !gShowLoops;
        gNeedsRebuild = true;
        return;
    }

    if (key == GLFW_KEY_Q) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        return;
    }

    if (key == GLFW_KEY_H) {
        fprintf(stderr, "\n=== COSMOS Galaxy Explorer - Controls ===\n");
        fprintf(stderr, "  Mouse drag bg : Pan camera\n");
        fprintf(stderr, "  Drag on point : Scrub (play samples sequentially)\n");
        fprintf(stderr, "  Scroll wheel  : Zoom\n");
        fprintf(stderr, "  Click point   : Select & play audio\n");
        fprintf(stderr, "  1-5           : Switch VAE head view\n");
        fprintf(stderr, "  O             : Toggle one-shots\n");
        fprintf(stderr, "  L             : Toggle loops\n");
        fprintf(stderr, "  Escape        : Reset view & clear search\n");
        fprintf(stderr, "  H             : Show this help\n");
        fprintf(stderr, "  Q             : Quit\n");
        fprintf(stderr, "=========================================\n\n");
        return;
    }
}

static void onFramebufferResize(GLFWwindow* window, int w, int h) {
    gWinWidth = (int)(w / gPixelRatio);
    gWinHeight = (int)(h / gPixelRatio);
    createPickingFBO(w, h);
}

static void onWindowResize(GLFWwindow* window, int w, int h) {
    gWinWidth = w;
    gWinHeight = h;
}

// ─── Initialization ──────────────────────────────────────────────────

static bool initGL() {
    gVisibleProgram = createProgram("shaders/pointcloud.vert", "shaders/pointcloud.frag");
    gPickingProgram = createProgram("shaders/picking.vert", "shaders/picking.frag");
    if (!gVisibleProgram || !gPickingProgram) return false;

    glGenVertexArrays(1, &gVAO);
    glGenBuffers(1, &gVBO);

    glBindVertexArray(gVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gVBO);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(PointVertex), (void*)offsetof(PointVertex, x));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(PointVertex), (void*)offsetof(PointVertex, familyId));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(PointVertex), (void*)offsetof(PointVertex, cosmosId));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(PointVertex), (void*)offsetof(PointVertex, confidence));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    int fbW, fbH;
    glfwGetFramebufferSize(gWindow, &fbW, &fbH);
    createPickingFBO(fbW, fbH);

    // Cursor glow
    initCursorGL();

    return true;
}

// ─── Main ────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    fprintf(stderr, "COSMOS Galaxy Explorer (C++/OpenGL)\n");
    fprintf(stderr, "===================================\n\n");

    if (!glfwInit()) {
        fprintf(stderr, "ERROR: Failed to initialize GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    gWindow = glfwCreateWindow(gWinWidth, gWinHeight, "COSMOS Galaxy Explorer", nullptr, nullptr);
    if (!gWindow) {
        fprintf(stderr, "ERROR: Failed to create GLFW window\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(gWindow);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        fprintf(stderr, "ERROR: GLEW init failed: %s\n", glewGetErrorString(glewErr));
        return 1;
    }
    while (glGetError() != GL_NO_ERROR) {}

    fprintf(stderr, "OpenGL: %s\n", glGetString(GL_VERSION));
    fprintf(stderr, "GLSL:   %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    fprintf(stderr, "GPU:    %s\n\n", glGetString(GL_RENDERER));

    int fbW, fbH, winW, winH;
    glfwGetFramebufferSize(gWindow, &fbW, &fbH);
    glfwGetWindowSize(gWindow, &winW, &winH);
    gPixelRatio = (float)fbW / (float)winW;
    gWinWidth = winW;
    gWinHeight = winH;
    fprintf(stderr, "Window: %dx%d (pixel ratio: %.1f)\n\n", winW, winH, gPixelRatio);

    // Load data
    if (!loadGalaxyData("data/galaxy_viz_data.json")) {
        fprintf(stderr, "ERROR: Failed to load galaxy data\n");
        glfwTerminate();
        return 1;
    }

    // Load tags (optional, non-fatal if missing)
    loadTagData("data/galaxy_tags.json");

    // Init OpenGL resources
    if (!initGL()) {
        fprintf(stderr, "ERROR: Failed to initialize OpenGL resources\n");
        glfwTerminate();
        return 1;
    }

    // Init ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr;  // No imgui.ini

    applyDarkCosmosTheme();

    // Install our callbacks FIRST, then ImGui wraps them and chains back to ours
    glfwSetMouseButtonCallback(gWindow, onMouseButton);
    glfwSetCursorPosCallback(gWindow, onCursorPos);
    glfwSetScrollCallback(gWindow, onScroll);
    glfwSetKeyCallback(gWindow, onKey);
    glfwSetFramebufferSizeCallback(gWindow, onFramebufferResize);
    glfwSetWindowSizeCallback(gWindow, onWindowResize);

    // ImGui init with install_callbacks=true: wraps our callbacks, chains to them
    ImGui_ImplGlfw_InitForOpenGL(gWindow, true);
    ImGui_ImplOpenGL3_Init("#version 410");

    // Init audio
    initAudio();

    // Build initial data
    buildVertexData();
    uploadVertexData();
    resetCamera();

    // Hide system cursor
    glfwSetInputMode(gWindow, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

    // Initialize cursor spring to center
    gCursorSpringX = (float)gWinWidth * 0.5f;
    gCursorSpringY = (float)gWinHeight * 0.5f;

    fprintf(stderr, "Press H for controls help\n\n");

    gLastFrameTime = glfwGetTime();
    gFpsTimer = gLastFrameTime;

    // Main loop
    while (!glfwWindowShouldClose(gWindow)) {
        double now = glfwGetTime();
        float dt = (float)(now - gLastFrameTime);
        if (dt > 0.1f) dt = 0.1f;  // clamp large dt (e.g. during window drag)

        glfwPollEvents();

        // Update animation
        bool animMoved = updateAnimation(dt);
        if (animMoved) {
            gNeedsRebuild = true;
        }

        // Rebuild vertex data if filters/head changed or animation tick
        if (gNeedsRebuild) {
            buildVertexData();
            uploadVertexData();
            gNeedsRebuild = false;
        }

        // Render point cloud
        renderPointCloud();

        // Render picking pass
        renderPickingPass();

        // Render custom cursor
        renderCursor(glfwGetTime());

        // Throttled hover detection
        now = glfwGetTime();
        if (gDragMode == DRAG_NONE && (now - gLastHoverPickTime >= HOVER_PICK_INTERVAL)) {
            gLastHoverPickTime = now;
            ImGuiIO& io = ImGui::GetIO();
            if (!io.WantCaptureMouse) {
                int id = readPickAtScreenPos(gLastMouseX, gLastMouseY);
                if (id > 0 && id != gHoveredId) {
                    gHoveredId = id;
                } else if (id <= 0 && gHoveredId != -1) {
                    gHoveredId = -1;
                }
            }
        }

        // Render ImGui overlay
        renderImGuiUI();

        glfwSwapBuffers(gWindow);

        // FPS
        gFrameCount++;
        if (now - gFpsTimer >= 1.0) {
            gFps = (float)gFrameCount / (float)(now - gFpsTimer);
            gFrameCount = 0;
            gFpsTimer = now;
        }
        gLastFrameTime = now;
    }

    // Cleanup
    stopSound();
    if (gAudioInitialized) ma_engine_uninit(&gAudioEngine);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glDeleteVertexArrays(1, &gVAO);
    glDeleteBuffers(1, &gVBO);
    glDeleteProgram(gVisibleProgram);
    glDeleteProgram(gPickingProgram);
    if (gCursorProgram) glDeleteProgram(gCursorProgram);
    if (gCursorVAO) glDeleteVertexArrays(1, &gCursorVAO);
    if (gCursorVBO) glDeleteBuffers(1, &gCursorVBO);
    if (gPickFBO) {
        glDeleteFramebuffers(1, &gPickFBO);
        glDeleteTextures(1, &gPickColorTex);
        glDeleteRenderbuffers(1, &gPickDepthRB);
    }

    glfwDestroyWindow(gWindow);
    glfwTerminate();

    fprintf(stderr, "\nCOSMOS Galaxy Explorer shut down cleanly.\n");
    return 0;
}
