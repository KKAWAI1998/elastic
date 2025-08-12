// wave3d.cpp
// 3D wavefield + CPU volume ray-marching (MIP / Compositing) with SDL2
// - Streaming texture reuse (fast)
// - Internal resolution scaling (renderScale)
// - Precomputed camera matrices per frame
// - Dynamic step count, fast trilinear
// Build:
//   g++ -std=c++17 wave3d_mip.cpp -lSDL2 -O2 -o wave3d_mip
//   # 並列化するなら: g++ -std=c++17 wave3d_mip.cpp -lSDL2 -O2 -fopenmp -o wave3d_mip
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ------------------------ 基本ユーティリティ ------------------------
static inline float clamp(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
static inline float clamp01(float x) { return clamp(x, 0.0f, 1.0f); }

// ------------------------ グリッド＆物理 ------------------------
struct Grid3D {
    int nx, ny, nz;
    float dx, c, dt; // CFL: c*dt/dx <= 1/sqrt(3)
    std::vector<float> u_prev, u_curr, u_next;
    inline int idx(int i, int j, int k) const { return (k * ny + j) * nx + i; }
};

static void stepWave(Grid3D &g) {
    const float s = (g.c * g.dt) * (g.c * g.dt) / (g.dx * g.dx);
// 内部セル更新
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for (int k = 1; k < g.nz - 1; ++k)
        for (int j = 1; j < g.ny - 1; ++j)
            for (int i = 1; i < g.nx - 1; ++i) {
                const int id = g.idx(i, j, k);
                const float lap = g.u_curr[g.idx(i + 1, j, k)] + g.u_curr[g.idx(i - 1, j, k)] +
                                  g.u_curr[g.idx(i, j + 1, k)] + g.u_curr[g.idx(i, j - 1, k)] +
                                  g.u_curr[g.idx(i, j, k + 1)] + g.u_curr[g.idx(i, j, k - 1)] -
                                  6.0f * g.u_curr[id];
                g.u_next[id] = 2.f * g.u_curr[id] - g.u_prev[id] + s * lap;
            }

// Neumann的境界（コピー）
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < g.nx; i++)
        for (int j = 0; j < g.ny; j++) {
            g.u_next[g.idx(i, j, 0)] = g.u_next[g.idx(i, j, 1)];
            g.u_next[g.idx(i, j, g.nz - 1)] = g.u_next[g.idx(i, j, g.nz - 2)];
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < g.nz; k++)
        for (int j = 0; j < g.ny; j++) {
            g.u_next[g.idx(0, j, k)] = g.u_next[g.idx(1, j, k)];
            g.u_next[g.idx(g.nx - 1, j, k)] = g.u_next[g.idx(g.nx - 2, j, k)];
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < g.nz; k++)
        for (int i = 0; i < g.nx; i++) {
            g.u_next[g.idx(i, 0, k)] = g.u_next[g.idx(i, 1, k)];
            g.u_next[g.idx(i, g.ny - 1, k)] = g.u_next[g.idx(i, g.ny - 2, k)];
        }

    std::swap(g.u_prev, g.u_curr);
    std::swap(g.u_curr, g.u_next);
}

// ------------------------ サンプリング＆レイ補助 ------------------------
inline float sampleTrilinearFast(const Grid3D &g, float x, float y, float z) {
    // 前提: x,y,z ∈ [0,1]（AABB交差で保証）
    float fx = x * (g.nx - 1), fy = y * (g.ny - 1), fz = z * (g.nz - 1);
    int x0 = (int)fx, y0 = (int)fy, z0 = (int)fz;
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
    if (x0 < 0)
        x0 = 0;
    if (y0 < 0)
        y0 = 0;
    if (z0 < 0)
        z0 = 0;
    if (x1 >= g.nx)
        x1 = g.nx - 1;
    if (y1 >= g.ny)
        y1 = g.ny - 1;
    if (z1 >= g.nz)
        z1 = g.nz - 1;
    const float tx = fx - x0, ty = fy - y0, tz = fz - z0;

    auto V = [&](int i, int j, int k) { return g.u_curr[g.idx(i, j, k)]; };

    float c000 = V(x0, y0, z0), c100 = V(x1, y0, z0);
    float c010 = V(x0, y1, z0), c110 = V(x1, y1, z0);
    float c001 = V(x0, y0, z1), c101 = V(x1, y0, z1);
    float c011 = V(x0, y1, z1), c111 = V(x1, y1, z1);

    float c00 = c000 * (1 - tx) + c100 * tx;
    float c10 = c010 * (1 - tx) + c110 * tx;
    float c01 = c001 * (1 - tx) + c101 * tx;
    float c11 = c011 * (1 - tx) + c111 * tx;

    float c0 = c00 * (1 - ty) + c10 * ty;
    float c1 = c01 * (1 - ty) + c11 * ty;
    return c0 * (1 - tz) + c1 * tz;
}

static bool intersectUnitCube(const float o[3], const float d[3], float &tmin, float &tmax) {
    tmin = 0.0f;
    tmax = 1e9f;
    for (int a = 0; a < 3; ++a) {
        float inv = 1.0f / d[a];
        float t0 = (0.0f - o[a]) * inv;
        float t1 = (1.0f - o[a]) * inv;
        if (inv < 0.0f)
            std::swap(t0, t1);
        tmin = t0 > tmin ? t0 : tmin;
        tmax = t1 < tmax ? t1 : tmax;
        if (tmax < tmin)
            return false;
    }
    return true;
}

// ------------------------ カメラ＆投影（中心付き） ------------------------
struct CamInfo {
    float cam[3];    // camera position (world)
    float R[9];      // world->camera rotation (3x3, row-major)
    float fov;       // vertical FOV
    int W, H;        // render size
    float center[3]; // look-at point
};

static inline void mul3x3(const float M[9], const float v[3], float out[3]) {
    out[0] = M[0] * v[0] + M[1] * v[1] + M[2] * v[2];
    out[1] = M[3] * v[0] + M[4] * v[1] + M[5] * v[2];
    out[2] = M[6] * v[0] + M[7] * v[1] + M[8] * v[2];
}

// yaw/pitch/dist & center → camera (lookAtで常にcenterを見る)
static CamInfo makeCamera(float yaw, float pitch, float dist, const float center[3], int W, int H) {
    CamInfo c{};
    c.fov = 60.0f * float(M_PI) / 180.0f;
    c.W = W;
    c.H = H;
    c.center[0] = center[0];
    c.center[1] = center[1];
    c.center[2] = center[2];

    // ピッチの行き過ぎ防止（±~89度）
    float p = clamp(pitch, -1.55f, 1.55f);
    float cy = std::cos(yaw), sy = std::sin(yaw);
    float cp = std::cos(p), sp = std::sin(p);

    // 中心を原点と見たときのオービット位置（Yが世界の上）
    float offset[3] = {dist * sy * cp, -dist * sp, dist * cy * cp};
    c.cam[0] = center[0] + offset[0];
    c.cam[1] = center[1] + offset[1];
    c.cam[2] = center[2] + offset[2];

    // lookAt: 目→中心 の正方向を fwd とする
    float fwd[3] = {center[0] - c.cam[0], center[1] - c.cam[1], center[2] - c.cam[2]};
    // 正規化
    float fn = std::sqrt(fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]) + 1e-12f;
    fwd[0] /= fn;
    fwd[1] /= fn;
    fwd[2] /= fn;

    // 世界の上方向
    float upW[3] = {0.0f, 1.0f, 0.0f};
    // fwdとほぼ平行なら別のupを採用
    if (std::fabs(fwd[0] * upW[0] + fwd[1] * upW[1] + fwd[2] * upW[2]) > 0.99f) {
        upW[0] = 0.0f;
        upW[1] = 0.0f;
        upW[2] = 1.0f;
    }

    // 右= fwd × upW
    float right[3] = {fwd[1] * upW[2] - fwd[2] * upW[1], fwd[2] * upW[0] - fwd[0] * upW[2],
                      fwd[0] * upW[1] - fwd[1] * upW[0]};
    float rn = std::sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]) + 1e-12f;
    right[0] /= rn;
    right[1] /= rn;
    right[2] /= rn;

    // 上 = 右 × fwd
    float up[3] = {right[1] * fwd[2] - right[2] * fwd[1], right[2] * fwd[0] - right[0] * fwd[2],
                   right[0] * fwd[1] - right[1] * fwd[0]};

    // world→camera 回転（行優先, カメラの -Z が前方）
    c.R[0] = right[0];
    c.R[1] = right[1];
    c.R[2] = right[2];
    c.R[3] = up[0];
    c.R[4] = up[1];
    c.R[5] = up[2];
    c.R[6] = -fwd[0];
    c.R[7] = -fwd[1];
    c.R[8] = -fwd[2];

    return c;
}

static bool projectToScreen(const CamInfo &c, const float P[3], int &sx, int &sy) {
    float Pw[3] = {P[0] - c.cam[0], P[1] - c.cam[1], P[2] - c.cam[2]};
    float Pc[3];
    mul3x3(c.R, Pw, Pc); // world→camera
    if (Pc[2] >= -1e-6f)
        return false; // camera front is -Z
    float invF = 1.0f / std::tan(0.5f * c.fov);
    float xn = (Pc[0] / -Pc[2]) * invF;
    float yn = (Pc[1] / -Pc[2]) * invF;
    float u = (xn * 0.5f + 0.5f) * c.W;
    float v = (yn * 0.5f + 0.5f) * c.H;
    sx = int(u + 0.5f);
    sy = int(v + 0.5f);
    return (sx >= 0 && sx < c.W && sy >= 0 && sy < c.H);
}

// ------------------------ 転送関数（透過合成用） ------------------------
static inline void transferDiverging(float s, float k_opacity, float gamma, float &r, float &g,
                                     float &b, float &a) {
    // s: -1..1 (use tanh(val/vmax) 推奨)
    float ap = s > 0 ? s : 0.0f;
    float an = s < 0 ? -s : 0.0f;
    // 正: オレンジ, 負: シアン
    float rp = 1.00f, gp = 0.62f, bp = 0.10f;
    float rn = 0.20f, gn = 0.75f, bn = 1.00f;
    float w = ap + an + 1e-6f;
    float cr = (ap * rp + an * rn) / w;
    float cg = (ap * gp + an * gn) / w;
    float cb = (ap * bp + an * bn) / w;
    float alpha = k_opacity * std::pow(ap + an, gamma);
    a = clamp01(alpha);
    r = clamp01(cr);
    g = clamp01(cg);
    b = clamp01(cb);
}

// ------------------------ レンダ（ストリーミングテクスチャに直接） ------------------------
static void renderMIP_stream(SDL_Renderer *ren, SDL_Texture *tex, int RW, int RH, const Grid3D &g,
                             float yaw, float pitch, float dist, const float center[3],
                             float stepWorld, float vmax) {
    void *pixels = nullptr;
    int pitchB = 0;
    SDL_LockTexture(tex, nullptr, &pixels, &pitchB);

    uint8_t *base = (uint8_t *)pixels;
    const float fov = 60.0f * float(M_PI) / 180.0f;
    const float invF = 1.0f / std::tan(0.5f * fov);

    CamInfo cam = makeCamera(yaw, pitch, dist, center, RW, RH);
    // Rcw = transpose(cam.R)
    const float Rcw[9] = {cam.R[0], cam.R[3], cam.R[6], cam.R[1], cam.R[4],
                          cam.R[7], cam.R[2], cam.R[5], cam.R[8]};

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < RH; ++y) {
        uint32_t *row = (uint32_t *)(base + y * pitchB);
        for (int x = 0; x < RW; ++x) {
            float u = ((x + 0.5f) / float(RW)) * 2.f - 1.f;
            float v = ((y + 0.5f) / float(RH)) * 2.f - 1.f;
            float dirCam[3] = {u, v, -invF};
            float len =
                std::sqrt(dirCam[0] * dirCam[0] + dirCam[1] * dirCam[1] + dirCam[2] * dirCam[2]);
            dirCam[0] /= len;
            dirCam[1] /= len;
            dirCam[2] /= len;
            float dir[3] = {Rcw[0] * dirCam[0] + Rcw[1] * dirCam[1] + Rcw[2] * dirCam[2],
                            Rcw[3] * dirCam[0] + Rcw[4] * dirCam[1] + Rcw[5] * dirCam[2],
                            Rcw[6] * dirCam[0] + Rcw[7] * dirCam[1] + Rcw[8] * dirCam[2]};

            float tmin, tmax;
            if (!intersectUnitCube(cam.cam, dir, tmin, tmax)) {
                row[x] = 0xFFFFFFFF;
                continue;
            }

            float t = std::max(tmin, 0.0f);
            float length = tmax - t;
            int maxSteps = int(length / stepWorld) + 2;

            float maxv = 0.0f;
            for (int s = 0; s < maxSteps; ++s, t += stepWorld) {
                float px = cam.cam[0] + dir[0] * t;
                float py = cam.cam[1] + dir[1] * t;
                float pz = cam.cam[2] + dir[2] * t;
                float val = std::fabs(sampleTrilinearFast(g, px, py, pz));
                if (val > maxv)
                    maxv = val;
            }
            int shade = int(255 * (1.0f - clamp01(maxv / vmax))); // 0..255
            row[x] = (0xFF << 24) | (shade << 16) | (shade << 8) | shade;
        }
    }
    SDL_UnlockTexture(tex);
}

static void renderComposite_stream(SDL_Renderer *ren, SDL_Texture *tex, int RW, int RH,
                                   const Grid3D &g, float yaw, float pitch, float dist,
                                   const float center[3], float stepWorld, float vmax) {
    void *pixels = nullptr;
    int pitchB = 0;
    SDL_LockTexture(tex, nullptr, &pixels, &pitchB);

    uint8_t *base = (uint8_t *)pixels;
    const float fov = 60.0f * float(M_PI) / 180.0f;
    const float invF = 1.0f / std::tan(0.5f * fov);

    CamInfo cam = makeCamera(yaw, pitch, dist, center, RW, RH);
    // Rcw = transpose(cam.R)
    const float Rcw[9] = {cam.R[0], cam.R[3], cam.R[6], cam.R[1], cam.R[4],
                          cam.R[7], cam.R[2], cam.R[5], cam.R[8]};

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < RH; ++y) {
        uint32_t *row = (uint32_t *)(base + y * pitchB);
        for (int x = 0; x < RW; ++x) {
            float u = ((x + 0.5f) / float(RW)) * 2.f - 1.f;
            float v = ((y + 0.5f) / float(RH)) * 2.f - 1.f;
            float dirCam[3] = {u, v, -invF};
            float len =
                std::sqrt(dirCam[0] * dirCam[0] + dirCam[1] * dirCam[1] + dirCam[2] * dirCam[2]);
            dirCam[0] /= len;
            dirCam[1] /= len;
            dirCam[2] /= len;
            float dir[3] = {Rcw[0] * dirCam[0] + Rcw[1] * dirCam[1] + Rcw[2] * dirCam[2],
                            Rcw[3] * dirCam[0] + Rcw[4] * dirCam[1] + Rcw[5] * dirCam[2],
                            Rcw[6] * dirCam[0] + Rcw[7] * dirCam[1] + Rcw[8] * dirCam[2]};

            float tmin, tmax;
            if (!intersectUnitCube(cam.cam, dir, tmin, tmax)) {
                row[x] = 0xFFFFFFFF;
                continue;
            }

            float t = std::max(tmin, 0.0f);
            float length = tmax - t;
            int maxSteps = int(length / stepWorld) + 2;

            float accR = 0, accG = 0, accB = 0, accA = 0;
            for (int s = 0; s < maxSteps; ++s, t += stepWorld) {
                float px = cam.cam[0] + dir[0] * t;
                float py = cam.cam[1] + dir[1] * t;
                float pz = cam.cam[2] + dir[2] * t;

                float val = sampleTrilinearFast(g, px, py, pz);
                float m = std::tanh(std::fabs(val) / vmax); // 0..1 に圧縮
                float a = 0.35f * std::pow(m, 0.9f);        // 不透明度（調整可）
                a = clamp01(a);

                accA += (1.0f - accA) * a; // front-to-back
                if (accA > 0.99f)
                    break; // 早期終了
            }
            int shade = int(255 * (1.0f - clamp01(accA)));
            row[x] = (0xFF << 24) | (shade << 16) | (shade << 8) | shade;
        }
    }
    SDL_UnlockTexture(tex);
}

// ------------------------ オーバーレイ（ボックス＆XYZ軸） ------------------------
static void drawLine(SDL_Renderer *ren, int x0, int y0, int x1, int y1) {
    SDL_RenderDrawLine(ren, x0, y0, x1, y1);
}

static void drawOverlay(SDL_Renderer *ren, float yaw, float pitch, float dist,
                        const float center[3]) {
    int W, H;
    SDL_GetRendererOutputSize(ren, &W, &H);
    CamInfo cam = makeCamera(yaw, pitch, dist, center, W, H);

    auto proj = [&](float x, float y, float z, int &sx, int &sy) -> bool {
        float P[3] = {x, y, z};
        return projectToScreen(cam, P, sx, sy);
    };

    // unit cube edges (white)
    SDL_SetRenderDrawColor(ren, 40, 40, 40, 230);
    const float C[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                           {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    const int E[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                          {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    for (auto &e : E) {
        int x0, y0, x1, y1;
        if (proj(C[e[0]][0], C[e[0]][1], C[e[0]][2], x0, y0) &&
            proj(C[e[1]][0], C[e[1]][1], C[e[1]][2], x1, y1)) {
            drawLine(ren, x0, y0, x1, y1);
        }
    }

    // axes at origin
    struct Axis {
        float x, y, z;
        Uint8 r, g, b;
    };
    Axis axes[3] = {{0.35f, 0.0f, 0.0f, 255, 60, 60},
                    {0.0f, 0.35f, 0.0f, 60, 255, 60},
                    {0.0f, 0.0f, 0.35f, 60, 160, 255}};
    int oX, oY;
    if (proj(0, 0, 0, oX, oY)) {
        for (auto &a : axes) {
            int ax, ay;
            if (proj(a.x, a.y, a.z, ax, ay)) {
                SDL_SetRenderDrawColor(ren, a.r, a.g, a.b, 255);
                drawLine(ren, oX, oY, ax, ay);
                int mx = (ax * 3 + oX) / 4, my = (ay * 3 + oY) / 4;
                drawLine(ren, ax, ay, mx + 3, my);
                drawLine(ren, ax, ay, mx, my + 3);
            }
        }
    }
}

// ------------------------ メイン ------------------------
int main() {
    // 物理グリッド
    Grid3D g{120, 96, 96, 1.0f, 1500.0f, 0.00022f}; // dt少し控えめ（CFL余裕）
    g.u_prev.resize(g.nx * g.ny * g.nz, 0.0f);
    g.u_curr.resize(g.nx * g.ny * g.nz, 0.0f);
    g.u_next.resize(g.nx * g.ny * g.nz, 0.0f);

    // 初期ガウシアンパルス
    int cx = g.nx / 2, cy = g.ny / 2, cz = g.nz / 2;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                float dx = i - cx, dy = j - cy, dz = k - cz;
                float r2 = dx * dx + dy * dy + dz * dz;
                g.u_curr[g.idx(i, j, k)] = std::exp(-r2 / 100.0f);
            }
    g.u_prev = g.u_curr;

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        std::cerr << "SDL_Init: " << SDL_GetError() << "\n";
        return 1;
    }
    SDL_Window *win = SDL_CreateWindow("3D Wave - Volume View", SDL_WINDOWPOS_CENTERED,
                                       SDL_WINDOWPOS_CENTERED, 1000, 750, SDL_WINDOW_RESIZABLE);
    SDL_Renderer *ren =
        SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!ren)
        ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);
    if (!ren) {
        std::cerr << "SDL_CreateRenderer: " << SDL_GetError() << "\n";
        return 1;
    }

    // ストリーミングテクスチャ（使い回し）
    SDL_Texture *tex = nullptr;
    int TW = 0, TH = 0;
    float renderScale = 0.65f; // 内部解像度（0.5〜0.8推奨）

    // カメラ＆レンダ設定
    float yaw = 0.6f, pitch = -0.3f, dist = 1.8f;
    float center[3] = {0.5f, 0.5f, 0.5f};
    float stepWorld = 0.010f;  // レイの歩幅（上げるほど速い/荒い）
    float vmax = 0.05f;        // 明るさスケール
    bool useComposite = false; // デフォルトは軽いMIP
    bool showOverlay = true;

    bool running = true;
    SDL_Event ev;

    auto ensureTexture = [&](int W, int H) {
        int RW = std::max(1, int(W * renderScale));
        int RH = std::max(1, int(H * renderScale));
        if (!tex || RW != TW || RH != TH) {
            if (tex)
                SDL_DestroyTexture(tex);
            tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, RW,
                                    RH);
            TW = RW;
            TH = RH;
            SDL_SetTextureBlendMode(tex, SDL_BLENDMODE_NONE);
        }
    };

    auto pan = [&](float dx, float dy) {
        int W, H;
        SDL_GetRendererOutputSize(ren, &W, &H);
        CamInfo cam = makeCamera(yaw, pitch, dist, center, W, H);
        float Rcw[9] = {cam.R[0], cam.R[3], cam.R[6], cam.R[1], cam.R[4],
                        cam.R[7], cam.R[2], cam.R[5], cam.R[8]};
        float right[3] = {Rcw[0], Rcw[3], Rcw[6]};
        float upv[3] = {Rcw[1], Rcw[4], Rcw[7]};
        float s = 0.15f * stepWorld * dist;
        for (int a = 0; a < 3; ++a) {
            center[a] += s * (right[a] * dx + upv[a] * dy);
            center[a] = clamp(center[a], 0.0f, 1.0f);
        }
    };

    while (running) {
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT)
                running = false;
            else if (ev.type == SDL_KEYDOWN) {
                const Uint16 mod = SDL_GetModState();
                const bool sh = (mod & KMOD_SHIFT);
                switch (ev.key.keysym.sym) {
                case SDLK_ESCAPE:
                    running = false;
                    break;
                case SDLK_LEFT:
                    if (sh)
                        pan(-1, 0);
                    else
                        yaw -= 0.05f;
                    break;
                case SDLK_RIGHT:
                    if (sh)
                        pan(+1, 0);
                    else
                        yaw += 0.05f;
                    break;
                case SDLK_UP:
                    if (sh)
                        pan(0, +1);
                    else
                        pitch += 0.05f;
                    break;
                case SDLK_DOWN:
                    if (sh)
                        pan(0, -1);
                    else
                        pitch -= 0.05f;
                    break;
                case SDLK_EQUALS:
                    dist = std::max(0.6f, dist - 0.1f);
                    break; // [+]
                case SDLK_MINUS:
                    dist += 0.1f;
                    break; // [-]
                case SDLK_1:
                    stepWorld = std::max(0.003f, stepWorld * 0.8f);
                    break;
                case SDLK_2:
                    stepWorld = std::min(0.03f, stepWorld * 1.25f);
                    break;
                case SDLK_3:
                    vmax *= 0.8f;
                    break;
                case SDLK_4:
                    vmax *= 1.25f;
                    break;
                case SDLK_m:
                    useComposite = !useComposite;
                    break;
                case SDLK_a:
                    showOverlay = !showOverlay;
                    break;
                case SDLK_r:
                    yaw = 0.6f;
                    pitch = -0.3f;
                    dist = 1.8f;
                    center[0] = center[1] = center[2] = 0.5f;
                    renderScale = 0.65f;
                    stepWorld = 0.010f;
                    vmax = 0.05f;
                    useComposite = false;
                    break;
                case SDLK_5: // 内部解像度↓（速い）
                    renderScale = std::max(0.40f, renderScale - 0.05f);
                    break;
                case SDLK_6: // 内部解像度↑（綺麗）
                    renderScale = std::min(1.00f, renderScale + 0.05f);
                    break;
                }
            }
        }

        // 物理更新（軽くするなら1回/フレームでOK）
        stepWave(g);

        int W, H;
        SDL_GetRendererOutputSize(ren, &W, &H);
        ensureTexture(W, H);

        // レンダ
        if (useComposite)
            renderComposite_stream(ren, tex, TW, TH, g, yaw, pitch, std::max(1.1f, dist), center,
                                   stepWorld, vmax);
        else
            renderMIP_stream(ren, tex, TW, TH, g, yaw, pitch, std::max(1.1f, dist), center,
                             stepWorld, vmax);

        // 画面へ拡大コピー
        SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
        SDL_RenderClear(ren);
        SDL_Rect dst{0, 0, W, H};
        SDL_RenderCopy(ren, tex, nullptr, &dst);
        if (showOverlay)
            drawOverlay(ren, yaw, pitch, std::max(1.1f, dist), center);
        SDL_RenderPresent(ren);
    }

    if (tex)
        SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
