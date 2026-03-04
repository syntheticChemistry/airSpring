// SPDX-License-Identifier: AGPL-3.0-or-later
//
// local_elementwise.wgsl — LEGACY f32 shader (superseded by local_elementwise_f64.wgsl).
//
// Retained for reference. Production code uses local_elementwise_f64.wgsl
// compiled via compile_shader_universal (f64 canonical → any precision).
//
// Six element-wise agricultural operations in f32. Each invocation reads
// three inputs (a, b, c) and writes one output per element.
//
// Op 0: SCS-CN runoff       — a=P(mm), b=CN, c=ia_ratio
// Op 1: Stewart yield ratio — a=Ky, b=ETa/ETc, c=unused
// Op 2: Makkink ET₀         — a=T(°C), b=Rs(MJ), c=elev(m)
// Op 3: Turc ET₀            — a=T(°C), b=Rs(MJ), c=RH(%)
// Op 4: Hamon PET           — a=T(°C), b=lat(rad), c=doy
// Op 5: Blaney-Criddle ET₀  — a=T(°C), b=lat(rad), c=doy

struct Params {
    op: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> in_a: array<f32>;
@group(0) @binding(2) var<storage, read> in_b: array<f32>;
@group(0) @binding(3) var<storage, read> in_c: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

const PI: f32 = 3.14159265;
const TWO_PI: f32 = 6.28318530;

fn sat_vp(t: f32) -> f32 {
    return 0.6108 * exp(17.27 * t / (t + 237.3));
}

fn vp_slope(t: f32) -> f32 {
    let es = sat_vp(t);
    let d = t + 237.3;
    return 4098.0 * es / (d * d);
}

fn atm_pressure(elev: f32) -> f32 {
    return 101.3 * pow((293.0 - 0.0065 * elev) / 293.0, 5.26);
}

fn psychro(p_kpa: f32) -> f32 {
    return 0.000665 * p_kpa;
}

fn daylight_hr(lat: f32, doy: f32) -> f32 {
    let decl = 0.4093 * sin(TWO_PI / 365.0 * doy - 1.39);
    let arg = clamp(-tan(lat) * tan(decl), -1.0, 1.0);
    let ws = acos(arg);
    return 24.0 / PI * ws;
}

fn scs_cn(p: f32, cn: f32, ia_ratio: f32) -> f32 {
    if p <= 0.0 || cn <= 0.0 { return 0.0; }
    let s = 25400.0 / cn - 254.0;
    let ia = ia_ratio * s;
    if p <= ia { return 0.0; }
    let pe = p - ia;
    return pe * pe / (pe + s);
}

fn stewart_yield(ky: f32, eta_etc: f32) -> f32 {
    return 1.0 - ky * (1.0 - eta_etc);
}

fn makkink(t: f32, rs: f32, elev: f32) -> f32 {
    let p_kpa = atm_pressure(elev);
    let gamma = psychro(p_kpa);
    let delta = vp_slope(t);
    return max(0.61 * delta / (delta + gamma) * rs / 2.45 - 0.12, 0.0);
}

fn turc(t: f32, rs: f32, rh: f32) -> f32 {
    let denom = t + 15.0;
    if denom == 0.0 { return 0.0; }
    let t_fac = t / denom;
    if t_fac < 0.0 { return 0.0; }
    let rs_cal = 23.8846 * rs + 50.0;
    var et0 = 0.013 * t_fac * rs_cal;
    if rh < 50.0 {
        et0 = et0 * (1.0 + (50.0 - rh) / 70.0);
    }
    return max(et0, 0.0);
}

fn hamon(t: f32, lat: f32, doy: f32) -> f32 {
    if t < 0.0 { return 0.0; }
    let n = daylight_hr(lat, doy);
    if n <= 0.0 { return 0.0; }
    let es = sat_vp(t);
    let rhosat = 216.7 * es / (t + 273.3);
    return 0.1651 * n * rhosat;
}

fn blaney_criddle(t: f32, lat: f32, doy: f32) -> f32 {
    let n = daylight_hr(lat, doy);
    let p = n / 43.80;
    return max(p * (0.46 * t + 8.13), 0.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }

    let a = in_a[idx];
    let b = in_b[idx];
    let c = in_c[idx];

    var result: f32;

    switch params.op {
        case 0u: { result = scs_cn(a, b, c); }
        case 1u: { result = stewart_yield(a, b); }
        case 2u: { result = makkink(a, b, c); }
        case 3u: { result = turc(a, b, c); }
        case 4u: { result = hamon(a, b, c); }
        case 5u: { result = blaney_criddle(a, b, c); }
        default: { result = 0.0; }
    }

    out[idx] = result;
}
