// SPDX-License-Identifier: AGPL-3.0-or-later
//! Synthetic data generators for cross-spring benchmarks.

use airspring_barracuda::eco::evapotranspiration::DailyEt0Input;
use airspring_barracuda::eco::richards::VanGenuchtenParams;
use airspring_barracuda::gpu::et0::StationDay;
use airspring_barracuda::gpu::richards::RichardsRequest;

pub fn sample_station_day(doy: u32) -> StationDay {
    StationDay {
        tmax: 0.01f64.mul_add(f64::from(doy), 21.5),
        tmin: 0.005f64.mul_add(f64::from(doy), 12.3),
        rh_max: 84.0,
        rh_min: 63.0,
        wind_2m: 2.078,
        rs: 0.02f64.mul_add(f64::from(doy), 22.07),
        elevation: 100.0,
        latitude: 50.80,
        doy,
    }
}

pub fn sample_et0_input(doy: u32) -> DailyEt0Input {
    DailyEt0Input {
        tmin: 0.005f64.mul_add(f64::from(doy), 12.3),
        tmax: 0.01f64.mul_add(f64::from(doy), 21.5),
        tmean: None,
        solar_radiation: 0.02f64.mul_add(f64::from(doy), 22.07),
        wind_speed_2m: 2.078,
        actual_vapour_pressure: 1.409,
        elevation_m: 100.0,
        latitude_deg: 50.80,
        day_of_year: doy,
    }
}

pub const fn sand_richards_request() -> RichardsRequest {
    RichardsRequest {
        params: VanGenuchtenParams {
            theta_r: 0.045,
            theta_s: 0.43,
            alpha: 0.145,
            n_vg: 2.68,
            ks: 712.8,
        },
        depth_cm: 100.0,
        n_nodes: 20,
        h_initial: -5.0,
        h_top: -5.0,
        zero_flux_top: true,
        bottom_free_drain: true,
        duration_days: 0.1,
        dt_days: 0.01,
    }
}
