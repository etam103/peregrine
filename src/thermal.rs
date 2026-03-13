//! Thermal monitoring via Darwin notifications (macOS, no root required).
//!
//! Polls `com.apple.system.thermalpressurelevel` to detect thermal throttling.
//! Non-macOS platforms always return `ThermalState::Nominal`.

/// System thermal pressure level (matches Darwin notification values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ThermalState {
    Nominal = 0,
    Moderate = 1,
    Heavy = 2,
    Trapping = 3,
    Sleeping = 4,
}

impl ThermalState {
    fn from_u64(v: u64) -> Self {
        match v {
            0 => ThermalState::Nominal,
            1 => ThermalState::Moderate,
            2 => ThermalState::Heavy,
            3 => ThermalState::Trapping,
            _ => ThermalState::Sleeping,
        }
    }

    /// Human-readable label.
    pub fn as_str(&self) -> &'static str {
        match self {
            ThermalState::Nominal => "nominal",
            ThermalState::Moderate => "moderate",
            ThermalState::Heavy => "heavy",
            ThermalState::Trapping => "trapping",
            ThermalState::Sleeping => "sleeping",
        }
    }
}

impl std::fmt::Display for ThermalState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(target_os = "macos")]
mod darwin {
    use super::ThermalState;
    use std::cell::Cell;
    use std::time::Instant;

    extern "C" {
        fn notify_register_check(name: *const u8, out_token: *mut i32) -> u32;
        fn notify_get_state(token: i32, out_state: *mut u64) -> u32;
    }

    const NOTIFY_STATUS_OK: u32 = 0;
    const THERMAL_KEY: &[u8] = b"com.apple.system.thermalpressurelevel\0";
    const CACHE_DURATION_MS: u128 = 100;

    pub struct ThermalMonitor {
        token: i32,
        cached_state: Cell<ThermalState>,
        last_poll: Cell<Option<Instant>>,
    }

    impl ThermalMonitor {
        pub fn new() -> Result<Self, String> {
            let mut token: i32 = 0;
            let status = unsafe {
                notify_register_check(THERMAL_KEY.as_ptr(), &mut token)
            };
            if status != NOTIFY_STATUS_OK {
                return Err(format!("notify_register_check failed: status {}", status));
            }
            Ok(ThermalMonitor {
                token,
                cached_state: Cell::new(ThermalState::Nominal),
                last_poll: Cell::new(None),
            })
        }

        pub fn state(&self) -> ThermalState {
            let now = Instant::now();
            if let Some(last) = self.last_poll.get() {
                if now.duration_since(last).as_millis() < CACHE_DURATION_MS {
                    return self.cached_state.get();
                }
            }

            let mut raw_state: u64 = 0;
            let status = unsafe { notify_get_state(self.token, &mut raw_state) };
            let state = if status == NOTIFY_STATUS_OK {
                ThermalState::from_u64(raw_state)
            } else {
                ThermalState::Nominal
            };

            self.cached_state.set(state);
            self.last_poll.set(Some(now));
            state
        }
    }

    thread_local! {
        static THERMAL_MONITOR: std::cell::RefCell<Option<ThermalMonitor>> =
            const { std::cell::RefCell::new(None) };
    }

    pub fn init() -> Result<(), String> {
        THERMAL_MONITOR.with(|m| {
            let mut m = m.borrow_mut();
            if m.is_none() {
                *m = Some(ThermalMonitor::new()?);
            }
            Ok(())
        })
    }

    pub fn state() -> ThermalState {
        THERMAL_MONITOR.with(|m| {
            m.borrow().as_ref().map(|mon| mon.state()).unwrap_or(ThermalState::Nominal)
        })
    }
}

#[cfg(not(target_os = "macos"))]
mod darwin {
    use super::ThermalState;

    pub fn init() -> Result<(), String> {
        Ok(())
    }

    pub fn state() -> ThermalState {
        ThermalState::Nominal
    }
}

/// Initialize thermal monitoring. Call once at startup.
/// Returns `Err` on macOS if Darwin notification registration fails.
/// No-op on other platforms.
pub fn thermal_init() -> Result<(), String> {
    darwin::init()
}

/// Get the current thermal pressure level (rate-limited to 100ms polling).
/// Returns `ThermalState::Nominal` if not initialized or on non-macOS.
pub fn thermal_state() -> ThermalState {
    darwin::state()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_init_and_poll() {
        thermal_init().expect("thermal_init should succeed");
        let state = thermal_state();
        // Should return a valid state (most likely Nominal on dev machines)
        let valid = matches!(
            state,
            ThermalState::Nominal
                | ThermalState::Moderate
                | ThermalState::Heavy
                | ThermalState::Trapping
                | ThermalState::Sleeping
        );
        assert!(valid, "unexpected thermal state: {:?}", state);
    }

    #[test]
    fn test_thermal_state_display() {
        assert_eq!(ThermalState::Nominal.as_str(), "nominal");
        assert_eq!(ThermalState::Heavy.to_string(), "heavy");
    }
}
