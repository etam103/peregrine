use serde::Serialize;

#[derive(Serialize, Clone)]
pub struct HardwareInfo {
    pub chip: String,
    pub model: String,
    pub cores_total: u32,
    pub cores_perf: u32,
    pub cores_efficiency: u32,
    pub memory_gb: u32,
    pub os_version: String,
    pub arch: String,
}

pub fn detect() -> HardwareInfo {
    HardwareInfo {
        chip: sysctl_string("machdep.cpu.brand_string")
            .unwrap_or_else(|| "Unknown".to_string()),
        model: sysctl_string("hw.model")
            .unwrap_or_else(|| "Unknown".to_string()),
        cores_total: sysctl_u32("hw.ncpu").unwrap_or(0),
        cores_perf: sysctl_u32("hw.perflevel0.logicalcpu").unwrap_or(0),
        cores_efficiency: sysctl_u32("hw.perflevel1.logicalcpu").unwrap_or(0),
        memory_gb: sysctl_u64("hw.memsize")
            .map(|b| (b / (1024 * 1024 * 1024)) as u32)
            .unwrap_or(0),
        os_version: sysctl_string("kern.osproductversion")
            .unwrap_or_else(|| "Unknown".to_string()),
        arch: std::env::consts::ARCH.to_string(),
    }
}

fn sysctl_string(name: &str) -> Option<String> {
    let c_name = std::ffi::CString::new(name).ok()?;
    let mut size: libc::size_t = 0;
    unsafe {
        if libc::sysctlbyname(c_name.as_ptr(), std::ptr::null_mut(), &mut size, std::ptr::null_mut(), 0) != 0 {
            return None;
        }
        if size == 0 {
            return None;
        }
        let mut buf = vec![0u8; size];
        if libc::sysctlbyname(c_name.as_ptr(), buf.as_mut_ptr() as *mut libc::c_void, &mut size, std::ptr::null_mut(), 0) != 0 {
            return None;
        }
        // Trim null terminator
        while buf.last() == Some(&0) {
            buf.pop();
        }
        String::from_utf8(buf).ok()
    }
}

fn sysctl_u32(name: &str) -> Option<u32> {
    let c_name = std::ffi::CString::new(name).ok()?;
    let mut value: u32 = 0;
    let mut size = std::mem::size_of::<u32>() as libc::size_t;
    unsafe {
        if libc::sysctlbyname(c_name.as_ptr(), &mut value as *mut u32 as *mut libc::c_void, &mut size, std::ptr::null_mut(), 0) != 0 {
            return None;
        }
    }
    Some(value)
}

fn sysctl_u64(name: &str) -> Option<u64> {
    let c_name = std::ffi::CString::new(name).ok()?;
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>() as libc::size_t;
    unsafe {
        if libc::sysctlbyname(c_name.as_ptr(), &mut value as *mut u64 as *mut libc::c_void, &mut size, std::ptr::null_mut(), 0) != 0 {
            return None;
        }
    }
    Some(value)
}
