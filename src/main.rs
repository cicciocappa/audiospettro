use rustfft::{num_complex::Complex, FftPlanner};
use std::{
    ffi::{c_void, CStr},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SampleFormat, SampleRate, SupportedStreamConfigRange,
};

use std::time::Instant;

struct Timer {
    last_instant: Instant,
}

impl Timer {
    fn new() -> Self {
        Timer {
            last_instant: Instant::now(),
        }
    }

    fn elapsed_since_last(&mut self) -> std::time::Duration {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_instant);
        self.last_instant = now;
        elapsed
    }
}

fn desired_config(cfg: &SupportedStreamConfigRange) -> bool {
    cfg.channels() == 2
        && cfg.sample_format() == SampleFormat::F32
        && cfg.max_sample_rate() >= SampleRate(48_000)
}

struct Module {
    handle: *mut libopenmpt_sys::openmpt_module,
    pub playback_end: Arc<AtomicBool>,
}

impl Module {
    fn read(&mut self, rate: i32, data: &mut [f32]) {
        unsafe {
            let n_read = libopenmpt_sys::openmpt_module_read_interleaved_float_stereo(
                self.handle,
                rate,
                data.len() / 2,
                data.as_mut_ptr(),
            );
            if n_read == 0 {
                self.playback_end.store(true, Ordering::SeqCst);
            }
        };
    }
}

unsafe impl Send for Module {}

extern "C" fn logfunc(message: *const ::std::os::raw::c_char, _user: *mut ::std::os::raw::c_void) {
    let openmpt_log_msg = unsafe { CStr::from_ptr(message) };
    dbg!(openmpt_log_msg);
}

fn main() {
    let path = std::env::args().nth(1).expect("Need path to module file");
    let mod_data = std::fs::read(path).unwrap();
    let mod_handle = unsafe {
        libopenmpt_sys::openmpt_module_create_from_memory2(
            mod_data.as_ptr() as *const c_void,
            mod_data.len(),
            Some(logfunc),
            std::ptr::null_mut(),
            None,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null(),
        )
    };

    let fft_size = 2048; // Increased for better low-frequency resolution
    let num_bars = 64;
    let spectrum = Arc::new(Mutex::new(vec![0.0; num_bars]));
    let spectrum_clone = Arc::clone(&spectrum);

    if mod_handle.is_null() {
        eprintln!("Failed to create module. Exiting");
        return;
    }
    let playback_over = Arc::new(AtomicBool::new(false));
    let mut mod_handle = Module {
        handle: mod_handle,
        playback_end: playback_over.clone(),
    };
    let cpal_host = cpal::default_host();
    let cpal_dev = cpal_host.default_output_device().unwrap();
    let mut supported_cfgs = cpal_dev.supported_output_configs().unwrap();
    let Some(cfg) = supported_cfgs.find(desired_config) else {
        println!("Output device doesn't support desired parameters");
        return;
    };
    let cfg = cfg.with_sample_rate(SampleRate(48_000)).config();
    let mut timer = Timer::new();
    let stream = cpal_dev
        .build_output_stream(
            &cfg,
            move |data: &mut [f32], _cpal| {
                let elapsed = timer.elapsed_since_last();

                println!("{:?}", elapsed);

                // Create FFT planner
                let mut planner = FftPlanner::new();
                let fft = planner.plan_fft_forward(fft_size);

                // Prepare input for FFT
                let mut fft_input: Vec<Complex<f32>> = data
                    .iter()
                    .take(fft_size)
                    .map(|&sample| Complex {
                        re: sample,
                        im: 0.0,
                    })
                    .collect();

                // Zero-pad if necessary
                fft_input.resize(fft_size, Complex { re: 0.0, im: 0.0 });

                // Perform FFT
                fft.process(&mut fft_input);

                // Calculate magnitude and group into 64 bars
                let mut spectrum_guard = spectrum_clone.lock().unwrap();
                let bins_per_bar = fft_size / (2 * num_bars);

                for (i, bar) in spectrum_guard.iter_mut().enumerate() {
                    let start = i * bins_per_bar;
                    let end = (i + 1) * bins_per_bar;
                    *bar = fft_input[start..end].iter().map(|c| c.norm()).sum::<f32>()
                        / bins_per_bar as f32;
                }

                // Apply some smoothing (optional)
                for i in 1..num_bars - 1 {
                    spectrum_guard[i] =
                        (spectrum_guard[i - 1] + spectrum_guard[i] + spectrum_guard[i + 1]) / 3.0;
                }

                // Normalize (optional)
                let max_value = spectrum_guard.iter().cloned().fold(0. / 0., f32::max);
                if max_value > 0.0 {
                    for bar in spectrum_guard.iter_mut() {
                        *bar /= max_value;
                    }
                }

                // Play audio

                // Step 4: Optionally visualize or process spectrum

                mod_handle.read(cfg.sample_rate.0 as _, data)
            },
            |err| {
                dbg!(err);
            },
            None,
        )
        .unwrap();
    stream.play().unwrap();
    while playback_over.load(Ordering::SeqCst) == false {
        std::thread::sleep(Duration::from_millis(500));
    }
}
