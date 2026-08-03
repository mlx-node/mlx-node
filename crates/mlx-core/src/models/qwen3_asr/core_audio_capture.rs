//! Native macOS audio capture used by Qwen3-ASR.
//!
//! Microphones are read through an AUHAL input unit. System audio uses a
//! private Core Audio process tap attached to an auto-starting private
//! aggregate device. Both paths expose packed mono float PCM to the caller.

use std::ffi::{CStr, c_void};
use std::mem::{MaybeUninit, size_of};
use std::ptr::{NonNull, null};
use std::sync::atomic::{AtomicU32, Ordering};

use coreaudio::audio_unit::audio_format::LinearPcmFlags;
use coreaudio::audio_unit::macos_helpers::{
    audio_unit_from_device_id_uninitialized, get_audio_device_ids, get_audio_device_supports_scope,
    get_default_device_id, get_device_name,
};
use coreaudio::audio_unit::render_callback::{self, data};
use coreaudio::audio_unit::{AudioUnit, Element, SampleFormat, Scope, StreamFormat};
use napi::bindgen_prelude::{Error, Result};
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2_core_audio::{
    AudioHardwareCreateAggregateDevice, AudioHardwareCreateProcessTap,
    AudioHardwareDestroyAggregateDevice, AudioHardwareDestroyProcessTap,
    AudioObjectGetPropertyData, AudioObjectGetPropertyDataSize, AudioObjectID,
    AudioObjectPropertyAddress, CATapDescription, CATapMuteBehavior,
    kAudioAggregateDeviceIsPrivateKey, kAudioAggregateDeviceNameKey,
    kAudioAggregateDeviceTapAutoStartKey, kAudioAggregateDeviceTapListKey,
    kAudioAggregateDeviceUIDKey, kAudioDevicePropertyDeviceUID,
    kAudioDevicePropertyNominalSampleRate, kAudioDevicePropertyStreamConfiguration,
    kAudioHardwarePropertyProcessObjectList, kAudioObjectPropertyElementMain,
    kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyScopeInput,
    kAudioObjectPropertyScopeOutput, kAudioObjectSystemObject, kAudioProcessPropertyBundleID,
    kAudioSubTapDriftCompensationKey, kAudioSubTapUIDKey,
};
use objc2_core_audio_types::{AudioBuffer, AudioBufferList};
use objc2_core_foundation::{
    CFArray, CFDictionary, CFMutableDictionary, CFRetained, CFString, kCFAllocatorDefault,
    kCFTypeArrayCallBacks, kCFTypeDictionaryKeyCallBacks, kCFTypeDictionaryValueCallBacks,
};
use objc2_foundation::{NSArray, NSNumber, NSString};

use super::config::{Qwen3AsrCaptureOptions, Qwen3AsrCaptureSource};

static CAPTURE_INSTANCE: AtomicU32 = AtomicU32::new(0);

#[derive(Clone)]
pub(super) struct NativeDeviceInfo {
    pub object_id: AudioObjectID,
    pub id: String,
    pub name: String,
    pub sample_rate: u32,
    pub channels: u32,
}

pub(super) struct NativeCapture {
    // AudioUnit must be dropped before the aggregate device and tap.
    audio_unit: AudioUnit,
    _system_tap: Option<SystemAudioTap>,
    pub device: NativeDeviceInfo,
}

impl NativeCapture {
    pub fn start(&mut self) -> Result<()> {
        self.audio_unit
            .start()
            .map_err(|error| native_error("start Core Audio capture", error))
    }

    pub fn stop(&mut self) -> Result<()> {
        self.audio_unit
            .stop()
            .map_err(|error| native_error("stop Core Audio capture", error))
    }
}

impl Drop for NativeCapture {
    fn drop(&mut self) {
        let _ = self.audio_unit.stop();
    }
}

struct SystemAudioTap {
    aggregate_device_id: AudioObjectID,
    tap_id: AudioObjectID,
}

impl Drop for SystemAudioTap {
    fn drop(&mut self) {
        unsafe {
            let _ = AudioHardwareDestroyAggregateDevice(self.aggregate_device_id);
            let _ = AudioHardwareDestroyProcessTap(self.tap_id);
        }
    }
}

pub(super) fn audio_devices(source: Qwen3AsrCaptureSource) -> Result<Vec<NativeDeviceInfo>> {
    let scope = source_scope(source);
    get_audio_device_ids()
        .map_err(|error| native_error("list Core Audio devices", error))?
        .into_iter()
        .filter(|&id| get_audio_device_supports_scope(id, scope).unwrap_or(false))
        .map(|id| device_info(id, source))
        .collect()
}

pub(super) fn default_device_id(source: Qwen3AsrCaptureSource) -> Option<AudioObjectID> {
    get_default_device_id(source == Qwen3AsrCaptureSource::Microphone)
}

pub(super) fn selected_device(
    options: &Qwen3AsrCaptureOptions,
) -> Result<(Qwen3AsrCaptureSource, NativeDeviceInfo)> {
    let source = options.source.unwrap_or_default();
    let device_id = select_device(source, options)?;
    Ok((source, device_info(device_id, source)?))
}

pub(super) fn build(
    options: &Qwen3AsrCaptureOptions,
    mut on_samples: impl FnMut(&[f32]) + Send + 'static,
) -> Result<(Qwen3AsrCaptureSource, NativeCapture)> {
    let (source, device) = selected_device(options)?;
    let device_id = device.object_id;

    let (capture_device_id, system_tap) = match source {
        Qwen3AsrCaptureSource::Microphone => {
            if options
                .application_bundle_ids
                .as_ref()
                .is_some_and(|ids| !ids.is_empty())
            {
                return Err(Error::from_reason(
                    "application_bundle_ids is only valid for systemAudio capture",
                ));
            }
            (device_id, None)
        }
        Qwen3AsrCaptureSource::SystemAudio => {
            let tap = create_system_tap(device_id, options.application_bundle_ids.as_deref())?;
            (tap.aggregate_device_id, Some(tap))
        }
    };

    let mut audio_unit = audio_unit_from_device_id_uninitialized(capture_device_id, true)
        .map_err(|error| native_error("create Core Audio input unit", error))?;
    audio_unit
        .set_stream_format(
            StreamFormat {
                sample_rate: device.sample_rate as f64,
                sample_format: SampleFormat::F32,
                flags: LinearPcmFlags::IS_FLOAT | LinearPcmFlags::IS_PACKED,
                channels: 1,
            },
            Scope::Output,
            Element::Input,
        )
        .map_err(|error| native_error("configure Core Audio capture format", error))?;

    type Args = render_callback::Args<data::Interleaved<f32>>;
    audio_unit
        .set_input_callback(move |args: Args| {
            on_samples(args.data.buffer);
            Ok(())
        })
        .map_err(|error| native_error("install Core Audio capture callback", error))?;
    audio_unit
        .initialize()
        .map_err(|error| native_error("initialize Core Audio capture", error))?;

    Ok((
        source,
        NativeCapture {
            audio_unit,
            _system_tap: system_tap,
            device: NativeDeviceInfo {
                channels: 1,
                ..device
            },
        },
    ))
}

fn source_scope(source: Qwen3AsrCaptureSource) -> Scope {
    match source {
        Qwen3AsrCaptureSource::Microphone => Scope::Input,
        Qwen3AsrCaptureSource::SystemAudio => Scope::Output,
    }
}

fn select_device(
    source: Qwen3AsrCaptureSource,
    options: &Qwen3AsrCaptureOptions,
) -> Result<AudioObjectID> {
    let scope = source_scope(source);
    let devices =
        get_audio_device_ids().map_err(|error| native_error("list Core Audio devices", error))?;

    if let Some(uid) = options.device_id.as_deref() {
        return devices
            .into_iter()
            .find(|&id| {
                get_audio_device_supports_scope(id, scope).unwrap_or(false)
                    && device_uid_string(id).is_ok_and(|candidate| candidate == uid)
            })
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "Core Audio {} device is unavailable: {uid}",
                    source_label(source)
                ))
            });
    }
    if let Some(name) = options.device_name.as_deref() {
        return devices
            .into_iter()
            .find(|&id| {
                get_audio_device_supports_scope(id, scope).unwrap_or(false)
                    && get_device_name(id).is_ok_and(|candidate| candidate == name)
            })
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "Core Audio {} device not found: {name}",
                    source_label(source)
                ))
            });
    }
    default_device_id(source).ok_or_else(|| {
        Error::from_reason(format!(
            "No default Core Audio {} device is available",
            source_label(source)
        ))
    })
}

fn source_label(source: Qwen3AsrCaptureSource) -> &'static str {
    match source {
        Qwen3AsrCaptureSource::Microphone => "microphone",
        Qwen3AsrCaptureSource::SystemAudio => "output",
    }
}

fn device_info(id: AudioObjectID, source: Qwen3AsrCaptureSource) -> Result<NativeDeviceInfo> {
    let sample_rate = property_value::<f64>(
        id,
        kAudioDevicePropertyNominalSampleRate,
        kAudioObjectPropertyScopeGlobal,
    )?;
    if !sample_rate.is_finite() || sample_rate <= 0.0 || sample_rate > u32::MAX as f64 {
        return Err(Error::from_reason(
            "Core Audio returned an invalid sample rate",
        ));
    }
    let channels = device_channels(
        id,
        match source {
            Qwen3AsrCaptureSource::Microphone => kAudioObjectPropertyScopeInput,
            Qwen3AsrCaptureSource::SystemAudio => kAudioObjectPropertyScopeOutput,
        },
    )?;
    Ok(NativeDeviceInfo {
        object_id: id,
        id: device_uid_string(id)?,
        name: get_device_name(id)
            .map_err(|error| native_error("read Core Audio device name", error))?,
        sample_rate: sample_rate.round() as u32,
        channels,
    })
}

fn property_value<T: Copy>(object_id: AudioObjectID, selector: u32, scope: u32) -> Result<T> {
    let address = AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain,
    };
    let mut value = MaybeUninit::<T>::uninit();
    let mut size = size_of::<T>() as u32;
    let status = unsafe {
        AudioObjectGetPropertyData(
            object_id,
            NonNull::from(&address),
            0,
            null(),
            NonNull::from(&mut size),
            NonNull::new(value.as_mut_ptr())
                .expect("MaybeUninit pointer")
                .cast(),
        )
    };
    check_status("read Core Audio property", status)?;
    if size as usize != size_of::<T>() {
        return Err(Error::from_reason(
            "Core Audio returned an invalid property size",
        ));
    }
    Ok(unsafe { value.assume_init() })
}

fn device_uid(id: AudioObjectID) -> Result<Retained<NSString>> {
    let pointer = property_value::<*mut c_void>(
        id,
        kAudioDevicePropertyDeviceUID,
        kAudioObjectPropertyScopeGlobal,
    )?;
    if pointer.is_null() {
        return Err(Error::from_reason("Core Audio device has no UID"));
    }
    unsafe { Retained::retain(pointer.cast::<NSString>()) }
        .ok_or_else(|| Error::from_reason("Core Audio device UID was released"))
}

fn device_uid_string(id: AudioObjectID) -> Result<String> {
    Ok(device_uid(id)?.to_string())
}

fn device_channels(id: AudioObjectID, scope: u32) -> Result<u32> {
    let address = AudioObjectPropertyAddress {
        mSelector: kAudioDevicePropertyStreamConfiguration,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain,
    };
    let mut size = 0u32;
    let status = unsafe {
        AudioObjectGetPropertyDataSize(
            id,
            NonNull::from(&address),
            0,
            null(),
            NonNull::from(&mut size),
        )
    };
    check_status("read Core Audio channel layout size", status)?;
    if size < size_of::<AudioBufferList>() as u32 {
        return Err(Error::from_reason(
            "Core Audio returned an invalid channel layout size",
        ));
    }
    // AudioObjectGetPropertyData writes an AudioBufferList whose alignment is
    // wider than `u8`. Back the byte-sized result with machine words so the
    // cast below is aligned even though the list has a variable-length tail.
    let words = (size as usize).div_ceil(size_of::<usize>());
    let mut storage = vec![0usize; words];
    let status = unsafe {
        AudioObjectGetPropertyData(
            id,
            NonNull::from(&address),
            0,
            null(),
            NonNull::from(&mut size),
            NonNull::new(storage.as_mut_ptr())
                .expect("non-empty channel layout")
                .cast(),
        )
    };
    check_status("read Core Audio channel layout", status)?;
    let list = unsafe { &*(storage.as_ptr().cast::<AudioBufferList>()) };
    let buffer_offset = std::mem::offset_of!(AudioBufferList, mBuffers);
    let max_buffers = (size as usize - buffer_offset) / size_of::<AudioBuffer>();
    if list.mNumberBuffers as usize > max_buffers {
        return Err(Error::from_reason(
            "Core Audio returned an invalid channel buffer count",
        ));
    }
    let buffers = unsafe {
        std::slice::from_raw_parts(
            (&raw const list.mBuffers).cast::<AudioBuffer>(),
            list.mNumberBuffers as usize,
        )
    };
    Ok(buffers.iter().map(|buffer| buffer.mNumberChannels).sum())
}

fn create_system_tap(
    output_device_id: AudioObjectID,
    application_bundle_ids: Option<&[String]>,
) -> Result<SystemAudioTap> {
    let bundle_ids = application_bundle_ids.unwrap_or_default();
    if bundle_ids.iter().any(|id| id.trim().is_empty()) {
        return Err(Error::from_reason(
            "application_bundle_ids must not contain empty identifiers",
        ));
    }

    let device_uid = device_uid(output_device_id)?;
    let process_numbers = process_ids_for_bundle_ids(bundle_ids)?
        .into_iter()
        .map(NSNumber::new_u32)
        .collect::<Vec<_>>();
    let processes = NSArray::from_retained_slice(&process_numbers);
    let tap = if bundle_ids.is_empty() {
        unsafe {
            CATapDescription::initExcludingProcesses_andDeviceUID_withStream(
                CATapDescription::alloc(),
                &processes,
                &device_uid,
                0,
            )
        }
    } else {
        let tap = unsafe {
            CATapDescription::initWithProcesses_andDeviceUID_withStream(
                CATapDescription::alloc(),
                &processes,
                &device_uid,
                0,
            )
        };
        let strings = bundle_ids
            .iter()
            .map(|id| NSString::from_str(id))
            .collect::<Vec<_>>();
        let array = NSArray::from_retained_slice(&strings);
        unsafe {
            tap.setBundleIDs(&array);
            tap.setProcessRestoreEnabled(true);
        }
        tap
    };

    let process_id = std::process::id();
    let instance = CAPTURE_INSTANCE.fetch_add(1, Ordering::Relaxed);
    unsafe {
        tap.setName(&NSString::from_str(&format!(
            "mlx-node system audio {process_id}.{instance}"
        )));
        tap.setPrivate(true);
        tap.setExclusive(bundle_ids.is_empty());
        tap.setMono(true);
        tap.setMixdown(true);
        tap.setMuteBehavior(CATapMuteBehavior::Unmuted);
    }

    let mut tap_id = MaybeUninit::<AudioObjectID>::uninit();
    let status = unsafe { AudioHardwareCreateProcessTap(Some(&tap), tap_id.as_mut_ptr()) };
    check_status(
        "create the Core Audio system tap (the host app needs audio capture permission)",
        status,
    )?;
    let tap_id = unsafe { tap_id.assume_init() };
    let tap_uid = unsafe { tap.UUID().UUIDString() };
    let aggregate_uid = format!("ai.mlxnode.asr.system-audio.{process_id}.{instance}");
    let properties = aggregate_properties(
        tap_uid,
        &aggregate_uid,
        &format!("mlx-node system audio {process_id}.{instance}"),
    );
    let mut aggregate_device_id = 0;
    let status = unsafe {
        AudioHardwareCreateAggregateDevice(
            properties.as_ref(),
            NonNull::from(&mut aggregate_device_id),
        )
    };
    if let Err(error) = check_status("create the Core Audio tap aggregate device", status) {
        unsafe {
            let _ = AudioHardwareDestroyProcessTap(tap_id);
        }
        return Err(error);
    }

    Ok(SystemAudioTap {
        aggregate_device_id,
        tap_id,
    })
}

fn process_ids_for_bundle_ids(bundle_ids: &[String]) -> Result<Vec<AudioObjectID>> {
    if bundle_ids.is_empty() {
        return Ok(Vec::new());
    }
    let address = AudioObjectPropertyAddress {
        mSelector: kAudioHardwarePropertyProcessObjectList,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    };
    let mut size = 0u32;
    let status = unsafe {
        AudioObjectGetPropertyDataSize(
            kAudioObjectSystemObject as AudioObjectID,
            NonNull::from(&address),
            0,
            null(),
            NonNull::from(&mut size),
        )
    };
    check_status("list Core Audio process objects", status)?;
    if !(size as usize).is_multiple_of(size_of::<AudioObjectID>()) {
        return Err(Error::from_reason(
            "Core Audio returned an invalid process object list size",
        ));
    }
    let count = size as usize / size_of::<AudioObjectID>();
    let mut process_ids = vec![0; count];
    let status = unsafe {
        AudioObjectGetPropertyData(
            kAudioObjectSystemObject as AudioObjectID,
            NonNull::from(&address),
            0,
            null(),
            NonNull::from(&mut size),
            NonNull::new(process_ids.as_mut_ptr())
                .expect("process object list pointer")
                .cast(),
        )
    };
    check_status("read Core Audio process objects", status)?;
    process_ids
        .retain(|&id| process_bundle_id(id).is_ok_and(|bundle_id| bundle_ids.contains(&bundle_id)));
    Ok(process_ids)
}

fn process_bundle_id(id: AudioObjectID) -> Result<String> {
    let pointer = property_value::<*mut c_void>(
        id,
        kAudioProcessPropertyBundleID,
        kAudioObjectPropertyScopeGlobal,
    )?;
    if pointer.is_null() {
        return Err(Error::from_reason("Core Audio process has no bundle ID"));
    }
    let string = unsafe { Retained::retain(pointer.cast::<NSString>()) }
        .ok_or_else(|| Error::from_reason("Core Audio process bundle ID was released"))?;
    Ok(string.to_string())
}

fn to_cfstring(value: &'static CStr) -> CFRetained<CFString> {
    unsafe {
        CFString::with_c_string(
            kCFAllocatorDefault,
            value.as_ptr(),
            0x0800_0100, // kCFStringEncodingUTF8
        )
    }
    .expect("Core Audio dictionary key")
}

fn aggregate_properties(
    tap_uid: Retained<NSString>,
    aggregate_uid: &str,
    aggregate_name: &str,
) -> CFRetained<CFDictionary> {
    let tap_dictionary = unsafe {
        let dictionary = CFMutableDictionary::new(
            kCFAllocatorDefault,
            2,
            &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks,
        )
        .expect("tap dictionary");
        CFMutableDictionary::set_value(
            Some(dictionary.as_ref()),
            &*to_cfstring(kAudioSubTapUIDKey) as *const _ as *const c_void,
            &*tap_uid as *const _ as *const c_void,
        );
        CFMutableDictionary::set_value(
            Some(dictionary.as_ref()),
            &*to_cfstring(kAudioSubTapDriftCompensationKey) as *const _ as *const c_void,
            &*NSNumber::new_bool(true) as *const _ as *const c_void,
        );
        dictionary
    };
    let tap_dictionaries = [tap_dictionary];
    let tap_list = unsafe {
        CFArray::new(
            kCFAllocatorDefault,
            tap_dictionaries.as_ptr().cast::<*const c_void>().cast_mut(),
            tap_dictionaries.len() as isize,
            &kCFTypeArrayCallBacks,
        )
        .expect("tap list")
    };
    unsafe {
        let dictionary = CFMutableDictionary::new(
            kCFAllocatorDefault,
            5,
            &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks,
        )
        .expect("aggregate dictionary");
        let aggregate_name = CFString::from_str(aggregate_name);
        let aggregate_uid = CFString::from_str(aggregate_uid);
        let auto_start = NSNumber::new_bool(true);
        let private = NSNumber::new_bool(true);
        let values: [(&'static CStr, *const c_void); 5] = [
            (
                kAudioAggregateDeviceNameKey,
                &*aggregate_name as *const _ as *const c_void,
            ),
            (
                kAudioAggregateDeviceUIDKey,
                &*aggregate_uid as *const _ as *const c_void,
            ),
            (
                kAudioAggregateDeviceTapListKey,
                &*tap_list as *const _ as *const c_void,
            ),
            (
                kAudioAggregateDeviceTapAutoStartKey,
                &*auto_start as *const _ as *const c_void,
            ),
            (
                kAudioAggregateDeviceIsPrivateKey,
                &*private as *const _ as *const c_void,
            ),
        ];
        for (key, value) in values {
            CFMutableDictionary::set_value(
                Some(dictionary.as_ref()),
                &*to_cfstring(key) as *const _ as *const c_void,
                value,
            );
        }
        CFRetained::cast_unchecked::<CFDictionary>(dictionary)
    }
}

fn check_status(action: &str, status: i32) -> Result<()> {
    coreaudio::Error::from_os_status(status).map_err(|error| native_error(action, error))
}

fn native_error(action: &str, error: coreaudio::Error) -> Error {
    Error::from_reason(format!("Failed to {action}: {error}"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    fn capture_frames(
        source: Qwen3AsrCaptureSource,
        application_bundle_ids: Option<Vec<String>>,
    ) -> Result<usize> {
        let frames = Arc::new(AtomicUsize::new(0));
        let callback_frames = frames.clone();
        let options = Qwen3AsrCaptureOptions {
            source: Some(source),
            application_bundle_ids,
            ..Default::default()
        };
        let (_, mut capture) = build(&options, move |samples| {
            callback_frames.fetch_add(samples.len(), Ordering::Relaxed);
        })?;
        capture.start()?;
        std::thread::sleep(Duration::from_secs(3));
        capture.stop()?;
        Ok(frames.load(Ordering::Relaxed))
    }

    #[test]
    #[ignore = "requires a live microphone and macOS privacy permission"]
    fn captures_live_microphone_frames() -> Result<()> {
        assert!(capture_frames(Qwen3AsrCaptureSource::Microphone, None)? > 0);
        Ok(())
    }

    #[test]
    #[ignore = "requires audible system output and macOS privacy permission"]
    fn captures_live_system_audio_frames() -> Result<()> {
        assert!(capture_frames(Qwen3AsrCaptureSource::SystemAudio, None)? > 0);
        Ok(())
    }
}
