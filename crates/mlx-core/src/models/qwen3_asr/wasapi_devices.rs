//! WASAPI endpoint discovery for Qwen3-ASR device selection.

use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::{NonNull, null, null_mut};

use napi::bindgen_prelude::{Error, Result};
use windows_sys::Win32::Foundation::PROPERTYKEY;
use windows_sys::Win32::Media::Audio::{WAVEFORMATEX, WAVEFORMATEXTENSIBLE};
use windows_sys::Win32::System::Com::StructuredStorage::{PROPVARIANT, PropVariantClear};
use windows_sys::Win32::System::Com::{
    CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree,
    CoUninitialize, STGM_READ,
};
use windows_sys::Win32::System::Variant::{VT_BLOB, VT_LPWSTR};
use windows_sys::core::{GUID, HRESULT};

use super::capture::{Qwen3AsrAudioDevice, Qwen3AsrInputDevice};
use super::config::Qwen3AsrCaptureSource;

const E_RENDER: i32 = 0;
const E_CAPTURE: i32 = 1;
const E_CONSOLE: i32 = 0;
const DEVICE_STATE_ACTIVE: u32 = 1;
const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xfffe;
const RPC_E_CHANGED_MODE: HRESULT = 0x80010106u32 as HRESULT;
const AUDCLNT_E_DEVICE_INVALIDATED: HRESULT = 0x88890004u32 as HRESULT;
const HRESULT_ERROR_FILE_NOT_FOUND: HRESULT = 0x80070002u32 as HRESULT;
const HRESULT_ERROR_PATH_NOT_FOUND: HRESULT = 0x80070003u32 as HRESULT;
const HRESULT_ERROR_NOT_FOUND: HRESULT = 0x80070490u32 as HRESULT;

const CLSID_MM_DEVICE_ENUMERATOR: GUID = GUID::from_u128(0xbcde0395_e52f_467c_8e3d_c4579291692e);
const IID_MM_DEVICE_ENUMERATOR: GUID = GUID::from_u128(0xa95664d2_9614_4f35_a746_de8db63617e6);
const KSDATAFORMAT_SUBTYPE_PCM: GUID = GUID::from_u128(0x00000001_0000_0010_8000_00aa00389b71);
const KSDATAFORMAT_SUBTYPE_IEEE_FLOAT: GUID =
    GUID::from_u128(0x00000003_0000_0010_8000_00aa00389b71);
const PKEY_DEVICE_FRIENDLY_NAME: PROPERTYKEY = PROPERTYKEY {
    fmtid: GUID::from_u128(0xa45c254e_df1c_4efd_8020_67d146a850e0),
    pid: 14,
};
const PKEY_AUDIOENGINE_DEVICE_FORMAT: PROPERTYKEY = PROPERTYKEY {
    fmtid: GUID::from_u128(0xf19f064d_082c_4e27_bc73_6882a1bb8e4c),
    pid: 0,
};

type NativeResult<T> = std::result::Result<T, WasapiError>;

#[derive(Debug)]
struct WasapiError {
    message: String,
    hresult: Option<HRESULT>,
}

impl WasapiError {
    fn hresult(operation: &str, hresult: HRESULT) -> Self {
        Self {
            message: format!("{operation} failed with HRESULT 0x{:08x}", hresult as u32),
            hresult: Some(hresult),
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            hresult: None,
        }
    }

    fn is_missing_device(&self) -> bool {
        self.hresult.is_some_and(is_missing_device_hresult)
    }

    fn into_napi_error(self) -> Error {
        Error::from_reason(self.message)
    }
}

fn check_hresult(operation: &str, hresult: HRESULT) -> NativeResult<()> {
    if hresult >= 0 {
        Ok(())
    } else {
        Err(WasapiError::hresult(operation, hresult))
    }
}

fn guid_eq(a: &GUID, b: &GUID) -> bool {
    a.data1 == b.data1 && a.data2 == b.data2 && a.data3 == b.data3 && a.data4 == b.data4
}

fn is_missing_device_hresult(hresult: HRESULT) -> bool {
    matches!(
        hresult,
        AUDCLNT_E_DEVICE_INVALIDATED
            | HRESULT_ERROR_FILE_NOT_FOUND
            | HRESULT_ERROR_PATH_NOT_FOUND
            | HRESULT_ERROR_NOT_FOUND
    )
}

#[repr(C)]
struct IUnknownVtbl {
    _query_interface: unsafe extern "system" fn(
        this: *mut c_void,
        iid: *const GUID,
        object: *mut *mut c_void,
    ) -> HRESULT,
    _add_ref: unsafe extern "system" fn(this: *mut c_void) -> u32,
    release: unsafe extern "system" fn(this: *mut c_void) -> u32,
}

#[repr(C)]
struct DeviceEnumerator {
    _lp_vtbl: *const DeviceEnumeratorVtbl,
}

#[repr(C)]
struct DeviceEnumeratorVtbl {
    _query_interface: unsafe extern "system" fn(
        this: *mut DeviceEnumerator,
        iid: *const GUID,
        object: *mut *mut c_void,
    ) -> HRESULT,
    _add_ref: unsafe extern "system" fn(this: *mut DeviceEnumerator) -> u32,
    _release: unsafe extern "system" fn(this: *mut DeviceEnumerator) -> u32,
    enum_audio_endpoints: unsafe extern "system" fn(
        this: *mut DeviceEnumerator,
        data_flow: i32,
        state_mask: u32,
        devices: *mut *mut DeviceCollection,
    ) -> HRESULT,
    get_default_audio_endpoint: unsafe extern "system" fn(
        this: *mut DeviceEnumerator,
        data_flow: i32,
        role: i32,
        device: *mut *mut Device,
    ) -> HRESULT,
}

#[repr(C)]
struct DeviceCollection {
    _lp_vtbl: *const DeviceCollectionVtbl,
}

#[repr(C)]
struct DeviceCollectionVtbl {
    _query_interface: unsafe extern "system" fn(
        this: *mut DeviceCollection,
        iid: *const GUID,
        object: *mut *mut c_void,
    ) -> HRESULT,
    _add_ref: unsafe extern "system" fn(this: *mut DeviceCollection) -> u32,
    _release: unsafe extern "system" fn(this: *mut DeviceCollection) -> u32,
    get_count: unsafe extern "system" fn(this: *mut DeviceCollection, count: *mut u32) -> HRESULT,
    item: unsafe extern "system" fn(
        this: *mut DeviceCollection,
        index: u32,
        device: *mut *mut Device,
    ) -> HRESULT,
}

#[repr(C)]
struct Device {
    _lp_vtbl: *const DeviceVtbl,
}

#[repr(C)]
struct DeviceVtbl {
    _query_interface: unsafe extern "system" fn(
        this: *mut Device,
        iid: *const GUID,
        object: *mut *mut c_void,
    ) -> HRESULT,
    _add_ref: unsafe extern "system" fn(this: *mut Device) -> u32,
    _release: unsafe extern "system" fn(this: *mut Device) -> u32,
    _activate: unsafe extern "system" fn(
        this: *mut Device,
        iid: *const GUID,
        cls_ctx: u32,
        activation_params: *mut PROPVARIANT,
        interface: *mut *mut c_void,
    ) -> HRESULT,
    open_property_store: unsafe extern "system" fn(
        this: *mut Device,
        access: u32,
        properties: *mut *mut PropertyStore,
    ) -> HRESULT,
    get_id: unsafe extern "system" fn(this: *mut Device, id: *mut *mut u16) -> HRESULT,
}

#[repr(C)]
struct PropertyStore {
    _lp_vtbl: *const PropertyStoreVtbl,
}

#[repr(C)]
struct PropertyStoreVtbl {
    _query_interface: unsafe extern "system" fn(
        this: *mut PropertyStore,
        iid: *const GUID,
        object: *mut *mut c_void,
    ) -> HRESULT,
    _add_ref: unsafe extern "system" fn(this: *mut PropertyStore) -> u32,
    _release: unsafe extern "system" fn(this: *mut PropertyStore) -> u32,
    _get_count: unsafe extern "system" fn(this: *mut PropertyStore, count: *mut u32) -> HRESULT,
    _get_at: unsafe extern "system" fn(
        this: *mut PropertyStore,
        index: u32,
        key: *mut PROPERTYKEY,
    ) -> HRESULT,
    get_value: unsafe extern "system" fn(
        this: *mut PropertyStore,
        key: *const PROPERTYKEY,
        value: *mut PROPVARIANT,
    ) -> HRESULT,
}

struct ComPtr<T>(NonNull<T>);

impl<T> ComPtr<T> {
    fn new(pointer: *mut T, operation: &str) -> NativeResult<Self> {
        NonNull::new(pointer)
            .map(Self)
            .ok_or_else(|| WasapiError::invalid(format!("{operation} returned a null interface")))
    }
}

impl<T> Drop for ComPtr<T> {
    fn drop(&mut self) {
        unsafe {
            let vtable = self.0.as_ptr().cast::<*const IUnknownVtbl>().read();
            ((*vtable).release)(self.0.as_ptr().cast());
        }
    }
}

unsafe fn interface_vtable<T, V>(object: &ComPtr<T>) -> &V {
    let vtable = unsafe { object.0.as_ptr().cast::<*const V>().read() };
    unsafe { &*vtable }
}

struct ComApartment {
    uninitialize: bool,
}

impl ComApartment {
    fn initialize() -> NativeResult<Self> {
        let hresult = unsafe { CoInitializeEx(null(), COINIT_MULTITHREADED as u32) };
        if hresult >= 0 {
            Ok(Self { uninitialize: true })
        } else if hresult == RPC_E_CHANGED_MODE {
            // Existing single-threaded apartments can still enumerate WASAPI endpoints.
            Ok(Self {
                uninitialize: false,
            })
        } else {
            Err(WasapiError::hresult("initialize COM", hresult))
        }
    }
}

impl Drop for ComApartment {
    fn drop(&mut self) {
        if self.uninitialize {
            unsafe { CoUninitialize() };
        }
    }
}

struct CoTaskMemWideString(*mut u16);

impl CoTaskMemWideString {
    fn to_string(&self) -> NativeResult<String> {
        wide_string(self.0)
    }
}

impl Drop for CoTaskMemWideString {
    fn drop(&mut self) {
        unsafe { CoTaskMemFree(self.0.cast::<c_void>()) };
    }
}

struct PropertyValue(PROPVARIANT);

impl PropertyValue {
    fn new() -> Self {
        Self(PROPVARIANT::default())
    }

    fn string(&self) -> NativeResult<String> {
        let header = unsafe { self.0.Anonymous.Anonymous };
        if header.vt != VT_LPWSTR {
            return Err(WasapiError::invalid(
                "WASAPI device name is not a wide string",
            ));
        }
        wide_string(unsafe { header.Anonymous.pwszVal })
    }

    fn bytes(&self) -> NativeResult<Vec<u8>> {
        let header = unsafe { self.0.Anonymous.Anonymous };
        if header.vt != VT_BLOB {
            return Err(WasapiError::invalid(
                "WASAPI device format is not a binary property",
            ));
        }
        let blob = unsafe { header.Anonymous.blob };
        if blob.pBlobData.is_null() || (blob.cbSize as usize) < size_of::<WAVEFORMATEX>() {
            return Err(WasapiError::invalid(
                "WASAPI returned an invalid device format",
            ));
        }
        Ok(unsafe { std::slice::from_raw_parts(blob.pBlobData, blob.cbSize as usize).to_vec() })
    }
}

impl Drop for PropertyValue {
    fn drop(&mut self) {
        unsafe {
            let _ = PropVariantClear(&mut self.0);
        }
    }
}

fn wide_string(pointer: *mut u16) -> NativeResult<String> {
    if pointer.is_null() {
        return Err(WasapiError::invalid("WASAPI returned a null string"));
    }
    let mut length = 0usize;
    unsafe {
        while *pointer.add(length) != 0 {
            length += 1;
        }
        String::from_utf16(std::slice::from_raw_parts(pointer, length)).map_err(|error| {
            WasapiError::invalid(format!("WASAPI returned invalid UTF-16: {error}"))
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EndpointKind {
    Capture,
    Render,
}

impl EndpointKind {
    fn flow(self) -> i32 {
        match self {
            Self::Capture => E_CAPTURE,
            Self::Render => E_RENDER,
        }
    }

    fn source(self) -> Qwen3AsrCaptureSource {
        match self {
            Self::Capture => Qwen3AsrCaptureSource::Microphone,
            Self::Render => Qwen3AsrCaptureSource::SystemAudio,
        }
    }
}

#[derive(Debug)]
struct EndpointInfo {
    id: String,
    name: String,
    kind: EndpointKind,
    is_default: bool,
    sample_rate: u32,
    channels: u32,
    sample_format: String,
}

#[derive(Debug, PartialEq, Eq)]
struct FormatInfo {
    sample_rate: u32,
    channels: u32,
    sample_format: String,
}

pub(super) fn input_devices() -> Result<Vec<Qwen3AsrInputDevice>> {
    Ok(enumerate(&[EndpointKind::Capture])
        .map_err(WasapiError::into_napi_error)?
        .into_iter()
        .map(|device| Qwen3AsrInputDevice {
            id: device.id,
            name: device.name,
            is_default: device.is_default,
            sample_rate: device.sample_rate,
            channels: device.channels,
            sample_format: device.sample_format,
        })
        .collect())
}

pub(super) fn audio_devices() -> Result<Vec<Qwen3AsrAudioDevice>> {
    Ok(enumerate(&[EndpointKind::Capture, EndpointKind::Render])
        .map_err(WasapiError::into_napi_error)?
        .into_iter()
        .map(|device| Qwen3AsrAudioDevice {
            id: device.id,
            name: device.name,
            source: device.kind.source(),
            is_default: device.is_default,
            sample_rate: device.sample_rate,
            channels: device.channels,
        })
        .collect())
}

fn enumerate(kinds: &[EndpointKind]) -> NativeResult<Vec<EndpointInfo>> {
    let _apartment = ComApartment::initialize()?;
    let enumerator = create_enumerator()?;
    let mut devices = Vec::new();
    for &kind in kinds {
        devices.extend(enumerate_kind(&enumerator, kind)?);
    }
    Ok(devices)
}

fn create_enumerator() -> NativeResult<ComPtr<DeviceEnumerator>> {
    let mut enumerator: *mut c_void = null_mut();
    let hresult = unsafe {
        CoCreateInstance(
            &CLSID_MM_DEVICE_ENUMERATOR,
            null_mut(),
            CLSCTX_ALL,
            &IID_MM_DEVICE_ENUMERATOR,
            &mut enumerator,
        )
    };
    check_hresult("create WASAPI device enumerator", hresult)?;
    ComPtr::new(enumerator.cast(), "create WASAPI device enumerator")
}

fn enumerate_kind(
    enumerator: &ComPtr<DeviceEnumerator>,
    kind: EndpointKind,
) -> NativeResult<Vec<EndpointInfo>> {
    let default_id = default_endpoint_id(enumerator, kind)?;
    let collection = endpoint_collection(enumerator, kind)?;
    let mut count = 0;
    let hresult = unsafe {
        (interface_vtable::<_, DeviceCollectionVtbl>(&collection).get_count)(
            collection.0.as_ptr(),
            &mut count,
        )
    };
    check_hresult("count WASAPI endpoints", hresult)?;

    let mut devices = Vec::with_capacity(count as usize);
    for index in 0..count {
        let device = match collection_item(&collection, index) {
            Ok(device) => device,
            Err(error) if error.is_missing_device() => continue,
            Err(error) => return Err(error),
        };
        match endpoint_info(&device, kind) {
            Ok(mut info) => {
                info.is_default = default_id.as_deref() == Some(info.id.as_str());
                devices.push(info);
            }
            Err(error) if error.is_missing_device() => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(devices)
}

fn endpoint_collection(
    enumerator: &ComPtr<DeviceEnumerator>,
    kind: EndpointKind,
) -> NativeResult<ComPtr<DeviceCollection>> {
    let mut collection = null_mut();
    let hresult = unsafe {
        (interface_vtable::<_, DeviceEnumeratorVtbl>(enumerator).enum_audio_endpoints)(
            enumerator.0.as_ptr(),
            kind.flow(),
            DEVICE_STATE_ACTIVE,
            &mut collection,
        )
    };
    check_hresult("enumerate active WASAPI endpoints", hresult)?;
    ComPtr::new(collection, "enumerate active WASAPI endpoints")
}

fn collection_item(
    collection: &ComPtr<DeviceCollection>,
    index: u32,
) -> NativeResult<ComPtr<Device>> {
    let mut device = null_mut();
    let hresult = unsafe {
        (interface_vtable::<_, DeviceCollectionVtbl>(collection).item)(
            collection.0.as_ptr(),
            index,
            &mut device,
        )
    };
    check_hresult("read WASAPI endpoint", hresult)?;
    ComPtr::new(device, "read WASAPI endpoint")
}

fn default_endpoint_id(
    enumerator: &ComPtr<DeviceEnumerator>,
    kind: EndpointKind,
) -> NativeResult<Option<String>> {
    let mut device = null_mut();
    let hresult = unsafe {
        (interface_vtable::<_, DeviceEnumeratorVtbl>(enumerator).get_default_audio_endpoint)(
            enumerator.0.as_ptr(),
            kind.flow(),
            E_CONSOLE,
            &mut device,
        )
    };
    if is_missing_device_hresult(hresult) {
        return Ok(None);
    }
    check_hresult("get default WASAPI endpoint", hresult)?;
    let device = ComPtr::new(device, "get default WASAPI endpoint")?;
    match endpoint_id(&device) {
        Ok(id) => Ok(Some(id)),
        Err(error) if error.is_missing_device() => Ok(None),
        Err(error) => Err(error),
    }
}

fn endpoint_id(device: &ComPtr<Device>) -> NativeResult<String> {
    let mut id = null_mut();
    let hresult =
        unsafe { (interface_vtable::<_, DeviceVtbl>(device).get_id)(device.0.as_ptr(), &mut id) };
    check_hresult("read WASAPI endpoint ID", hresult)?;
    if id.is_null() {
        return Err(WasapiError::invalid("WASAPI returned a null endpoint ID"));
    }
    CoTaskMemWideString(id).to_string()
}

fn endpoint_info(device: &ComPtr<Device>, kind: EndpointKind) -> NativeResult<EndpointInfo> {
    let id = endpoint_id(device)?;
    let properties = open_property_store(device)?;
    let name = property(&properties, &PKEY_DEVICE_FRIENDLY_NAME)?.string()?;
    let bytes = property(&properties, &PKEY_AUDIOENGINE_DEVICE_FORMAT)?.bytes()?;
    let format = parse_wave_format(&bytes)?;
    Ok(EndpointInfo {
        id,
        name,
        kind,
        is_default: false,
        sample_rate: format.sample_rate,
        channels: format.channels,
        sample_format: format.sample_format,
    })
}

fn open_property_store(device: &ComPtr<Device>) -> NativeResult<ComPtr<PropertyStore>> {
    let mut properties = null_mut();
    let hresult = unsafe {
        (interface_vtable::<_, DeviceVtbl>(device).open_property_store)(
            device.0.as_ptr(),
            STGM_READ,
            &mut properties,
        )
    };
    check_hresult("open WASAPI endpoint properties", hresult)?;
    ComPtr::new(properties, "open WASAPI endpoint properties")
}

fn property(store: &ComPtr<PropertyStore>, key: &PROPERTYKEY) -> NativeResult<PropertyValue> {
    let mut value = PropertyValue::new();
    let hresult = unsafe {
        (interface_vtable::<_, PropertyStoreVtbl>(store).get_value)(
            store.0.as_ptr(),
            key,
            &mut value.0,
        )
    };
    check_hresult("read WASAPI endpoint property", hresult)?;
    Ok(value)
}

fn parse_wave_format(bytes: &[u8]) -> NativeResult<FormatInfo> {
    if bytes.len() < size_of::<WAVEFORMATEX>() {
        return Err(WasapiError::invalid("WASAPI device format is truncated"));
    }
    let format = unsafe { bytes.as_ptr().cast::<WAVEFORMATEX>().read_unaligned() };
    let mut format_tag = format.wFormatTag;
    if format_tag == WAVE_FORMAT_EXTENSIBLE {
        let extension_size = size_of::<WAVEFORMATEXTENSIBLE>() - size_of::<WAVEFORMATEX>();
        if (format.cbSize as usize) < extension_size
            || bytes.len() < size_of::<WAVEFORMATEXTENSIBLE>()
        {
            return Err(WasapiError::invalid(
                "WASAPI extensible device format is truncated",
            ));
        }
        let extensible = unsafe {
            bytes
                .as_ptr()
                .cast::<WAVEFORMATEXTENSIBLE>()
                .read_unaligned()
        };
        let subtype = extensible.SubFormat;
        if guid_eq(&subtype, &KSDATAFORMAT_SUBTYPE_PCM) {
            format_tag = WAVE_FORMAT_PCM;
        } else if guid_eq(&subtype, &KSDATAFORMAT_SUBTYPE_IEEE_FLOAT) {
            format_tag = WAVE_FORMAT_IEEE_FLOAT;
        }
    }

    let sample_rate = format.nSamplesPerSec;
    let channels = format.nChannels as u32;
    if sample_rate == 0 || channels == 0 {
        return Err(WasapiError::invalid(
            "WASAPI returned an invalid sample rate or channel count",
        ));
    }
    let bits = format.wBitsPerSample;
    let sample_format = match (format_tag, bits) {
        (WAVE_FORMAT_IEEE_FLOAT, 32) => "f32".to_string(),
        (WAVE_FORMAT_IEEE_FLOAT, 64) => "f64".to_string(),
        (WAVE_FORMAT_IEEE_FLOAT, _) => format!("ieee-float-{bits}"),
        (WAVE_FORMAT_PCM, 8) => "u8".to_string(),
        (WAVE_FORMAT_PCM, 16) => "i16".to_string(),
        (WAVE_FORMAT_PCM, 24) => "i24".to_string(),
        (WAVE_FORMAT_PCM, 32) => "i32".to_string(),
        (WAVE_FORMAT_PCM, _) => format!("pcm-{bits}"),
        _ => format!("wave-0x{format_tag:04x}"),
    };
    Ok(FormatInfo {
        sample_rate,
        channels,
        sample_format,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes_of<T: Copy>(value: &T) -> Vec<u8> {
        unsafe {
            std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()).to_vec()
        }
    }

    #[test]
    fn reports_pcm_and_extensible_float_formats() {
        let pcm = WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_PCM,
            nChannels: 1,
            nSamplesPerSec: 16_000,
            nAvgBytesPerSec: 32_000,
            nBlockAlign: 2,
            wBitsPerSample: 16,
            cbSize: 0,
        };
        assert_eq!(
            parse_wave_format(&bytes_of(&pcm)).unwrap(),
            FormatInfo {
                sample_rate: 16_000,
                channels: 1,
                sample_format: "i16".into(),
            }
        );

        let float = WAVEFORMATEXTENSIBLE {
            Format: WAVEFORMATEX {
                wFormatTag: WAVE_FORMAT_EXTENSIBLE,
                nChannels: 2,
                nSamplesPerSec: 48_000,
                nAvgBytesPerSec: 384_000,
                nBlockAlign: 8,
                wBitsPerSample: 32,
                cbSize: 22,
            },
            Samples: Default::default(),
            dwChannelMask: 3,
            SubFormat: KSDATAFORMAT_SUBTYPE_IEEE_FLOAT,
        };
        assert_eq!(
            parse_wave_format(&bytes_of(&float)).unwrap(),
            FormatInfo {
                sample_rate: 48_000,
                channels: 2,
                sample_format: "f32".into(),
            }
        );
    }

    #[test]
    fn rejects_invalid_or_truncated_formats() {
        assert!(parse_wave_format(&[0; 4]).is_err());
        let invalid = WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_IEEE_FLOAT,
            wBitsPerSample: 32,
            ..Default::default()
        };
        assert!(parse_wave_format(&bytes_of(&invalid)).is_err());
        let truncated = WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_EXTENSIBLE,
            nChannels: 2,
            nSamplesPerSec: 48_000,
            wBitsPerSample: 32,
            cbSize: 22,
            ..Default::default()
        };
        assert!(parse_wave_format(&bytes_of(&truncated)).is_err());
    }

    #[test]
    fn distinguishes_hotplug_races_from_other_device_errors() {
        for hresult in [
            AUDCLNT_E_DEVICE_INVALIDATED,
            HRESULT_ERROR_FILE_NOT_FOUND,
            HRESULT_ERROR_PATH_NOT_FOUND,
            HRESULT_ERROR_NOT_FOUND,
        ] {
            assert!(WasapiError::hresult("read endpoint", hresult).is_missing_device());
        }
        assert!(
            !WasapiError::hresult("read endpoint", 0x80070005u32 as HRESULT).is_missing_device()
        );
        assert!(!WasapiError::invalid("invalid format").is_missing_device());
    }
}
