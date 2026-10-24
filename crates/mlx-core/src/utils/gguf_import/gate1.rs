use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::gate::{manifest_owned, preload_input};
use super::{
    GgufPreload, LoadStats, PreloadFile, PreloadInput, ReadConfig, RepackBackend, Route, route_for,
};
use crate::array::clear_cache;
use crate::utils::gguf::{
    GgufFile, GgufTensorType, NativeGgufFamily, gguf_name_to_hf_for_metadata,
};
use crate::utils::gguf_gpu::{band_offset_rule, gpu_kind_for};
use crate::utils::gguf_plan::ImportPlan;
use crate::utils::gguf_read::plan_reads;

const FLIP_MODEL: &str = "qwen38-27b";
const REQUIRED_MODELS: [&str; 8] = [
    "qwen38-27b",
    "qwen36-35b-q4km",
    "qwen36-35b-iq4xs",
    "qwen36-35b-iq3s",
    "qwen36-35b-mxfp4",
    "gemma4-12b",
    "muse-30b",
    "clef-flash",
];
const GPU_TYPES: [GgufTensorType; 10] = [
    GgufTensorType::Q4K,
    GgufTensorType::Q5K,
    GgufTensorType::Q6K,
    GgufTensorType::IQ4XS,
    GgufTensorType::Q3K,
    GgufTensorType::IQ4NL,
    GgufTensorType::IQ3S,
    GgufTensorType::Q8_0,
    GgufTensorType::MXFP4,
    GgufTensorType::Q4_0,
];

type Manifest = BTreeMap<String, Value>;
type Manifests = BTreeMap<ManifestRole, Manifest>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ManifestRole {
    Main,
    Vision,
    Draft,
    JointHead,
}

impl ManifestRole {
    fn from_count_key(key: &str) -> Option<Self> {
        match key {
            "model" => Some(Self::Main),
            "vision" => Some(Self::Vision),
            "draft" => Some(Self::Draft),
            "joint_head" => Some(Self::JointHead),
            _ => None,
        }
    }

    fn count_key(self) -> &'static str {
        match self {
            Self::Main => "model",
            Self::Vision => "vision",
            Self::Draft => "draft",
            Self::JointHead => "joint_head",
        }
    }

    fn manifest_filename(self) -> &'static str {
        match self {
            Self::Main => "manifest.json",
            Self::Vision => "manifest-vision.json",
            Self::Draft => "manifest-draft.json",
            Self::JointHead => "manifest-joint_head.json",
        }
    }
}

#[derive(Debug)]
struct GateRow {
    model: String,
    gguf: PathBuf,
    family: NativeGgufFamily,
    companion: Option<PathBuf>,
    counts: BTreeMap<ManifestRole, usize>,
}

impl GateRow {
    fn roles(&self) -> Vec<ManifestRole> {
        self.counts.keys().copied().collect()
    }

    fn count(&self, role: ManifestRole) -> usize {
        self.counts[&role]
    }

    fn companion_role(&self) -> Option<ManifestRole> {
        companion_role(self.family)
    }
}

fn family_from_class(class: &str) -> Option<NativeGgufFamily> {
    match class {
        "Qwen35Model" => Some(NativeGgufFamily::Qwen35),
        "Qwen35MoeModel" => Some(NativeGgufFamily::Qwen35Moe),
        "Gemma4Model" => Some(NativeGgufFamily::Gemma4),
        "MuseGlimmerModel" => Some(NativeGgufFamily::MuseGlimmer),
        "ClefModel" => Some(NativeGgufFamily::Clef),
        _ => None,
    }
}

fn companion_role(family: NativeGgufFamily) -> Option<ManifestRole> {
    match family {
        NativeGgufFamily::Gemma4 => Some(ManifestRole::Vision),
        NativeGgufFamily::MuseGlimmer => Some(ManifestRole::Draft),
        _ => None,
    }
}

fn required_roles(family: NativeGgufFamily) -> Vec<ManifestRole> {
    let mut roles = vec![ManifestRole::Main];
    roles.extend(companion_role(family));
    if family == NativeGgufFamily::Clef {
        roles.push(ManifestRole::JointHead);
    }
    roles
}

fn parse_gate_list(text: &str) -> Result<Vec<GateRow>, String> {
    let mut rows = Vec::new();
    let mut models = BTreeSet::new();
    for (index, line) in text.lines().enumerate() {
        let line_number = index + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let columns: Vec<&str> = line.split('\t').map(str::trim).collect();
        if columns.len() != 5 {
            return Err(format!(
                "gate list line {line_number}: expected 5 tab-separated columns, got {}",
                columns.len()
            ));
        }
        if columns[..4].iter().any(|column| column.is_empty()) {
            return Err(format!(
                "gate list line {line_number}: model, GGUF, class, and companion columns must be non-empty"
            ));
        }
        let model = columns[0].to_string();
        if !models.insert(model.clone()) {
            return Err(format!(
                "gate list line {line_number}: duplicate model '{model}'"
            ));
        }
        let class = columns[2];
        let family = family_from_class(class).ok_or_else(|| {
            format!("gate list line {line_number}: model '{model}' has unsupported class '{class}'")
        })?;
        let companion = match columns[3] {
            "-" => None,
            path => Some(PathBuf::from(path)),
        };
        match (companion_role(family), companion.is_some()) {
            (Some(role), false) => {
                return Err(format!(
                    "gate list line {line_number}: model '{model}' class '{class}' requires a {} companion",
                    role.count_key()
                ));
            }
            (None, true) => {
                return Err(format!(
                    "gate list line {line_number}: model '{model}' class '{class}' does not accept a companion"
                ));
            }
            _ => {}
        }

        let mut counts = BTreeMap::new();
        for item in columns[4].split_ascii_whitespace() {
            let (key, value) = item.split_once('=').ok_or_else(|| {
                format!(
                    "gate list line {line_number}: model '{model}' has malformed tensor count '{item}'"
                )
            })?;
            let role = ManifestRole::from_count_key(key).ok_or_else(|| {
                format!(
                    "gate list line {line_number}: model '{model}' has unknown manifest role '{key}'"
                )
            })?;
            let count = value.parse::<usize>().map_err(|_| {
                format!(
                    "gate list line {line_number}: model '{model}' role '{}' has invalid count '{value}'",
                    role.count_key()
                )
            })?;
            if count == 0 {
                return Err(format!(
                    "gate list line {line_number}: model '{model}' role '{}' has a zero tensor count",
                    role.count_key()
                ));
            }
            if counts.insert(role, count).is_some() {
                return Err(format!(
                    "gate list line {line_number}: model '{model}' repeats manifest role '{}'",
                    role.count_key()
                ));
            }
        }
        let expected = required_roles(family);
        let actual: Vec<ManifestRole> = counts.keys().copied().collect();
        if actual != expected {
            let missing: Vec<&str> = expected
                .iter()
                .filter(|role| !counts.contains_key(role))
                .map(|role| role.count_key())
                .collect();
            let extra: Vec<&str> = actual
                .iter()
                .filter(|role| !expected.contains(role))
                .map(|role| role.count_key())
                .collect();
            return Err(format!(
                "gate list line {line_number}: model '{model}' manifest roles differ; missing {missing:?}, unexpected {extra:?}"
            ));
        }
        rows.push(GateRow {
            model,
            gguf: PathBuf::from(columns[1]),
            family,
            companion,
            counts,
        });
    }
    if rows.is_empty() {
        return Err("gate list contains no model rows".to_string());
    }
    Ok(rows)
}

fn validate_inventory(rows: &[GateRow]) -> Result<(), String> {
    let expected: BTreeSet<&str> = REQUIRED_MODELS.into_iter().collect();
    let actual: BTreeSet<&str> = rows.iter().map(|row| row.model.as_str()).collect();
    let missing: Vec<&str> = expected.difference(&actual).copied().collect();
    let extra: Vec<&str> = actual.difference(&expected).copied().collect();
    if rows.len() != REQUIRED_MODELS.len() || !missing.is_empty() || !extra.is_empty() {
        return Err(format!(
            "G1 inventory must contain exactly {} rows; got {}, missing {missing:?}, unexpected {extra:?}",
            REQUIRED_MODELS.len(),
            rows.len()
        ));
    }
    Ok(())
}

fn validate_manifest(
    model: &str,
    role: ManifestRole,
    manifest: &Manifest,
    expected_count: usize,
    source: &Path,
) {
    assert_eq!(
        manifest.len(),
        expected_count,
        "model '{model}' role '{}': golden '{}' has {} tensors, gate list expects {expected_count}",
        role.count_key(),
        source.display(),
        manifest.len()
    );
    for (tensor, entry) in manifest {
        let object = entry.as_object().unwrap_or_else(|| {
            panic!(
                "model '{model}' role '{}' tensor '{tensor}': golden entry is not an object",
                role.count_key()
            )
        });
        for field in ["dtype", "shape", "sha256"] {
            assert!(
                object.contains_key(field),
                "model '{model}' role '{}' tensor '{tensor}': golden entry lacks '{field}'",
                role.count_key()
            );
        }
        assert_eq!(
            object.len(),
            3,
            "model '{model}' role '{}' tensor '{tensor}': golden entry has unexpected fields",
            role.count_key()
        );
        assert!(
            entry["dtype"]
                .as_str()
                .is_some_and(|dtype| !dtype.is_empty()),
            "model '{model}' role '{}' tensor '{tensor}': invalid golden dtype",
            role.count_key()
        );
        assert!(
            entry["shape"].as_array().is_some_and(|shape| {
                shape.iter().all(Value::is_i64) || shape.iter().all(Value::is_u64)
            }),
            "model '{model}' role '{}' tensor '{tensor}': invalid golden shape",
            role.count_key()
        );
        let sha = entry["sha256"].as_str().unwrap_or_else(|| {
            panic!(
                "model '{model}' role '{}' tensor '{tensor}': invalid golden sha256",
                role.count_key()
            )
        });
        assert!(
            sha.len() == 64 && sha.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "model '{model}' role '{}' tensor '{tensor}': invalid golden sha256 '{sha}'",
            role.count_key()
        );
    }
}

fn read_golden(root: &Path, row: &GateRow, role: ManifestRole) -> Manifest {
    let path = root.join(&row.model).join(role.manifest_filename());
    let bytes = fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "model '{}' role '{}': cannot read golden manifest '{}': {error}",
            row.model,
            role.count_key(),
            path.display()
        )
    });
    let manifest: Manifest = serde_json::from_slice(&bytes).unwrap_or_else(|error| {
        panic!(
            "model '{}' role '{}': cannot parse golden manifest '{}': {error}",
            row.model,
            role.count_key(),
            path.display()
        )
    });
    validate_manifest(&row.model, role, &manifest, row.count(role), &path);
    manifest
}

fn assert_manifest_equal(
    model: &str,
    role: ManifestRole,
    comparison: &str,
    got: &Manifest,
    want: &Manifest,
    expected_count: usize,
) {
    assert_eq!(
        got.len(),
        expected_count,
        "model '{model}' role '{}' {comparison}: got {} tensors, expected {expected_count}",
        role.count_key(),
        got.len()
    );
    assert_eq!(
        want.len(),
        expected_count,
        "model '{model}' role '{}' {comparison}: reference has {} tensors, expected {expected_count}",
        role.count_key(),
        want.len()
    );
    let mut differences = Vec::new();
    for (tensor, wanted) in want {
        match got.get(tensor) {
            None => differences.push(format!("MISSING tensor '{tensor}'")),
            Some(actual) if actual != wanted => differences.push(format!(
                "DIFF tensor '{tensor}': got {}, want {}",
                serde_json::to_string(actual).unwrap(),
                serde_json::to_string(wanted).unwrap()
            )),
            Some(_) => {}
        }
    }
    for tensor in got.keys().filter(|tensor| !want.contains_key(*tensor)) {
        differences.push(format!("EXTRA tensor '{tensor}'"));
    }
    if !differences.is_empty() {
        let total = differences.len();
        differences.truncate(24);
        panic!(
            "model '{model}' role '{}' {comparison}: {total} tensor manifest differences\n{}",
            role.count_key(),
            differences.join("\n")
        );
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RouteExpectation {
    gpu_tensors: usize,
    cpu_tensors: usize,
    bytes: u64,
}

impl RouteExpectation {
    fn add(&mut self, other: Self) {
        self.gpu_tensors += other.gpu_tensors;
        self.cpu_tensors += other.cpu_tensors;
        self.bytes += other.bytes;
    }
}

struct PlannedRoutes {
    routes: Vec<Route>,
    bytes: u64,
}

fn planned_routes(
    label: &str,
    header: &GgufFile,
    plan: &ImportPlan,
    file_len: u64,
) -> PlannedRoutes {
    let mut routes = header
        .tensors
        .iter()
        .map(|tensor| {
            route_for(tensor, plan, true).unwrap_or_else(|error| {
                panic!(
                    "{label}: cannot route tensor '{}': {}",
                    tensor.name, error.reason
                )
            })
        })
        .collect::<Vec<_>>();
    for (tensor, route) in header.tensors.iter().zip(&routes) {
        if GPU_TYPES.contains(&tensor.tensor_type) {
            assert!(
                matches!(route, Route::Gpu(..)),
                "{label}: tensor '{}' type {} has no supported GPU route",
                tensor.name,
                tensor.tensor_type.name()
            );
        }
    }
    let windows = plan_reads(
        header,
        file_len,
        ReadConfig::default().slot_bytes,
        &|index| routes[index].row_multiple(),
    )
    .unwrap_or_else(|error| panic!("{label}: cannot plan reads: {}", error.reason));
    let bytes = windows.iter().map(|window| window.len as u64).sum();
    let mut bands = vec![0usize; header.tensors.len()];
    for band in windows.iter().flat_map(|window| &window.bands) {
        bands[band.tensor] += 1;
        if let Route::Gpu(kind, layout) = routes[band.tensor]
            && !band.src_off.is_multiple_of(band_offset_rule(kind))
        {
            routes[band.tensor] = if kind.is_kquant() {
                Route::CpuKQuant(layout)
            } else {
                Route::CpuAffine
            };
        }
    }
    for (index, count) in bands.into_iter().enumerate() {
        assert!(
            count > 0,
            "{label}: tensor '{}' has no planned data bands",
            header.tensors[index].name
        );
    }
    PlannedRoutes { routes, bytes }
}

fn summarize_routes(header: &GgufFile, planned: PlannedRoutes) -> RouteExpectation {
    let gpu_tensors = planned
        .routes
        .iter()
        .filter(|route| matches!(route, Route::Gpu(..)))
        .count();
    RouteExpectation {
        gpu_tensors,
        cpu_tensors: header.tensors.len() - gpu_tensors,
        bytes: planned.bytes,
    }
}

fn expected_routes(
    label: &str,
    header: &GgufFile,
    plan: &ImportPlan,
    file_len: u64,
) -> RouteExpectation {
    summarize_routes(header, planned_routes(label, header, plan, file_len))
}

fn file_len(model: &str, role: ManifestRole, file: &PreloadFile) -> u64 {
    fs::metadata(&file.0)
        .unwrap_or_else(|error| {
            panic!(
                "model '{model}' role '{}': cannot stat GGUF '{}': {error}",
                role.count_key(),
                file.0.display()
            )
        })
        .len()
}

fn expected_file_routes(model: &str, role: ManifestRole, file: &PreloadFile) -> RouteExpectation {
    let label = format!(
        "model '{model}' role '{}' tensor routes for '{}'",
        role.count_key(),
        file.0.display()
    );
    expected_routes(&label, &file.1, &file.3, file_len(model, role, file))
}

fn expected_input_routes(row: &GateRow, input: &PreloadInput) -> RouteExpectation {
    let mut expected = expected_file_routes(&row.model, ManifestRole::Main, &input.main);
    if input.load_companion {
        let role = row.companion_role().expect("companion role");
        let companion = input.companion.as_ref().unwrap_or_else(|| {
            panic!(
                "model '{}' role '{}': selected companion is missing from PreloadInput",
                row.model,
                role.count_key()
            )
        });
        expected.add(expected_file_routes(&row.model, role, companion));
    }
    expected
}

fn backend_name(backend: RepackBackend) -> &'static str {
    match backend {
        RepackBackend::Auto => "Auto",
        RepackBackend::CpuOnly => "CpuOnly",
    }
}

fn assert_routes(
    model: &str,
    backend: RepackBackend,
    stats: &LoadStats,
    expected: RouteExpectation,
) {
    let actual = (stats.gpu_tensors, stats.cpu_tensors, stats.bytes);
    let wanted = (expected.gpu_tensors, expected.cpu_tensors, expected.bytes);
    assert_eq!(
        actual,
        wanted,
        "model '{model}' backend '{}': (GPU tensors, CPU tensors, aligned read bytes) differ",
        backend_name(backend)
    );
}

fn into_manifests(
    model: &str,
    backend: RepackBackend,
    roles: &[ManifestRole],
    preload: GgufPreload,
) -> (Manifests, LoadStats) {
    let GgufPreload {
        main,
        companion,
        clef_head,
        ple_shards,
        stats,
        ..
    } = preload;
    assert!(
        ple_shards.is_none(),
        "model '{model}' backend '{}': G1 does not expect PLE row shards",
        backend_name(backend)
    );
    let selected_companion = roles
        .iter()
        .copied()
        .find(|role| matches!(role, ManifestRole::Vision | ManifestRole::Draft));
    match selected_companion {
        Some(role) => assert!(
            !companion.is_empty(),
            "model '{model}' backend '{}' role '{}': companion produced no tensors",
            backend_name(backend),
            role.count_key()
        ),
        None => assert!(
            companion.is_empty(),
            "model '{model}' backend '{}': unexpected companion tensors",
            backend_name(backend)
        ),
    }
    let expects_joint = roles.contains(&ManifestRole::JointHead);
    assert_eq!(
        clef_head.is_some(),
        expects_joint,
        "model '{model}' backend '{}' role 'joint_head': CLEF head presence differs",
        backend_name(backend)
    );

    let mut main = Some(manifest_owned(main));
    let mut companion = Some(manifest_owned(companion));
    let mut joint = clef_head.map(|(head, _)| manifest_owned(head));
    clear_cache();
    let mut manifests = Manifests::new();
    for &role in roles {
        let manifest = match role {
            ManifestRole::Main => main.take().expect("main manifest consumed twice"),
            ManifestRole::Vision | ManifestRole::Draft => {
                companion.take().expect("companion manifest consumed twice")
            }
            ManifestRole::JointHead => joint.take().unwrap_or_else(|| {
                panic!(
                    "model '{model}' backend '{}' role 'joint_head': manifest missing",
                    backend_name(backend)
                )
            }),
        };
        manifests.insert(role, manifest);
    }
    (manifests, stats)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlipCase {
    Q4K,
    Q5K,
    Q4KTiled64,
    Q5KTiled64,
}

impl FlipCase {
    const ALL: [Self; 4] = [Self::Q4K, Self::Q5K, Self::Q4KTiled64, Self::Q5KTiled64];

    fn label(self) -> &'static str {
        match self {
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q4KTiled64 => "Q4_K Tiled64",
            Self::Q5KTiled64 => "Q5_K Tiled64",
        }
    }

    fn tensor_type(self) -> GgufTensorType {
        match self {
            Self::Q4K | Self::Q4KTiled64 => GgufTensorType::Q4K,
            Self::Q5K | Self::Q5KTiled64 => GgufTensorType::Q5K,
        }
    }

    /// Unsuffixed cases prefer RowMajor but accept the plan's available layout;
    /// Tiled64-suffixed cases are strict.
    fn layouts(self) -> &'static [bool] {
        const ROW_FIRST: [bool; 2] = [false, true];
        const TILED: [bool; 1] = [true];
        match self {
            Self::Q4K | Self::Q5K => &ROW_FIRST,
            Self::Q4KTiled64 | Self::Q5KTiled64 => &TILED,
        }
    }

    fn qs_offset(self) -> usize {
        match self.tensor_type() {
            GgufTensorType::Q4K => 16,
            GgufTensorType::Q5K => 48,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
struct FlipSelection {
    case: FlipCase,
    gguf_tensor: String,
    output_tensor: String,
    byte_offset: usize,
    bit: u8,
    tiled: bool,
}

fn select_flip_cases(
    model: &str,
    header: &GgufFile,
    plan: &ImportPlan,
    file_len: u64,
    main_manifest: &Manifest,
) -> Result<Vec<FlipSelection>, String> {
    let planned = planned_routes(
        &format!("model '{model}' bit-flip route selection"),
        header,
        plan,
        file_len,
    );
    let mut selections = Vec::new();
    let mut used_sources = BTreeSet::new();
    let mut used_outputs = BTreeSet::new();
    for case in FlipCase::ALL {
        let mut selected = None;
        for &wanted_tiled in case.layouts() {
            for (index, tensor) in header.tensors.iter().enumerate() {
                let tiled = plan.tiled_gguf_names.contains(&tensor.name);
                if tensor.tensor_type != case.tensor_type()
                    || tiled != wanted_tiled
                    || !matches!(planned.routes[index], Route::Gpu(..))
                    || !tensor.name.ends_with(".weight")
                    || used_sources.contains(&tensor.name)
                {
                    continue;
                }
                let Some(output_tensor) =
                    gguf_name_to_hf_for_metadata(&tensor.name, &header.metadata)
                else {
                    continue;
                };
                if !main_manifest.contains_key(&output_tensor)
                    || used_outputs.contains(&output_tensor)
                {
                    continue;
                }
                let byte_offset = case.qs_offset();
                let data_size = tensor.data_size().map_err(|error| {
                    format!(
                        "model '{model}' case '{}': cannot size tensor '{}': {}",
                        case.label(),
                        tensor.name,
                        error.reason
                    )
                })?;
                if data_size <= byte_offset as u64 {
                    return Err(format!(
                        "model '{model}' case '{}': tensor '{}' has only {data_size} bytes, cannot flip qs byte {byte_offset}",
                        case.label(),
                        tensor.name
                    ));
                }
                selected = Some(FlipSelection {
                    case,
                    gguf_tensor: tensor.name.clone(),
                    output_tensor,
                    byte_offset,
                    bit: 0,
                    tiled,
                });
                break;
            }
            if selected.is_some() {
                break;
            }
        }
        let selection = selected.ok_or_else(|| {
            format!(
                "model '{model}': no distinct GPU tensor with a main manifest output satisfies bit-flip case '{}'",
                case.label()
            )
        })?;
        used_sources.insert(selection.gguf_tensor.clone());
        used_outputs.insert(selection.output_tensor.clone());
        selections.push(selection);
    }
    Ok(selections)
}

fn assert_flip_digest(
    model: &str,
    selection: &FlipSelection,
    baseline: &Manifests,
    flipped: &Manifests,
) {
    assert_eq!(
        baseline.keys().collect::<Vec<_>>(),
        flipped.keys().collect::<Vec<_>>(),
        "model '{model}' case '{}': manifest roles changed",
        selection.case.label()
    );
    let mut changed = Vec::new();
    for (role, before) in baseline {
        let after = &flipped[role];
        let names: BTreeSet<&String> = before.keys().chain(after.keys()).collect();
        for name in names {
            if before.get(name) != after.get(name) {
                changed.push((*role, name.clone()));
            }
        }
    }
    let intended = (ManifestRole::Main, selection.output_tensor.clone());
    if changed != [intended.clone()] {
        let details = changed
            .iter()
            .map(|(role, tensor)| {
                format!(
                    "role '{}' tensor '{tensor}': baseline {:?}, flipped {:?}",
                    role.count_key(),
                    baseline[role].get(tensor),
                    flipped[role].get(tensor)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "model '{model}' case '{}' GGUF tensor '{}': intended only role 'model' tensor '{}' to change; changed {changed:?}\n{details}",
            selection.case.label(),
            selection.gguf_tensor,
            selection.output_tensor
        );
    }
    let before = &baseline[&ManifestRole::Main][&selection.output_tensor];
    let after = &flipped[&ManifestRole::Main][&selection.output_tensor];
    assert_ne!(
        &before["sha256"],
        &after["sha256"],
        "model '{model}' case '{}' tensor '{}': source flip did not change the digest",
        selection.case.label(),
        selection.output_tensor
    );
    let mut only_digest = before.clone();
    only_digest["sha256"] = after["sha256"].clone();
    assert_eq!(
        &only_digest,
        after,
        "model '{model}' case '{}' tensor '{}': dtype, shape, or another manifest field changed with the digest",
        selection.case.label(),
        selection.output_tensor
    );
}

pub(crate) fn run() {
    let list = PathBuf::from(std::env::var_os("MLX_GATE_LIST").expect("set MLX_GATE_LIST"));
    let golden_root = PathBuf::from(
        std::env::var_os("MLX_GATE_GOLDEN").expect("set MLX_GATE_GOLDEN=<golden dir>"),
    );
    let text = fs::read_to_string(&list)
        .unwrap_or_else(|error| panic!("cannot read MLX_GATE_LIST '{}': {error}", list.display()));
    let rows = parse_gate_list(&text).unwrap_or_else(|error| panic!("{}: {error}", list.display()));
    validate_inventory(&rows).unwrap_or_else(|error| panic!("{}: {error}", list.display()));

    let mut completed = 0;
    let mut flip_baseline = None;
    for row in rows {
        let roles = row.roles();
        assert!(
            row.gguf.is_file(),
            "model '{}' role 'model': GGUF '{}' is missing",
            row.model,
            row.gguf.display()
        );
        if let Some(companion) = &row.companion {
            let role = row.companion_role().expect("companion role");
            assert!(
                companion.is_file(),
                "model '{}' role '{}': companion GGUF '{}' is missing",
                row.model,
                role.count_key(),
                companion.display()
            );
        }
        let golden: Manifests = roles
            .iter()
            .map(|&role| (role, read_golden(&golden_root, &row, role)))
            .collect();
        let input = preload_input(
            &row.gguf,
            row.companion.as_deref(),
            row.family,
            row.companion.is_some(),
        )
        .unwrap_or_else(|error| {
            panic!(
                "model '{}': cannot plan GGUF '{}': {}",
                row.model,
                row.gguf.display(),
                error.reason
            )
        });
        assert_eq!(
            input.companion.is_some(),
            row.companion.is_some(),
            "model '{}': selected companion was not planned",
            row.model
        );
        let expected = expected_input_routes(&row, &input);
        assert!(
            expected.gpu_tensors > 0 && expected.bytes > 0,
            "model '{}': Auto expectation has {} GPU tensors and {} read bytes",
            row.model,
            expected.gpu_tensors,
            expected.bytes
        );

        let preload = GgufPreload::load_with(&input, RepackBackend::Auto).unwrap_or_else(|error| {
            panic!(
                "model '{}' backend 'Auto': preload failed for '{}': {}",
                row.model,
                row.gguf.display(),
                error.reason
            )
        });
        let (auto, auto_stats) = into_manifests(&row.model, RepackBackend::Auto, &roles, preload);
        assert_routes(&row.model, RepackBackend::Auto, &auto_stats, expected);
        for &role in &roles {
            assert_manifest_equal(
                &row.model,
                role,
                "Auto vs golden",
                &auto[&role],
                &golden[&role],
                row.count(role),
            );
        }

        let preload =
            GgufPreload::load_with(&input, RepackBackend::CpuOnly).unwrap_or_else(|error| {
                panic!(
                    "model '{}' backend 'CpuOnly': preload failed for '{}': {}",
                    row.model,
                    row.gguf.display(),
                    error.reason
                )
            });
        let (cpu, cpu_stats) = into_manifests(&row.model, RepackBackend::CpuOnly, &roles, preload);
        assert_routes(
            &row.model,
            RepackBackend::CpuOnly,
            &cpu_stats,
            RouteExpectation {
                gpu_tensors: 0,
                cpu_tensors: expected.gpu_tensors + expected.cpu_tensors,
                bytes: expected.bytes,
            },
        );
        for &role in &roles {
            assert_manifest_equal(
                &row.model,
                role,
                "CpuOnly vs golden",
                &cpu[&role],
                &golden[&role],
                row.count(role),
            );
            assert_manifest_equal(
                &row.model,
                role,
                "Auto vs CpuOnly",
                &auto[&role],
                &cpu[&role],
                row.count(role),
            );
        }
        let counts = roles
            .iter()
            .map(|role| format!("{}={}", role.count_key(), auto[role].len()))
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "G1_FILE_OK model={} counts={counts} auto_gpu={} auto_cpu={} cpu_gpu={} cpu_cpu={} bytes={}",
            row.model,
            auto_stats.gpu_tensors,
            auto_stats.cpu_tensors,
            cpu_stats.gpu_tensors,
            cpu_stats.cpu_tensors,
            auto_stats.bytes
        );
        if row.model == FLIP_MODEL {
            flip_baseline = Some((row, input, auto, expected));
        }
        completed += 1;
    }
    assert_eq!(completed, REQUIRED_MODELS.len(), "G1 model count");

    let (row, input, baseline, expected_routes) = flip_baseline
        .unwrap_or_else(|| panic!("G1 bit-flip subtest requires model row '{FLIP_MODEL}'"));
    let roles = row.roles();
    let selections = select_flip_cases(
        &row.model,
        &input.main.1,
        &input.main.3,
        file_len(&row.model, ManifestRole::Main, &input.main),
        &baseline[&ManifestRole::Main],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(selections.len(), 4, "G1 bit-flip case count");
    let mut flip_passed = 0;
    for selection in selections {
        println!(
            "G1_FLIP_SELECT model={} case={} gguf_tensor={} output_tensor={} layout={} byte={} bit={}",
            row.model,
            selection.case.label(),
            selection.gguf_tensor,
            selection.output_tensor,
            if selection.tiled {
                "Tiled64"
            } else {
                "RowMajor"
            },
            selection.byte_offset,
            selection.bit
        );
        let mut case_input = PreloadInput::new(
            input.main.clone(),
            input.companion.clone(),
            input.load_companion,
            input.opts.clone(),
        );
        case_input.flip = Some((
            selection.gguf_tensor.clone(),
            selection.byte_offset,
            selection.bit,
        ));
        let preload =
            GgufPreload::load_with(&case_input, RepackBackend::Auto).unwrap_or_else(|error| {
                panic!(
                    "model '{}' case '{}' tensor '{}': flipped preload failed: {}",
                    row.model,
                    selection.case.label(),
                    selection.gguf_tensor,
                    error.reason
                )
            });
        let (flipped, stats) = into_manifests(&row.model, RepackBackend::Auto, &roles, preload);
        assert_routes(&row.model, RepackBackend::Auto, &stats, expected_routes);
        assert_flip_digest(&row.model, &selection, &baseline, &flipped);
        flip_passed += 1;
        println!(
            "G1_FLIP_OK model={} case={} gguf_tensor={} output_tensor={} changed=1 collateral=0",
            row.model,
            selection.case.label(),
            selection.gguf_tensor,
            selection.output_tensor
        );
    }
    assert_eq!(flip_passed, 4, "G1 successful bit-flip cases");
    println!(
        "G1_TEST_COUNT files={completed} bit_flip=1 total={}",
        completed + 1
    );
    println!("G1_OK files={completed} flip_cases={flip_passed}");
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::utils::gguf::{GgufMetaValue, GgufTensorInfo};
    use crate::utils::gguf_plan::OutputRole;

    fn tensor(
        name: &str,
        dims: &[u64],
        tensor_type: GgufTensorType,
        offset: u64,
    ) -> GgufTensorInfo {
        GgufTensorInfo {
            name: name.to_string(),
            n_dims: dims.len() as u32,
            dims: dims.to_vec(),
            tensor_type,
            offset,
        }
    }

    fn synthetic_header(tensors: Vec<GgufTensorInfo>) -> GgufFile {
        GgufFile {
            version: 3,
            tensor_count: tensors.len() as u64,
            metadata: HashMap::from([(
                "general.architecture".to_string(),
                GgufMetaValue::String("qwen35".to_string()),
            )]),
            tensors,
            alignment: 32,
            data_offset: 16 * 1024,
        }
    }

    fn synthetic_plan(tiled: &[&str]) -> ImportPlan {
        ImportPlan {
            role: OutputRole::Primary,
            import_k_quants: true,
            native_qwen35_layout: true,
            prism: false,
            tie_word_embeddings: false,
            preserve_dtype_keys: Default::default(),
            tiled_gguf_names: tiled.iter().map(|name| name.to_string()).collect(),
            tiled_prefixes: Default::default(),
            config: None,
            clef_head_config: None,
            asset_copies: Vec::new(),
            strict_asset_copies: false,
            muse_secondary: false,
            ple_shard: None,
        }
    }

    fn file_len(header: &GgufFile) -> u64 {
        header
            .tensors
            .iter()
            .map(|tensor| header.data_offset + tensor.offset + tensor.data_size().unwrap())
            .max()
            .unwrap_or(header.data_offset)
    }

    fn manifest_for_header(header: &GgufFile) -> Manifest {
        header
            .tensors
            .iter()
            .filter_map(|tensor| {
                gguf_name_to_hf_for_metadata(&tensor.name, &header.metadata).map(|name| {
                    (
                        name,
                        serde_json::json!({
                            "dtype": "U32",
                            "shape": [1],
                            "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
                        }),
                    )
                })
            })
            .collect()
    }

    const LIST: &str = "# golden-dir<TAB>gguf<TAB>class<TAB>companion<TAB>counts\n\
qwen38-27b\t/models/qwen.gguf\tQwen35Model\t-\tmodel=1768\n\
qwen36-35b-q4km\t/models/q4km.gguf\tQwen35MoeModel\t-\tmodel=1260\n\
qwen36-35b-iq4xs\t/models/iq4xs.gguf\tQwen35MoeModel\t-\tmodel=1260\n\
qwen36-35b-iq3s\t/models/iq3s.gguf\tQwen35MoeModel\t-\tmodel=1511\n\
qwen36-35b-mxfp4\t/models/mxfp4.gguf\tQwen35MoeModel\t-\tmodel=1179\n\
gemma4-12b\t/models/gemma.gguf\tGemma4Model\t/models/mmproj.gguf\tmodel=996 vision=11\n\
muse-30b\t/models/muse.gguf\tMuseGlimmerModel\t/models/draft.gguf\tmodel=1463 draft=130\n\
clef-flash\t/models/clef.gguf\tClefModel\t-\tmodel=775 joint_head=122\n";

    #[test]
    fn list_parser_covers_models_companions_and_roles() {
        let rows = parse_gate_list(LIST).unwrap();
        assert_eq!(rows.len(), 8);
        validate_inventory(&rows).unwrap();
        assert_eq!(rows[0].roles(), vec![ManifestRole::Main]);
        assert_eq!(
            rows[5].roles(),
            vec![ManifestRole::Main, ManifestRole::Vision]
        );
        assert_eq!(
            rows[5].companion.as_deref(),
            Some(Path::new("/models/mmproj.gguf"))
        );
        assert_eq!(
            rows[6].roles(),
            vec![ManifestRole::Main, ManifestRole::Draft]
        );
        assert_eq!(
            rows[7].roles(),
            vec![ManifestRole::Main, ManifestRole::JointHead]
        );
        assert_eq!(rows[5].count(ManifestRole::Vision), 11);
        assert_eq!(rows[6].count(ManifestRole::Draft), 130);
        assert_eq!(rows[7].count(ManifestRole::JointHead), 122);
        assert_eq!(
            ManifestRole::Vision.manifest_filename(),
            "manifest-vision.json"
        );
        assert_eq!(
            ManifestRole::Draft.manifest_filename(),
            "manifest-draft.json"
        );
        assert_eq!(
            ManifestRole::JointHead.manifest_filename(),
            "manifest-joint_head.json"
        );
    }

    #[test]
    fn list_parser_rejects_malformed_and_missing_rows() {
        let malformed = [
            "only\tthree\tcolumns",
            "m\t/m.gguf\tUnknownModel\t-\tmodel=1",
            "m\t/m.gguf\tQwen35Model\t-\t",
            "m\t/m.gguf\tQwen35Model\t-\tvision=1",
            "m\t/m.gguf\tQwen35Model\t-\tmodel=one",
            "m\t/m.gguf\tQwen35Model\t-\tmodel=0",
            "m\t/m.gguf\tQwen35Model\t-\tmodel=1 model=2",
            "m\t/m.gguf\tGemma4Model\t-\tmodel=1 vision=1",
            "m\t/m.gguf\tQwen35Model\t/c.gguf\tmodel=1",
            "m\t/m.gguf\tClefModel\t-\tmodel=1",
            "m\t/m.gguf\tQwen35Model\t-\tmodel=1 tensor=2",
        ];
        for text in malformed {
            let error = parse_gate_list(text).unwrap_err();
            assert!(error.contains("line 1"), "{text}: {error}");
        }
        let duplicate = "m\t/a.gguf\tQwen35Model\t-\tmodel=1\nm\t/b.gguf\tQwen35Model\t-\tmodel=1";
        assert!(
            parse_gate_list(duplicate)
                .unwrap_err()
                .contains("duplicate model 'm'")
        );
        let missing_companion = "gemma\t/m.gguf\tGemma4Model\t-\tmodel=1 vision=1";
        let error = parse_gate_list(missing_companion).unwrap_err();
        assert!(
            error.contains("gemma") && error.contains("companion"),
            "{error}"
        );
        let missing_joint = "clef\t/m.gguf\tClefModel\t-\tmodel=1";
        let error = parse_gate_list(missing_joint).unwrap_err();
        assert!(
            error.contains("clef") && error.contains("joint_head"),
            "{error}"
        );

        let without_iq3s = LIST
            .lines()
            .filter(|line| !line.starts_with("qwen36-35b-iq3s\t"))
            .collect::<Vec<_>>()
            .join("\n");
        let rows = parse_gate_list(&without_iq3s).unwrap();
        let error = validate_inventory(&rows).unwrap_err();
        assert!(error.contains("qwen36-35b-iq3s"), "{error}");
    }

    #[test]
    fn route_expectation_counts_gpu_types_and_cpu_fallback() {
        for tensor_type in GPU_TYPES {
            assert!(
                gpu_kind_for(tensor_type).is_some(),
                "{tensor_type:?} lost its GPU kind"
            );
        }
        for tensor_type in [
            GgufTensorType::Q4_1,
            GgufTensorType::Q5_0,
            GgufTensorType::Q5_1,
        ] {
            assert!(
                gpu_kind_for(tensor_type).is_none(),
                "{tensor_type:?} must retain its CPU route"
            );
        }
        let header = synthetic_header(vec![
            tensor("blk.0.ffn_gate.weight", &[256, 64], GgufTensorType::Q4K, 0),
            tensor(
                "blk.1.ffn_down.weight",
                &[256, 64],
                GgufTensorType::Q5K,
                9_216,
            ),
            tensor(
                "blk.2.ffn_up.weight",
                &[256, 64],
                GgufTensorType::Q4K,
                20_482,
            ),
            tensor(
                "blk.3.attn_q.weight",
                &[32, 2],
                GgufTensorType::Q8_0,
                29_698,
            ),
            tensor(
                "blk.4.attn_k.weight",
                &[32, 2],
                GgufTensorType::Q4_0,
                29_766,
            ),
            tensor("blk.4.attn_norm.weight", &[4], GgufTensorType::F32, 29_802),
            tensor(
                "blk.5.attn_v.weight",
                &[256, 1],
                GgufTensorType::Q2K,
                29_818,
            ),
        ]);
        let plan = synthetic_plan(&["blk.1.ffn_down.weight"]);
        let expected = expected_routes("synthetic routes", &header, &plan, file_len(&header));
        assert_eq!(expected.gpu_tensors, 4);
        assert_eq!(expected.cpu_tensors, 3);
        assert!(expected.bytes > 0);
        let planned = planned_routes("synthetic routes", &header, &plan, file_len(&header));
        assert!(matches!(planned.routes[0], Route::Gpu(..)));
        assert!(matches!(planned.routes[1], Route::Gpu(..)));
        assert!(
            matches!(planned.routes[2], Route::CpuKQuant(..)),
            "the two-byte-misaligned Q4_K tensor must fall back to CPU"
        );
        assert!(matches!(planned.routes[3], Route::Gpu(..)));
        assert!(matches!(planned.routes[4], Route::Gpu(..)));
    }

    #[test]
    fn flip_selection_uses_plan_layout_and_weight_fields() {
        let header = synthetic_header(vec![
            tensor("blk.0.ffn_gate.weight", &[256, 64], GgufTensorType::Q4K, 0),
            tensor(
                "blk.1.ffn_down.weight",
                &[256, 64],
                GgufTensorType::Q5K,
                9_216,
            ),
            tensor(
                "blk.2.attn_q.weight",
                &[256, 64],
                GgufTensorType::Q4K,
                20_480,
            ),
            tensor(
                "blk.3.attn_v.weight",
                &[256, 64],
                GgufTensorType::Q5K,
                29_696,
            ),
            tensor("output_norm.weight", &[4], GgufTensorType::F32, 40_960),
        ]);
        let plan = synthetic_plan(&["blk.0.ffn_gate.weight", "blk.1.ffn_down.weight"]);
        let manifest = manifest_for_header(&header);
        let selected =
            select_flip_cases("synthetic", &header, &plan, file_len(&header), &manifest).unwrap();
        assert_eq!(selected.len(), 4);
        assert_eq!(selected[0].gguf_tensor, "blk.2.attn_q.weight");
        assert_eq!(
            selected[0].output_tensor,
            "model.layers.2.self_attn.q_proj.weight"
        );
        assert_eq!((selected[0].byte_offset, selected[0].tiled), (16, false));
        assert_eq!(selected[1].gguf_tensor, "blk.3.attn_v.weight");
        assert_eq!((selected[1].byte_offset, selected[1].tiled), (48, false));
        assert_eq!(selected[2].gguf_tensor, "blk.0.ffn_gate.weight");
        assert_eq!((selected[2].byte_offset, selected[2].tiled), (16, true));
        assert_eq!(selected[3].gguf_tensor, "blk.1.ffn_down.weight");
        assert_eq!((selected[3].byte_offset, selected[3].tiled), (48, true));

        let all_q5_tiled = synthetic_header(vec![
            tensor("blk.0.ffn_gate.weight", &[256, 64], GgufTensorType::Q4K, 0),
            tensor(
                "blk.1.ffn_down.weight",
                &[256, 64],
                GgufTensorType::Q5K,
                9_216,
            ),
            tensor(
                "blk.2.attn_q.weight",
                &[256, 64],
                GgufTensorType::Q4K,
                20_480,
            ),
            tensor(
                "blk.3.attn_v.weight",
                &[256, 64],
                GgufTensorType::Q5K,
                29_696,
            ),
        ]);
        let plan = synthetic_plan(&[
            "blk.0.ffn_gate.weight",
            "blk.1.ffn_down.weight",
            "blk.3.attn_v.weight",
        ]);
        let manifest = manifest_for_header(&all_q5_tiled);
        let selected = select_flip_cases(
            "synthetic",
            &all_q5_tiled,
            &plan,
            file_len(&all_q5_tiled),
            &manifest,
        )
        .unwrap();
        assert!(
            selected[1].tiled,
            "unsuffixed Q5_K uses the available plan layout"
        );
        assert!(selected[3].tiled);
        assert_ne!(selected[1].gguf_tensor, selected[3].gguf_tensor);
    }

    #[test]
    fn flip_selection_reports_missing_cases() {
        let header = synthetic_header(vec![tensor(
            "blk.0.ffn_gate.weight",
            &[256, 64],
            GgufTensorType::Q4K,
            0,
        )]);
        let plan = synthetic_plan(&[]);
        let manifest = manifest_for_header(&header);
        let error = select_flip_cases(
            "synthetic-missing",
            &header,
            &plan,
            file_len(&header),
            &manifest,
        )
        .unwrap_err();
        assert!(
            error.contains("synthetic-missing") && error.contains("Q5_K"),
            "{error}"
        );
    }
}
