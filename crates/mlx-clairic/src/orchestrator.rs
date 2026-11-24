//! End-to-end Clairic dual-path request orchestrator.
//!
//! Direct Rust port of upstream `clairic_runtime/{api.py, backend.py, text_ar.py, large.py}`,
//! decoupling pure-Rust request planning, prompt encoding, grammar construction, recovery,
//! and post-assembly from the MLX execution backend via [`ClairicBackend`].

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, atomic::AtomicBool};
use std::time::Instant;

use image::RgbImage;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;
use uuid::Uuid;

use crate::grammar::GrammarSpec;

use super::ar;
use super::completion::{
    self, ArBatchItemResult, ArTaskDescriptor, CohortHooks, OrdinaryListFrontiers,
};
use super::contracts;
use super::crops;
use super::embedded::{
    RenderedPdfPageData, ValidatedImage, decoded_base64_size, validate_image_count,
    validate_request_media,
};
use super::errors::{ClairicError, Result, check_cancel, inference_error, invalid_request};
use super::grounding;
use super::identifier_spelling;
use super::image_rows;
use super::json_validate;
use super::lists;
use super::media::{hex_digest, rgb_image_digest};
use super::media_cache::{MediaKeyItem, media_cache_key};
use super::media_state_prompt;
use super::printed_numbers;
use super::printed_temporal;
use super::profile::ClairicProfile;
use super::receipt;
use super::row_omission;
use super::schema;
use super::scopes;
use super::settings::{self, MAX_IMAGES, MAX_LENGTH, MAX_OUTPUT_TOKENS, VISION_MAX_PIXELS};
use super::source_dates;
use super::source_rows;
use super::table_rows;
use super::typed_decode;
use super::typed_prompt::{self, EncodeRecordOptions, EncodedMedia, EncodedRecord};
use super::types::{ArGenOutput, ArGenTask, FinishReason};

/// Maximum raw JSON request body size accepted by Clairic (16 MiB).
pub const MAX_REQUEST_BYTES: usize = 16 * 1024 * 1024;

/// Backend interface bridging the pure-Rust Clairic orchestrator to an execution
/// engine (such as `ClairicEngine` backed by `Qwen35Inner` in `mlx-core`).
pub trait ClairicBackend {
    /// Return the shared HuggingFace tokenizer used for prompt encoding and decoding.
    fn tokenizer(&self) -> Result<Arc<Tokenizer>>;

    /// Resolve the primary EOS / stop token ID for AR generation.
    fn eos_token_id(&self) -> Result<u32>;

    /// Return the image placeholder token ID used for vision-token usage accounting.
    fn image_token_id(&self) -> u32;

    /// Return the `<|vision_end|>` token ID delimiting vision token spans.
    fn vision_end_token_id(&self) -> u32;

    /// Return the active model behavioral profile (`Flash` vs `Large27B`).
    fn profile(&self) -> ClairicProfile;

    /// Build the vision placeholder token sequence and total merged image token count
    /// for a slice of prepared RGB images.
    fn build_vision_tokens(&self, images: &[Arc<RgbImage>]) -> Result<(Vec<u32>, usize)>;

    /// Path B: run typed-head readout on an encoded record with optional media.
    fn read_typed(
        &mut self,
        item: &EncodedRecord,
        media: Option<&[Arc<RgbImage>]>,
        cancelled: Option<&Arc<AtomicBool>>,
    ) -> Result<Map<String, Value>>;

    /// Path A: execute a batch of grammar-constrained autoregressive generation tasks.
    fn generate_batch(
        &mut self,
        tasks: &[ArGenTask],
        score_token_ids: &[u32],
        cancelled: Option<&Arc<AtomicBool>>,
    ) -> Result<Vec<ArGenOutput>>;

    /// Release any per-request vision/prefix caches after a request completes or fails.
    fn clear_request_caches(&mut self);

    /// Whether opt-in `[clairic-phase]` timing instrumentation is enabled.
    fn phase_timing_enabled(&self) -> bool {
        false
    }

    /// Record request-level start timestamp and `parse_request` latency so
    /// [`Self::finish_request_timing`] can emit a complete `[clairic-phase]`
    /// summary including `clear_cache_ms` and total wall time.
    fn set_request_timing_start(&mut self, _start: Instant, _parse_ms: f64) {}

    /// Record `encode_request` latency for `[clairic-phase]` reporting.
    fn set_request_encode_ms(&mut self, _encode_ms: f64) {}

    /// Emit any recorded `[clairic-phase]` timing breakdown once request cache
    /// cleanup finishes.
    fn finish_request_timing(&mut self) {}
}

const ALLOWED_TOP_LEVEL_KEYS: &[&str] = &[
    "model",
    "state",
    "images",
    "documents",
    "fields",
    "reference_date",
    "options",
];

/// Maximum number of expanded query slots allowed for a single list field.
const MAX_LIST_EXPANSION_BUDGET: usize = contracts::MAX_LIST_EXPANSION_BUDGET;
/// Maximum total number of AR tasks (scalar + expanded list items) allowed across a single request.
const MAX_TOTAL_EXPANDED_TASKS: usize = 2048;
/// Maximum total prompt tokens materialized across all AR tasks in a single request
/// (~16 MiB of u32 token IDs), preventing multi-GB OOM from large scalar field fans.
const MAX_TOTAL_AR_PROMPT_TOKENS: usize = 4_194_304;

const USER_HEADER: &str = "<|im_start|>user\n";
const ASSISTANT_FOOTER: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n";

fn check_cancel_opt(cancelled: Option<&Arc<AtomicBool>>) -> Result<()> {
    if let Some(flag) = cancelled {
        check_cancel(flag)?;
    }
    Ok(())
}

fn is_truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

// =============================================================================
// Request parsing & validation (`api.py`)
// =============================================================================

#[derive(Debug, Clone, Copy, Default)]
struct PredictOptions {
    include_quantiles: bool,
    return_state_text: bool,
}

#[derive(Clone)]
struct InternalRequest {
    state: String,
    image_count: usize,
    pdf_page_count: usize,
    decoded_images: Vec<Arc<RgbImage>>,
    source_media_key: Option<String>,
    fields: Map<String, Value>,
    reference_date: Option<String>,
}

impl InternalRequest {
    fn has_media(&self) -> bool {
        self.image_count > 0 || self.pdf_page_count > 0
    }

    fn has_documents(&self) -> bool {
        self.pdf_page_count > 0
    }

    fn with_fields(&self, fields: Map<String, Value>) -> Self {
        let mut copy = self.clone();
        copy.fields = fields;
        copy
    }
}

struct ParsedRequest {
    internal: InternalRequest,
    options: PredictOptions,
}

fn extract_document_base64_size(doc: &Value) -> Result<usize> {
    match doc {
        Value::String(s) => {
            if let Some((_, b64)) = s.split_once(";base64,") {
                Ok(decoded_base64_size(b64))
            } else {
                Err(invalid_request(
                    "document must be a base64 data URI or document object",
                ))
            }
        }
        Value::Object(map) => {
            let Some(b64) = map.get("base64").and_then(Value::as_str) else {
                return Err(invalid_request("document object missing base64 string"));
            };
            Ok(decoded_base64_size(b64))
        }
        _ => Err(invalid_request("invalid document entry")),
    }
}

fn parse_request(
    raw_request: &str,
    rendered_pdf_pages: Vec<RenderedPdfPageData>,
) -> Result<ParsedRequest> {
    if raw_request.len() > MAX_REQUEST_BYTES {
        return Err(invalid_request("request exceeds 16 MiB"));
    }

    let root: Value = serde_json::from_str(raw_request)
        .map_err(|e| invalid_request(format!("invalid JSON body: {e}")))?;
    let Some(obj) = root.as_object() else {
        return Err(invalid_request("request body must be a JSON object"));
    };

    for key in obj.keys() {
        if !ALLOWED_TOP_LEVEL_KEYS.contains(&key.as_str()) {
            return Err(invalid_request(format!("unknown request key {key:?}")));
        }
    }

    // Optional `model` validation (matches CLEF: accept any non-empty, non-whitespace
    // served checkpoint ID string so renamed local checkpoints advertised by `/v1/models` work).
    if let Some(model_val) = obj.get("model")
        && !model_val.is_null()
    {
        let Some(alias) = model_val.as_str() else {
            return Err(invalid_request("model must be a non-empty string"));
        };
        if alias.trim().is_empty() {
            return Err(invalid_request("model must be a non-empty string"));
        }
    }

    // `state`: missing/null -> "", string -> verbatim, non-string JSON -> compact JSON
    let state = match obj.get("state") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => serde_json::to_string(other)
            .map_err(|e| invalid_request(format!("failed to serialize state: {e}")))?,
    };

    // `images`: optional array of data URIs or {content_type, base64, rotation?}
    let mut validated_images: Vec<ValidatedImage> = Vec::new();
    if let Some(images_val) = obj.get("images")
        && !images_val.is_null()
    {
        let Some(arr) = images_val.as_array() else {
            return Err(invalid_request("images must be an array"));
        };
        validate_image_count(arr.len(), MAX_IMAGES)?;
        validated_images.reserve(arr.len());
        for elem in arr {
            validated_images.push(ValidatedImage::from_json(elem)?);
        }
    }

    // `documents`: optional array; validate count and decoded byte size
    let mut doc_encoded_bytes: Vec<usize> = Vec::new();
    if let Some(docs_val) = obj.get("documents")
        && !docs_val.is_null()
    {
        let Some(arr) = docs_val.as_array() else {
            return Err(invalid_request("documents must be an array"));
        };
        validate_image_count(arr.len(), MAX_IMAGES)?;
        for doc in arr {
            doc_encoded_bytes.push(extract_document_base64_size(doc)?);
        }
        if !arr.is_empty() && rendered_pdf_pages.is_empty() {
            return Err(invalid_request(
                "PDF documents require pre-rendered rendered_pdf_pages",
            ));
        }
    }

    validate_request_media(&validated_images, &rendered_pdf_pages, &doc_encoded_bytes)?;

    // `fields`: required non-empty JSON object
    let Some(fields_val) = obj.get("fields") else {
        return Err(invalid_request("fields must declare at least one field"));
    };
    let Some(fields) = fields_val.as_object() else {
        return Err(invalid_request("fields must be a JSON object"));
    };
    if fields.is_empty() {
        return Err(invalid_request("fields must declare at least one field"));
    }

    // `reference_date`: optional string
    let reference_date = match obj.get("reference_date") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => return Err(invalid_request("reference_date must be a string")),
    };

    // `options`: optional { include_quantiles?, return_state_text? }
    let mut options = PredictOptions::default();
    if let Some(opts_val) = obj.get("options")
        && !opts_val.is_null()
    {
        let Some(opts_obj) = opts_val.as_object() else {
            return Err(invalid_request("options must be a JSON object"));
        };
        for (k, v) in opts_obj {
            match k.as_str() {
                "include_quantiles" => {
                    let Some(b) = v.as_bool() else {
                        return Err(invalid_request(
                            "options.include_quantiles must be a boolean",
                        ));
                    };
                    options.include_quantiles = b;
                }
                "return_state_text" => {
                    let Some(b) = v.as_bool() else {
                        return Err(invalid_request(
                            "options.return_state_text must be a boolean",
                        ));
                    };
                    options.return_state_text = b;
                }
                _ => {
                    return Err(invalid_request(format!("unknown option {k:?}")));
                }
            }
        }
    }

    if state.trim().is_empty() && validated_images.is_empty() && rendered_pdf_pages.is_empty() {
        return Err(invalid_request(
            "state must be non-empty when no images are sent",
        ));
    }

    // Validate public fields + reference_date up front
    let ref_val = reference_date
        .as_deref()
        .map(|s| Value::String(s.to_string()));
    contracts::validate_public_request(fields_val, ref_val.as_ref())?;

    // Decode & orient images and pre-rendered PDF pages
    let mut decoded_images: Vec<Arc<RgbImage>> =
        Vec::with_capacity(validated_images.len() + rendered_pdf_pages.len());
    for img in &validated_images {
        decoded_images.push(Arc::new(img.to_rgb_image()?));
    }
    for page in &rendered_pdf_pages {
        decoded_images.push(Arc::new(page.to_rgb_image()?));
    }

    let source_media_key = if !decoded_images.is_empty() {
        let mut hasher = Sha256::new();
        if !validated_images.is_empty() {
            let items: Vec<MediaKeyItem<'_>> =
                validated_images.iter().map(MediaKeyItem::from).collect();
            hasher.update(media_cache_key(&items).as_bytes());
        }
        for page_img in decoded_images.iter().skip(validated_images.len()) {
            hasher.update(rgb_image_digest(page_img).as_bytes());
        }
        Some(hex_digest(&hasher.finalize()))
    } else {
        None
    };

    Ok(ParsedRequest {
        internal: InternalRequest {
            state,
            image_count: validated_images.len(),
            pdf_page_count: rendered_pdf_pages.len(),
            decoded_images,
            source_media_key,
            fields: fields.clone(),
            reference_date,
        },
        options,
    })
}

// =============================================================================
// Internal request expansion & encoding (`backend.py` + `large.py`)
// =============================================================================

#[derive(Clone)]
enum EvidenceSource {
    Text(Arc<grounding::Source>),
    Date(Arc<source_dates::DateSource>),
}

impl EvidenceSource {
    fn span(&self, value: &Value, spec: &Value) -> (bool, Option<[usize; 2]>) {
        match self {
            Self::Text(src) => {
                if src.normalize_whitespace {
                    src.span(value, Some(spec))
                } else {
                    src.span(value, None)
                }
            }
            Self::Date(src) => src.span(value),
        }
    }

    fn is_whitespace_normalized(&self) -> bool {
        match self {
            Self::Text(src) => src.normalize_whitespace,
            Self::Date(_) => false,
        }
    }

    fn text(&self) -> &str {
        match self {
            Self::Text(src) => &src.text,
            Self::Date(src) => &src.text,
        }
    }
}

#[derive(Clone)]
struct EncodedArTask {
    name: String,
    input_ids: Vec<u32>,
    media: Option<Vec<Arc<RgbImage>>>,
    media_key: Option<String>,
    constraint: GrammarSpec,
    constraint_json: Value,
}

#[derive(Default)]
struct EncodeOptions<'a> {
    identifier_view: bool,
    party_name_view: bool,
    image_views: Option<Vec<Arc<RgbImage>>>,
    physical_row_view: bool,
    image_view_key: Option<String>,
    pdf_spelling_view: bool,
    identifier_spelling_disabled: bool,
    po_detail_view: bool,
    po_supplement_disabled: bool,
    share_image_row_description: bool,
    unfocused_default_image: Option<&'a Arc<RgbImage>>,
    cancelled: Option<&'a Arc<AtomicBool>>,
}

impl EncodeOptions<'_> {
    fn is_flash_detail(&self) -> bool {
        self.identifier_view
            || self.party_name_view
            || self.physical_row_view
            || self.identifier_spelling_disabled
    }

    fn is_large_detail(&self) -> bool {
        self.is_flash_detail() || self.pdf_spelling_view || self.po_detail_view
    }
}

struct EncodedDateSupplement {
    encoded: Box<EncodedRequest>,
    source: Arc<source_dates::DateSource>,
    names: Vec<String>,
}

struct EncodedRequest {
    record: Value,
    forecasts: Vec<(String, Vec<String>)>,
    tasks: Vec<EncodedArTask>,
    typed: Option<EncodedRecord>,
    typed_media: Option<Vec<Arc<RgbImage>>>,
    eos: u32,
    ar_source: Option<EvidenceSource>,
    ar_sources: HashMap<String, EvidenceSource>,
    ar_provenance: Option<Arc<grounding::Source>>,
    ar_provenance_map: HashMap<String, Arc<grounding::Source>>,
    original_images: Vec<Arc<RgbImage>>,
    default_unfocused_single: bool,
    list_source_bounds: BTreeMap<String, usize>,
    identifier_spelling_fallback: Option<Box<EncodedRequest>>,
    date_supplement: Option<EncodedDateSupplement>,
    receipt_sum_fields: Vec<String>,
    cache_salt: String,
}

fn constraint_to_json(spec: &GrammarSpec) -> Result<Value> {
    match spec {
        GrammarSpec::JsonSchema(val) => {
            let s = serde_json::to_string(val)
                .map_err(|e| invalid_request(format!("failed to serialize json_schema: {e}")))?;
            Ok(json!({ "json_schema": s }))
        }
        GrammarSpec::ClairicEbnf(ebnf) => Ok(json!({ "ebnf": ebnf })),
        GrammarSpec::GroundedText { .. } => Ok(json!({ "grounded_text": true })),
        GrammarSpec::Lark(lark) => Ok(json!({ "lark": lark })),
    }
}

fn expand_request(
    req: &InternalRequest,
    profile: ClairicProfile,
    cancelled: Option<&Arc<AtomicBool>>,
) -> Result<(Value, Vec<(String, Vec<String>)>)> {
    let has_media = req.has_media();
    contracts::validate_public_fields(&req.fields)?;

    let financial_fields = scopes::financial::prepare(
        &req.fields,
        has_media,
        profile.financial_fee_guidance(),
        profile.financial_service_guidance(),
    );
    let (mut fields, forecasts) = schema::expand_forecasts_grouped(&financial_fields)?;
    schema::validate_fields(&Value::Object(fields.clone()))?;
    let ref_date_val = req
        .reference_date
        .as_deref()
        .map(|s| Value::String(s.to_string()));
    schema::validate_reference_date(ref_date_val.as_ref())?;

    if profile.date_role_guidance() {
        for spec in fields.values_mut() {
            if let Some(obj) = spec.as_object_mut()
                && obj.get("type").and_then(Value::as_str) == Some("date")
            {
                let existing = obj
                    .get("instructions")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let updated = if existing.is_empty() {
                    "Follow field instructions. Match the requested date role; null if unavailable."
                        .to_string()
                } else {
                    format!(
                        "{existing}\nFollow field instructions. Match the requested date role; null if unavailable."
                    )
                };
                obj.insert("instructions".to_string(), Value::String(updated));
            }
        }
    }

    let mut annotations: Map<String, Value> = Map::new();
    let mut row_inputs: Map<String, Value> = Map::new();
    let mut row_names: Map<String, Value> = Map::new();

    let source_tables = table_rows::prepare(
        &req.state,
        &fields,
        has_media,
        profile.source_row_eligibility() || profile.local_source_scope(),
    );

    let mut total_expanded_tasks: usize = 0;
    let field_names: Vec<String> = fields.keys().cloned().collect();
    for name in field_names {
        check_cancel_opt(cancelled)?;
        let Some(spec) = fields.get(&name).cloned() else {
            continue;
        };
        if spec.get("type").and_then(Value::as_str) != Some("list") {
            continue;
        }
        let source_candidates = source_tables.get(&name);
        let row_eligibility = profile.source_row_eligibility()
            || source_candidates.is_some_and(|rows| rows.iter().any(|row| row.eligibility_only));

        let budget = if let Some(rows) = source_candidates
            && !rows.is_empty()
        {
            rows.len()
        } else if has_media && spec.get("max_items").is_none() {
            schema::MAX_LIST_ITEMS + 1
        } else {
            lists::list_query_budget(&req.state, &spec)
                .map_err(ClairicError::from)?
                .0
        };

        if budget > MAX_LIST_EXPANSION_BUDGET {
            return Err(invalid_request(format!(
                "list {name}: expansion budget ({budget}) exceeds maximum allowed ({MAX_LIST_EXPANSION_BUDGET})"
            )));
        }
        total_expanded_tasks = total_expanded_tasks
            .checked_add(budget)
            .filter(|&total| total <= MAX_TOTAL_EXPANDED_TASKS)
            .ok_or_else(|| {
                invalid_request(format!(
                    "total list expansion tasks exceed maximum allowed ({MAX_TOTAL_EXPANDED_TASKS})"
                ))
            })?;

        let mut queries: Vec<String> = Vec::with_capacity(budget);
        let item_spec = contracts::item_spec(&spec);
        let item_type = item_spec
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let spec_instructions = spec
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("");

        for index in 1..=budget {
            check_cancel_opt(cancelled)?;
            let child = schema::list_item_name(&name, index);
            if fields.contains_key(&child) {
                return Err(invalid_request(format!(
                    "Generated list field conflicts with {child}"
                )));
            }

            let mut child_obj = Map::new();
            if let Some(spec_obj) = spec.as_object() {
                for (k, v) in spec_obj {
                    if !matches!(k.as_str(), "type" | "items" | "instructions" | "max_items") {
                        child_obj.insert(k.clone(), v.clone());
                    }
                }
            }
            if let Some(item_obj) = item_spec.as_object() {
                for (k, v) in item_obj {
                    child_obj.insert(k.clone(), v.clone());
                }
            }
            child_obj.insert("type".to_string(), Value::String(item_type.clone()));
            let child_inst = format!(
                "Item {index} of {name}, following requested order and duplicate policy. {spec_instructions} Return null if this item does not exist."
            )
            .trim()
            .to_string();
            child_obj.insert("instructions".to_string(), Value::String(child_inst));

            let mut child_val = Value::Object(child_obj);
            if let Some(rows) = source_candidates
                && let Some(row) = rows.get(index - 1)
            {
                child_val = table_rows::row_spec(&spec, row, row_eligibility)?;
            }
            if matches!(item_type.as_str(), "text" | "number" | "object")
                && !(source_candidates.is_some() && row_eligibility)
                && let Some(c_obj) = child_val.as_object_mut()
            {
                c_obj.insert("nullable".to_string(), Value::Bool(true));
                if let Some(Value::Array(en)) = c_obj.get_mut("enum")
                    && !en.iter().any(Value::is_null)
                {
                    en.push(Value::Null);
                }
            }

            fields.insert(child.clone(), child_val);
            annotations.insert(child.clone(), json!({ "parent": name, "index": index }));
            queries.push(child);
        }

        if let Some(rows) = source_candidates
            && !rows.is_empty()
        {
            let rows_json = serde_json::to_value(rows)
                .map_err(|e| invalid_request(format!("failed to serialize source rows: {e}")))?;
            annotations.insert(
                name.clone(),
                json!({
                    "source_fields": queries,
                    "source_rows": rows_json,
                    "source_row_eligibility": row_eligibility,
                }),
            );
            if profile.local_source_scope() {
                let (inputs, names) =
                    table_rows::local_scope_context(rows, &queries, row_names.len());
                for (k, v) in inputs {
                    row_inputs.insert(k, Value::String(v));
                }
                for (k, v) in names {
                    row_names.insert(k, Value::String(v));
                }
            }
        } else {
            annotations.insert(name.clone(), json!({ "query_fields": queries }));
        }
    }

    let (fields, temporal_kinds) = printed_temporal::prepare(&fields);
    let payments = if profile.literal_image_payment_rows() && has_media {
        printed_numbers::payment_lists(&req.fields)
    } else {
        Vec::new()
    };
    let mut payment_rows: Vec<String> = Vec::new();
    for parent in &payments {
        if let Some(q_arr) = annotations
            .get(parent)
            .and_then(|a| a.get("query_fields"))
            .and_then(Value::as_array)
        {
            for q in q_arr {
                if let Some(s) = q.as_str() {
                    payment_rows.push(s.to_string());
                }
            }
        }
    }
    let (fields, number_contracts) = printed_numbers::prepare(&fields, &payment_rows);

    let mut source_row_gates: Vec<Value> = Vec::new();
    for ann in annotations.values() {
        if is_truthy(ann.get("source_row_eligibility"))
            && let Some(sf) = ann.get("source_fields").and_then(Value::as_array)
        {
            for q in sf {
                source_row_gates.push(q.clone());
            }
        }
    }

    let temporal_map: Map<String, Value> = temporal_kinds
        .into_iter()
        .map(|(k, v)| (k, Value::String(v)))
        .collect();

    let record = json!({
        "input": req.state,
        "fields": Value::Object(fields),
        "reference_date": ref_date_val.unwrap_or(Value::Null),
        "_meta": {
            "category": "extraction",
            "construction": schema::EVIDENCE_CONSTRUCTION,
            "extraction_contract": { "fields": Value::Object(annotations) },
            "printed_temporal": Value::Object(temporal_map),
            "printed_numbers": Value::Object(number_contracts),
        },
        "_source_row_inputs": Value::Object(row_inputs),
        "_source_row_names": Value::Object(row_names),
        "_source_row_gates": Value::Array(source_row_gates),
    });

    Ok((record, forecasts))
}

fn typed_readout_record(
    req: &InternalRequest,
    record: &Value,
    automatic_temporal: &Map<String, Value>,
    profile: ClairicProfile,
    cancelled: Option<&Arc<AtomicBool>>,
) -> Result<Option<Value>> {
    let typed = ar::typed_record(record);
    if typed.is_none() || automatic_temporal.is_empty() {
        return Ok(typed);
    }
    let mut restored_fields = req.fields.clone();
    for (k, v) in automatic_temporal {
        restored_fields.insert(k.clone(), v.clone());
    }
    let restored_req = req.with_fields(restored_fields);
    let (original_record, _) = expand_request(&restored_req, profile, cancelled)?;
    Ok(ar::typed_record(&original_record))
}

fn single_token_id(tokenizer: &Tokenizer, text: &str) -> Option<u32> {
    let enc = tokenizer.encode(text, false).ok()?;
    enc.get_ids().first().copied()
}

fn encode_base<B: ClairicBackend + ?Sized>(
    backend: &B,
    profile: ClairicProfile,
    req: &InternalRequest,
    opts: EncodeOptions<'_>,
) -> Result<EncodedRequest> {
    let tokenizer_arc = backend.tokenizer()?;
    let tokenizer = tokenizer_arc.as_ref();
    let eos = backend.eos_token_id()?;
    let has_media = req.has_media();
    let original_request = req.clone();
    let mut active_req = req.clone();

    // 1. Explicit temporal reads
    let mut automatic_temporal: Map<String, Value> = Map::new();
    if profile.explicit_temporal_reads() {
        let routed = printed_temporal::explicit_fields(&active_req.fields);
        for (name, orig_spec) in &active_req.fields {
            if routed.get(name) != Some(orig_spec) {
                automatic_temporal.insert(name.clone(), orig_spec.clone());
            }
        }
        active_req.fields = routed;
    }

    // 2. Entity name scope (27B)
    if profile.preserve_image_entity_names() {
        active_req.fields = scopes::entity::prepare(&active_req.fields, has_media);
    }

    // 3. Identifier role scope
    let (scoped_fields, query_labels) =
        scopes::identifier::prepare_fields(&active_req.fields, has_media);
    active_req.fields = scoped_fields;

    // 4. Sparse image scalar preservation
    if (profile.focus_sparse_images() && active_req.image_count > 0) || opts.pdf_spelling_view {
        active_req.fields = crops::focus::preserve_scalar_fields(&active_req.fields);
    }

    // 5. Expand request
    let (mut record, forecasts) = expand_request(&active_req, profile, opts.cancelled)?;

    let state_only_fields: HashSet<String> = original_request
        .fields
        .iter()
        .filter(|(_, spec)| grounding::state_only(spec))
        .map(|(name, _)| name.clone())
        .collect();

    // 6. Visual PO identifier spelling
    if profile.spell_visual_po_references() && !opts.identifier_spelling_disabled {
        identifier_spelling::prepare(
            &mut record,
            &original_request.fields,
            &query_labels,
            has_media,
        );
    }

    if !query_labels.is_empty() {
        let spelling_keys: HashSet<String> = record
            .get("_meta")
            .and_then(|m| m.get("identifier_spelling"))
            .and_then(Value::as_object)
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        if let Some(names_obj) = record
            .get_mut("_source_row_names")
            .and_then(Value::as_object_mut)
        {
            for (name, label) in &query_labels {
                let display = if spelling_keys.contains(name) {
                    name.clone()
                } else {
                    label.clone()
                };
                names_obj.insert(name.clone(), Value::String(display));
            }
        }
    }

    if opts.physical_row_view
        && let Some(fields_mut) = record.get_mut("fields").and_then(Value::as_object_mut)
    {
        table_rows::flag_physical_row_scope(fields_mut);
    }

    record = image_rows::prepare(record, has_media, opts.share_image_row_description);

    let list_source_bounds =
        source_rows::list_bounds(&active_req.state, &active_req.fields, has_media);
    if let Some(rec_obj) = record.as_object_mut() {
        let bounds_map: Map<String, Value> = list_source_bounds
            .iter()
            .map(|(k, &v)| (k.clone(), json!(v)))
            .collect();
        rec_obj.insert("_list_source_bounds".to_string(), Value::Object(bounds_map));
    }

    // 7. Prepare images & focus views
    let (prepared_images, original_images, default_unfocused_single) = if has_media {
        let should_focus =
            profile.focus_sparse_images() || opts.pdf_spelling_view || opts.po_detail_view;
        let focus_one = |img: &Arc<RgbImage>| -> (Arc<RgbImage>, bool) {
            if profile.is_flash()
                && opts.identifier_view
                && active_req.image_count == 1
                && !active_req.has_documents()
                && opts
                    .unfocused_default_image
                    .is_some_and(|unfocused| Arc::ptr_eq(img, unfocused))
            {
                return (img.clone(), false);
            }
            let focused = if opts.po_detail_view {
                crops::focus::focus_document(img, identifier_spelling::MANY_IMAGE_MAX_PIXELS, 3.0)
            } else {
                crops::focus::focus_document_default(img)
            };
            match focused {
                Some(f) => (Arc::new(f), true),
                None => (img.clone(), false),
            }
        };

        if let Some(ref views) = opts.image_views {
            let originals = views.clone();
            let mut prepared = Vec::with_capacity(views.len());
            for img in views {
                if should_focus {
                    prepared.push(focus_one(img).0);
                } else {
                    prepared.push(img.clone());
                }
            }
            (prepared, originals, false)
        } else {
            let originals = active_req.decoded_images.clone();
            let req_img_count = active_req.image_count.min(originals.len());
            let mut prepared = Vec::with_capacity(originals.len());
            let mut single_unfocused = false;
            for (idx, img) in originals[..req_img_count].iter().enumerate() {
                if should_focus {
                    let (out, was_focused) = focus_one(img);
                    if idx == 0 && req_img_count == 1 && !active_req.has_documents() && !was_focused
                    {
                        single_unfocused = true;
                    }
                    prepared.push(out);
                } else {
                    prepared.push(img.clone());
                }
            }
            // Append PDF pages without default sparse focusing (matching backend.py:587-591)
            for pdf_page in &originals[req_img_count..] {
                prepared.push(pdf_page.clone());
            }
            (prepared, originals, single_unfocused)
        }
    } else {
        (Vec::new(), Vec::new(), false)
    };

    let cache_key = active_req.source_media_key.as_ref().map(|base| {
        let mut k = base.clone();
        if opts.identifier_view {
            k.push_str(":identifier:");
            k.push_str(crops::identifier::VIEW_ID);
        } else {
            k.push_str(":default");
        }
        if let Some(ref vk) = opts.image_view_key {
            k.push_str(":view:");
            k.push_str(vk);
        }
        k
    });
    let cache_salt = Uuid::new_v4().simple().to_string();

    if let Some(rec_obj) = record.as_object_mut() {
        rec_obj.insert(
            "_media_payload".to_string(),
            Value::Bool(!prepared_images.is_empty()),
        );
        rec_obj.insert(
            "_original_images".to_string(),
            Value::Bool(!original_images.is_empty()),
        );
        rec_obj.insert("_cache_salt".to_string(), Value::String(cache_salt.clone()));
    }

    // 8. Build AR task prompts
    let ar_names = ar::names(&record);
    if ar_names.len() > MAX_TOTAL_EXPANDED_TASKS {
        return Err(invalid_request(format!(
            "request expands to {} AR tasks, exceeding the maximum of {MAX_TOTAL_EXPANDED_TASKS}",
            ar_names.len()
        )));
    }
    let mut tasks: Vec<EncodedArTask> = Vec::with_capacity(ar_names.len());
    let mut total_ar_prompt_tokens: usize = 0;
    let mut state_tasks: HashSet<String> = HashSet::new();

    let footer_ids = typed_prompt::tokenize_ids(tokenizer, ASSISTANT_FOOTER)?;
    let (source_ids, _) =
        typed_prompt::tokenize_input_with_char_offsets(tokenizer, &active_req.state)?;

    let (vision_tokens_only, _vision_tok_count) = if !prepared_images.is_empty() {
        backend.build_vision_tokens(&prepared_images)?
    } else {
        (Vec::new(), 0)
    };

    if !prepared_images.is_empty() {
        let mut vision_prefix = typed_prompt::tokenize_ids(tokenizer, USER_HEADER)?;
        vision_prefix.extend_from_slice(&vision_tokens_only);

        let scoped_media: HashSet<String> = original_request
            .fields
            .iter()
            .filter(|(_, spec)| grounding::explicit_media_scope(spec))
            .map(|(n, _)| n.clone())
            .collect();

        for name in &ar_names {
            check_cancel_opt(opts.cancelled)?;
            let is_state_only = state_only_fields.contains(name)
                || record
                    .get("fields")
                    .and_then(|f| f.get(name))
                    .is_some_and(grounding::state_only);

            if is_state_only {
                state_tasks.insert(name.clone());
                let inst = ar::instruction(&record, name, profile.compact_list_prompts())?;
                let mut ids = typed_prompt::tokenize_ids(
                    tokenizer,
                    &format!("{USER_HEADER}{inst}\n\nDocument:\n"),
                )?;
                ids.extend_from_slice(&source_ids);
                ids.extend_from_slice(&footer_ids);
                if ids.len() > MAX_LENGTH {
                    return Err(invalid_request(
                        "AR input exceeds the model context; no input was truncated",
                    ));
                }
                total_ar_prompt_tokens = total_ar_prompt_tokens.saturating_add(ids.len());
                if total_ar_prompt_tokens > MAX_TOTAL_AR_PROMPT_TOKENS {
                    return Err(invalid_request(format!(
                        "total AR prompt tokens ({total_ar_prompt_tokens}) exceed request budget of {MAX_TOTAL_AR_PROMPT_TOKENS}"
                    )));
                }
                tasks.push(EncodedArTask {
                    name: name.clone(),
                    input_ids: ids,
                    media: None,
                    media_key: None,
                    constraint: GrammarSpec::JsonSchema(Value::Bool(true)),
                    constraint_json: Value::Null,
                });
            } else {
                let is_scoped_media = scoped_media.contains(name);
                let inst = ar::instruction(&record, name, profile.compact_list_prompts())?;
                let inst_trimmed = inst.trim_end();
                let mut mid = String::new();
                if !active_req.state.is_empty() && is_scoped_media {
                    mid.push_str(ar::MEDIA_EVIDENCE_INSTRUCTION);
                    mid.push_str("\n\n");
                }
                mid.push_str(inst_trimmed);
                if !active_req.state.is_empty() {
                    if is_scoped_media {
                        mid.push_str("\n\nAccompanying text:\n");
                    } else {
                        mid.push_str("\n\nDocument:\n");
                    }
                }
                let mid_ids = typed_prompt::tokenize_ids(tokenizer, &mid)?;
                let mut ids = Vec::with_capacity(
                    vision_prefix.len() + mid_ids.len() + source_ids.len() + footer_ids.len(),
                );
                ids.extend_from_slice(&vision_prefix);
                ids.extend_from_slice(&mid_ids);
                if !active_req.state.is_empty() {
                    ids.extend_from_slice(&source_ids);
                }
                ids.extend_from_slice(&footer_ids);
                if ids.len() > MAX_LENGTH {
                    return Err(invalid_request(
                        "AR input exceeds the model context; no input was truncated",
                    ));
                }
                total_ar_prompt_tokens = total_ar_prompt_tokens.saturating_add(ids.len());
                if total_ar_prompt_tokens > MAX_TOTAL_AR_PROMPT_TOKENS {
                    return Err(invalid_request(format!(
                        "total AR prompt tokens ({total_ar_prompt_tokens}) exceed request budget of {MAX_TOTAL_AR_PROMPT_TOKENS}"
                    )));
                }
                tasks.push(EncodedArTask {
                    name: name.clone(),
                    input_ids: ids,
                    media: Some(prepared_images.clone()),
                    media_key: cache_key.clone(),
                    constraint: GrammarSpec::JsonSchema(Value::Bool(true)),
                    constraint_json: Value::Null,
                });
            }
        }
    } else {
        let source_row_inputs = record
            .get("_source_row_inputs")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        for name in &ar_names {
            check_cancel_opt(opts.cancelled)?;
            let inst = ar::instruction(&record, name, profile.compact_list_prompts())?;
            let mut ids = typed_prompt::tokenize_ids(
                tokenizer,
                &format!("{USER_HEADER}{inst}\n\nDocument:\n"),
            )?;
            if let Some(row_inp) = source_row_inputs.get(name).and_then(Value::as_str) {
                let (row_ids, _) =
                    typed_prompt::tokenize_input_with_char_offsets(tokenizer, row_inp)?;
                ids.extend_from_slice(&row_ids);
            } else {
                ids.extend_from_slice(&source_ids);
            }
            ids.extend_from_slice(&footer_ids);
            if ids.len() > MAX_LENGTH {
                return Err(invalid_request(
                    "AR input exceeds the model context; no input was truncated",
                ));
            }
            total_ar_prompt_tokens = total_ar_prompt_tokens.saturating_add(ids.len());
            if total_ar_prompt_tokens > MAX_TOTAL_AR_PROMPT_TOKENS {
                return Err(invalid_request(format!(
                    "total AR prompt tokens ({total_ar_prompt_tokens}) exceed request budget of {MAX_TOTAL_AR_PROMPT_TOKENS}"
                )));
            }
            tasks.push(EncodedArTask {
                name: name.clone(),
                input_ids: ids,
                media: None,
                media_key: None,
                constraint: GrammarSpec::JsonSchema(Value::Bool(true)),
                constraint_json: Value::Null,
            });
        }
    }

    // 9. Path B typed readout record
    let typed_rec = typed_readout_record(
        &active_req,
        &record,
        &automatic_temporal,
        profile,
        opts.cancelled,
    )?;
    let (typed, typed_media) = if let Some(typed_val) = typed_rec {
        let media_opt = if !prepared_images.is_empty() {
            Some(EncodedMedia {
                ids: vision_tokens_only,
                expansion: 0,
            })
        } else {
            None
        };
        let enc_opts = EncodeRecordOptions {
            max_length: MAX_LENGTH,
            media: media_opt,
            commits: HashMap::new(),
            answer_slots: true,
            text_answer_slots: true,
            template_variant: 0,
        };
        let item = typed_prompt::encode_record(tokenizer, &typed_val, &enc_opts)?;
        let Some(item) = item else {
            return Err(invalid_request(
                "Typed fields and input must fit one complete context",
            ));
        };
        let expected_text = if !prepared_images.is_empty() && active_req.state.trim().is_empty() {
            ""
        } else {
            active_req.state.as_str()
        };
        if item.text != expected_text {
            return Err(invalid_request(
                "Typed fields and input must fit one complete context",
            ));
        }
        let media_for_typed = if !prepared_images.is_empty() {
            Some(prepared_images.clone())
        } else {
            None
        };
        (Some(item), media_for_typed)
    } else {
        (None, None)
    };

    // 10. Grounding source & per-task constraints
    let shared_text_source =
        if !tasks.is_empty() && (prepared_images.is_empty() || !state_tasks.is_empty()) {
            Some(Arc::new(grounding::Source::new(&active_req.state, true)))
        } else {
            None
        };
    let ar_provenance = if prepared_images.is_empty() {
        shared_text_source.clone()
    } else {
        None
    };
    let ar_source = ar_provenance.clone().map(EvidenceSource::Text);
    let mut ar_provenance_map: HashMap<String, Arc<grounding::Source>> = HashMap::new();
    let mut ar_sources: HashMap<String, EvidenceSource> = HashMap::new();
    if let Some(ref src) = shared_text_source {
        for name in &state_tasks {
            ar_provenance_map.insert(name.clone(), src.clone());
            ar_sources.insert(name.clone(), EvidenceSource::Text(src.clone()));
        }
    }

    let mut constraints_map = Map::new();
    for task in &mut tasks {
        let Some(spec) = record.get("fields").and_then(|f| f.get(&task.name)) else {
            return Err(inference_error(format!(
                "missing field spec for AR task {:?}",
                task.name
            )));
        };
        let src_opt = if task.media.is_none() {
            ar_provenance_map.get(&task.name).or(ar_provenance.as_ref())
        } else {
            None
        };
        let constraint = if let Some(src) = src_opt
            && grounding::applies(spec, task.media.is_some())
        {
            src.constraint(spec)?
        } else {
            GrammarSpec::JsonSchema(ar::schema(spec)?)
        };
        let c_json = constraint_to_json(&constraint)?;
        constraints_map.insert(task.name.clone(), c_json.clone());
        task.constraint = constraint;
        task.constraint_json = c_json;
    }

    if let Some(rec_obj) = record.as_object_mut() {
        rec_obj.insert(
            "_ar_constraints".to_string(),
            Value::Object(constraints_map),
        );
    }

    let mut encoded = EncodedRequest {
        record,
        forecasts,
        tasks,
        typed,
        typed_media,
        eos,
        ar_source,
        ar_sources,
        ar_provenance,
        ar_provenance_map,
        original_images,
        default_unfocused_single,
        list_source_bounds,
        identifier_spelling_fallback: None,
        date_supplement: None,
        receipt_sum_fields: Vec::new(),
        cache_salt,
    };

    // 11. Visual detail view planners (`backend.py:815-873`)
    if !opts.is_flash_detail() {
        if profile.bind_image_table_rows() {
            prepare_image_table_rows(
                backend,
                profile,
                &active_req,
                &mut encoded,
                profile.image_table_row_scale() as f64,
            )?;
        }
        if profile.bind_image_ragged_rows() {
            prepare_image_ragged_rows(
                backend,
                profile,
                &active_req,
                &mut encoded,
                profile.image_table_row_scale() as f64,
            )?;
        }
        if profile.bind_image_financial_lines() {
            prepare_image_financial_lines(backend, profile, &active_req, &mut encoded)?;
        }
        if profile.focus_image_identifiers() {
            prepare_image_identifiers(backend, profile, &original_request, &mut encoded)?;
        }
        if profile.focus_image_party_names() {
            prepare_party_name_tasks(backend, profile, &original_request, &mut encoded)?;
        }
        prepare_pdf_number_focus(
            backend,
            profile,
            &original_request,
            &mut encoded,
            profile.focus_sparse_images(),
        )?;
        prepare_identifier_spelling_pdf_tasks(backend, profile, &original_request, &mut encoded)?;
        prepare_identifier_spelling_many_image_tasks(
            backend,
            profile,
            &original_request,
            &mut encoded,
        )?;
        prepare_identifier_spelling_fallback(backend, profile, &original_request, &mut encoded)?;
    }

    if let Some(rec_obj) = encoded.record.as_object_mut() {
        rec_obj.insert(
            "_first_item_list_gating".to_string(),
            Value::Bool(opts.share_image_row_description),
        );
    }

    Ok(encoded)
}

/// 27B encode wrapper adding literal-scalar detail view and PO horizontal supplement (`large.py:30-161`).
fn encode_for_profile<B: ClairicBackend + ?Sized>(
    backend: &B,
    profile: ClairicProfile,
    req: &InternalRequest,
    mut opts: EncodeOptions<'_>,
) -> Result<EncodedRequest> {
    if !profile.is_large_27b() {
        return encode_base(backend, profile, req, opts);
    }

    let supplement_disabled = opts.po_supplement_disabled;
    let detail = opts.is_large_detail();
    if opts.po_detail_view {
        opts.pdf_spelling_view = true;
    }

    let mut encoded = encode_base(backend, profile, req, opts)?;

    if !detail {
        apply_27b_literal_scalar_view(backend, profile, req, &mut encoded)?;
    }

    if !supplement_disabled && !detail {
        apply_27b_po_supplement(backend, profile, req, &mut encoded)?;
    }

    Ok(encoded)
}

fn replace_selected_tasks(
    encoded: &mut EncodedRequest,
    selected_names: &[String],
    child: EncodedRequest,
    copy_fields: bool,
) {
    let mut by_name: HashMap<String, EncodedArTask> = child
        .tasks
        .into_iter()
        .map(|t| (t.name.clone(), t))
        .collect();
    for task in &mut encoded.tasks {
        if let Some(replacement) = by_name.remove(&task.name) {
            *task = replacement;
        }
    }
    for name in selected_names {
        if copy_fields
            && let Some(c_field) = child
                .record
                .get("fields")
                .and_then(|f| f.get(name))
                .cloned()
            && let Some(fields_mut) = encoded
                .record
                .get_mut("fields")
                .and_then(Value::as_object_mut)
        {
            fields_mut.insert(name.clone(), c_field);
        }
        if let Some(c_cons) = child
            .record
            .get("_ar_constraints")
            .and_then(|c| c.get(name))
            .cloned()
            && let Some(cons_mut) = encoded
                .record
                .get_mut("_ar_constraints")
                .and_then(Value::as_object_mut)
        {
            cons_mut.insert(name.clone(), c_cons);
        }
    }
}

fn prepare_image_table_rows<B: ClairicBackend + ?Sized>(
    backend: &B,
    profile: ClairicProfile,
    req: &InternalRequest,
    encoded: &mut EncodedRequest,
    scale: f64,
) -> Result<()> {
    if req.image_count != 1 || req.has_documents() {
        return Ok(());
    }
    let parents: Vec<String> = encoded
        .record
        .get("_meta")
        .and_then(|m| m.get("extraction_contract"))
        .and_then(|c| c.get("fields"))
        .and_then(Value::as_object)
        .map(|anns| {
            anns.iter()
                .filter(|(_, ann)| {
                    ann.get("image_row_fields")
                        .and_then(Value::as_array)
                        .is_some_and(|a| !a.is_empty())
                })
                .map(|(n, _)| n.clone())
                .collect()
        })
        .unwrap_or_default();
    if parents.is_empty() {
        return Ok(());
    }
    let Some(image) = encoded.original_images.first().cloned() else {
        return Ok(());
    };
    let Some(table) = crops::table_rows::detect(&image) else {
        return Ok(());
    };
    let views = table.scaled_views(&image, scale, VISION_MAX_PIXELS);

    for name in parents {
        let Some(spec) = req.fields.get(&name) else {
            continue;
        };
        let max_items = spec
            .get("max_items")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(schema::MAX_LIST_ITEMS);
        if views.len() > max_items {
            continue;
        }
        let previous: Vec<String> = encoded
            .record
            .get("_meta")
            .and_then(|m| m.get("extraction_contract"))
            .and_then(|c| c.get("fields"))
            .and_then(|f| f.get(&name))
            .and_then(|a| a.get("image_row_fields"))
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        if views.len() > previous.len() {
            continue;
        }
        let Some(row_spec) = crops::table_rows::physical_row_spec(spec) else {
            continue;
        };

        let mut replacements: Vec<EncodedArTask> = Vec::with_capacity(views.len());
        for (index, view) in views.iter().enumerate() {
            let child = &previous[index];
            let view_field = "source_row".to_string();
            let mut row_fields = Map::new();
            row_fields.insert(view_field.clone(), row_spec.clone());
            let row_req = req.with_fields(row_fields);
            let mut row_encoded = encode_for_profile(
                backend,
                profile,
                &row_req,
                EncodeOptions {
                    physical_row_view: true,
                    image_views: Some(vec![Arc::new(view.clone())]),
                    image_view_key: crops::table_rows::view_cache_key(&table, index, scale),
                    ..EncodeOptions::default()
                },
            )?;
            if row_encoded.tasks.len() != 1 {
                return Err(inference_error("Expected 1 physical table row task"));
            }
            let mut row_task = row_encoded.tasks.remove(0);
            row_task.name = child.clone();
            if let Some(f_val) = row_encoded
                .record
                .get("fields")
                .and_then(|f| f.get(&view_field))
                .cloned()
                && let Some(fields_mut) = encoded
                    .record
                    .get_mut("fields")
                    .and_then(Value::as_object_mut)
            {
                fields_mut.insert(child.clone(), f_val);
            }
            if let Some(cons_mut) = encoded
                .record
                .get_mut("_ar_constraints")
                .and_then(Value::as_object_mut)
            {
                cons_mut.insert(child.clone(), row_task.constraint_json.clone());
            }
            replacements.push(row_task);
        }

        let mut replaced_names: HashSet<String> = previous.iter().cloned().collect();
        replaced_names.insert(name.clone());
        encoded.tasks.retain(|t| !replaced_names.contains(&t.name));
        let retained: HashSet<String> = replacements.iter().map(|t| t.name.clone()).collect();
        let replacement_names: Vec<Value> = replacements
            .iter()
            .map(|t| Value::String(t.name.clone()))
            .collect();
        encoded.tasks.extend(replacements);

        for child in &previous {
            if !retained.contains(child) {
                if let Some(fields_mut) = encoded
                    .record
                    .get_mut("fields")
                    .and_then(Value::as_object_mut)
                {
                    fields_mut.remove(child);
                }
                if let Some(cons_mut) = encoded
                    .record
                    .get_mut("_ar_constraints")
                    .and_then(Value::as_object_mut)
                {
                    cons_mut.remove(child);
                }
                if let Some(anns_mut) = encoded
                    .record
                    .get_mut("_meta")
                    .and_then(|m| m.get_mut("extraction_contract"))
                    .and_then(|c| c.get_mut("fields"))
                    .and_then(Value::as_object_mut)
                {
                    anns_mut.remove(child);
                }
            }
        }
        if let Some(ann_mut) = encoded
            .record
            .get_mut("_meta")
            .and_then(|m| m.get_mut("extraction_contract"))
            .and_then(|c| c.get_mut("fields"))
            .and_then(|f| f.get_mut(&name))
            .and_then(Value::as_object_mut)
        {
            ann_mut.remove("image_row_fields");
            ann_mut.insert(
                "geometric_row_fields".to_string(),
                Value::Array(replacement_names),
            );
        }
        if let Some(parent_spec) = encoded
            .record
            .get_mut("fields")
            .and_then(|f| f.get_mut(&name))
            .and_then(Value::as_object_mut)
        {
            parent_spec.remove("_image_row_inventory");
        }
    }

    Ok(())
}

fn prepare_image_ragged_rows<B: ClairicBackend + ?Sized>(
    backend: &B,
    profile: ClairicProfile,
    req: &InternalRequest,
    encoded: &mut EncodedRequest,
    scale: f64,
) -> Result<()> {
    if req.image_count != 1 || req.has_documents() {
        return Ok(());
    }
    let parents: Vec<String> = {
        let anns = encoded
            .record
            .get("_meta")
            .and_then(|m| m.get("extraction_contract"))
            .and_then(|c| c.get("fields"))
            .and_then(Value::as_object);
        req.fields
            .iter()
            .filter(|(name, spec)| {
                crops::ragged_rows::is_eligible(spec)
                    && anns
                        .and_then(|a| a.get(*name))
                        .and_then(|v| v.get("image_row_fields"))
                        .and_then(Value::as_array)
                        .is_some_and(|arr| !arr.is_empty())
            })
            .map(|(n, _)| n.clone())
            .collect()
    };
    if parents.is_empty() {
        return Ok(());
    }
    let Some(image) = encoded.original_images.first().cloned() else {
        return Ok(());
    };
    let Some(plan) = crops::ragged_rows::detect(&image) else {
        return Ok(());
    };
    let views = plan.scaled_views(&image, scale, VISION_MAX_PIXELS);

    let mut gates: Vec<Value> = encoded
        .record
        .get("_source_row_gates")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    for name in parents {
        let Some(spec) = req.fields.get(&name) else {
            continue;
        };
        let previous: Vec<String> = encoded
            .record
            .get("_meta")
            .and_then(|m| m.get("extraction_contract"))
            .and_then(|c| c.get("fields"))
            .and_then(|f| f.get(&name))
            .and_then(|a| a.get("image_row_fields"))
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let maximum = spec
            .get("max_items")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(schema::MAX_LIST_ITEMS);
        if views.len() > maximum + 1 || views.len() > previous.len() {
            continue;
        }
        let Some(row_spec_fields) = crops::ragged_rows::row_fields(spec) else {
            continue;
        };

        let mut replacements: Vec<EncodedArTask> = Vec::with_capacity(views.len() * 2);
        let mut pairs: Vec<Value> = Vec::with_capacity(views.len());
        let mut retained: HashSet<String> = HashSet::new();

        for (index, (view, &band)) in views.iter().zip(plan.rows.iter()).enumerate() {
            let child = &previous[index];
            let existing: HashSet<String> = encoded
                .record
                .get("fields")
                .and_then(Value::as_object)
                .map(|m| m.keys().cloned().collect())
                .unwrap_or_default();
            let gate = crops::ragged_rows::gate_field_name(child, &existing);

            let row_req = req.with_fields(row_spec_fields.clone());
            let row_encoded = encode_for_profile(
                backend,
                profile,
                &row_req,
                EncodeOptions {
                    physical_row_view: true,
                    image_views: Some(vec![image.clone(), Arc::new(view.clone())]),
                    image_view_key: Some(crops::ragged_rows::view_cache_key(band, scale)),
                    ..EncodeOptions::default()
                },
            )?;
            if row_encoded.tasks.len() != 2 {
                return Err(inference_error("Expected 2 ragged row tasks (gate + row)"));
            }

            for mut rtask in row_encoded.tasks {
                let source_name = rtask.name.clone();
                let target = if source_name == crops::ragged_rows::TARGET_GATE {
                    gate.clone()
                } else {
                    child.clone()
                };
                rtask.name = target.clone();
                if let Some(f_val) = row_encoded
                    .record
                    .get("fields")
                    .and_then(|f| f.get(&source_name))
                    .cloned()
                    && let Some(fields_mut) = encoded
                        .record
                        .get_mut("fields")
                        .and_then(Value::as_object_mut)
                {
                    fields_mut.insert(target.clone(), f_val);
                }
                if let Some(cons_mut) = encoded
                    .record
                    .get_mut("_ar_constraints")
                    .and_then(Value::as_object_mut)
                {
                    cons_mut.insert(target.clone(), rtask.constraint_json.clone());
                }
                if let Some(anns_mut) = encoded
                    .record
                    .get_mut("_meta")
                    .and_then(|m| m.get_mut("extraction_contract"))
                    .and_then(|c| c.get_mut("fields"))
                    .and_then(Value::as_object_mut)
                {
                    anns_mut.insert(target.clone(), json!({ "parent": name }));
                }
                replacements.push(rtask);
            }
            gates.push(Value::String(gate.clone()));
            retained.insert(child.clone());
            pairs.push(json!([gate, child]));
        }

        let mut replaced_names: HashSet<String> = previous.iter().cloned().collect();
        replaced_names.insert(name.clone());
        encoded.tasks.retain(|t| !replaced_names.contains(&t.name));
        encoded.tasks.extend(replacements);

        for child in &previous {
            if !retained.contains(child) {
                if let Some(fields_mut) = encoded
                    .record
                    .get_mut("fields")
                    .and_then(Value::as_object_mut)
                {
                    fields_mut.remove(child);
                }
                if let Some(cons_mut) = encoded
                    .record
                    .get_mut("_ar_constraints")
                    .and_then(Value::as_object_mut)
                {
                    cons_mut.remove(child);
                }
                if let Some(anns_mut) = encoded
                    .record
                    .get_mut("_meta")
                    .and_then(|m| m.get_mut("extraction_contract"))
                    .and_then(|c| c.get_mut("fields"))
                    .and_then(Value::as_object_mut)
                {
                    anns_mut.remove(child);
                }
            }
        }

        if let Some(ann_mut) = encoded
            .record
            .get_mut("_meta")
            .and_then(|m| m.get_mut("extraction_contract"))
            .and_then(|c| c.get_mut("fields"))
            .and_then(|f| f.get_mut(&name))
            .and_then(Value::as_object_mut)
        {
            ann_mut.remove("image_row_fields");
            ann_mut.insert("ragged_row_fields".to_string(), Value::Array(pairs));
            ann_mut.insert("ragged_row_max_items".to_string(), json!(maximum));
        }
        if let Some(parent_spec) = encoded
            .record
            .get_mut("fields")
            .and_then(|f| f.get_mut(&name))
            .and_then(Value::as_object_mut)
        {
            parent_spec.remove("_image_row_inventory");
        }
    }

    if let Some(rec_obj) = encoded.record.as_object_mut() {
        rec_obj.insert("_source_row_gates".to_string(), Value::Array(gates));
    }

    Ok(())
}

fn prepare_image_financial_lines<B: ClairicBackend + ?Sized>(
    backend: &B,
    profile: ClairicProfile,
    req: &InternalRequest,
    encoded: &mut EncodedRequest,
) -> Result<()> {
    if req.image_count != 1 || req.has_documents() {
        return Ok(());
    }
    let mut selected: Vec<(String, crops::financial_lines::FinancialLineRole)> = {
        let anns = encoded
            .record
            .get("_meta")
            .and_then(|m| m.get("extraction_contract"))
            .and_then(|c| c.get("fields"))
            .and_then(Value::as_object);
        req.fields
            .iter()
            .filter_map(|(name, spec)| {
                let kind = crops::financial_lines::scope(spec)?;
                let has_qf = anns
                    .and_then(|a| a.get(name))
                    .and_then(|v| v.get("query_fields"))
                    .and_then(Value::as_array)
                    .is_some_and(|arr| !arr.is_empty());
                if has_qf {
                    Some((name.clone(), kind))
                } else {
                    None
                }
            })
            .collect()
    };
    if selected.is_empty() {
        return Ok(());
    }
    let Some(image) = encoded.original_images.first().cloned() else {
        return Ok(());
    };
    let Some(rows) = crops::financial_lines::bands(&image) else {
        return Ok(());
    };
    if crops::table_rows::detect(&image).is_some() || crops::ragged_rows::detect(&image).is_some() {
        selected.retain(|(_, kind)| *kind != crops::financial_lines::FinancialLineRole::Charge);
    }
    if selected.is_empty() {
        return Ok(());
    }

    let add_child = |child_req: &InternalRequest,
                     image_views: Option<Vec<Arc<RgbImage>>>,
                     image_view_key: Option<String>,
                     encoded_ref: &mut EncodedRequest|
     -> Result<Vec<String>> {
        let child = encode_for_profile(
            backend,
            profile,
            child_req,
            EncodeOptions {
                physical_row_view: true,
                image_views,
                image_view_key,
                ..EncodeOptions::default()
            },
        )?;
        let mut added_names = Vec::with_capacity(child.tasks.len());
        for mut task in child.tasks {
            let source_name = task.name.clone();
            let field_count = encoded_ref
                .record
                .get("fields")
                .and_then(Value::as_object)
                .map_or(0, Map::len);
            let mut target = format!("financial_line_{field_count}");
            while encoded_ref
                .record
                .get("fields")
                .and_then(Value::as_object)
                .is_some_and(|m| m.contains_key(&target))
            {
                target.push('_');
            }
            task.name = target.clone();
            if let Some(f_val) = child
                .record
                .get("fields")
                .and_then(|f| f.get(&source_name))
                .cloned()
                && let Some(fields_mut) = encoded_ref
                    .record
                    .get_mut("fields")
                    .and_then(Value::as_object_mut)
            {
                fields_mut.insert(target.clone(), f_val);
            }
            if let Some(cons_mut) = encoded_ref
                .record
                .get_mut("_ar_constraints")
                .and_then(Value::as_object_mut)
            {
                cons_mut.insert(target.clone(), task.constraint_json.clone());
            }
            encoded_ref.tasks.push(task);
            added_names.push(target);
        }
        Ok(added_names)
    };

    let mut queries: Vec<Value> = Vec::with_capacity(rows.len());
    let views = crops::financial_lines::band_views(&image, &rows);
    for ((top, bottom), view) in rows.into_iter().zip(views) {
        let mut line_fields = Map::new();
        line_fields.insert(
            "printed_line".to_string(),
            crops::financial_lines::line_spec(),
        );
        let child_req = req.with_fields(line_fields);
        let names = add_child(
            &child_req,
            Some(vec![Arc::new(view)]),
            Some(crops::financial_lines::view_cache_key(top, bottom)),
            encoded,
        )?;
        for n in names {
            queries.push(Value::String(n));
        }
    }

    let mut plans = Map::new();
    for (name, kind) in selected {
        let Some(spec) = req.fields.get(&name).cloned() else {
            continue;
        };
        let mut pres_fields = Map::new();
        pres_fields.insert(name.clone(), crops::financial_lines::presence_spec(&spec));
        let pres_req = req.with_fields(pres_fields);
        let gate_names = add_child(&pres_req, None, None, encoded)?;
        let Some(gate) = gate_names.into_iter().next() else {
            continue;
        };

        let previous: Vec<String> = encoded
            .record
            .get_mut("_meta")
            .and_then(|m| m.get_mut("extraction_contract"))
            .and_then(|c| c.get_mut("fields"))
            .and_then(|f| f.get_mut(&name))
            .and_then(Value::as_object_mut)
            .and_then(|a| a.remove("query_fields"))
            .and_then(|v| v.as_array().cloned())
            .map(|arr| {
                arr.into_iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        let mut removed: HashSet<String> = previous.iter().cloned().collect();
        removed.insert(name.clone());
        encoded.tasks.retain(|t| !removed.contains(&t.name));
        for child in &previous {
            if let Some(fields_mut) = encoded
                .record
                .get_mut("fields")
                .and_then(Value::as_object_mut)
            {
                fields_mut.remove(child);
            }
            if let Some(cons_mut) = encoded
                .record
                .get_mut("_ar_constraints")
                .and_then(Value::as_object_mut)
            {
                cons_mut.remove(child);
            }
            if let Some(anns_mut) = encoded
                .record
                .get_mut("_meta")
                .and_then(|m| m.get_mut("extraction_contract"))
                .and_then(|c| c.get_mut("fields"))
                .and_then(Value::as_object_mut)
            {
                anns_mut.remove(child);
            }
        }
        plans.insert(
            name,
            json!({
                "kind": kind.as_str(),
                "presence": gate,
                "spec": spec,
            }),
        );
    }

    if let Some(rec_obj) = encoded.record.as_object_mut() {
        rec_obj.insert("_financial_line_queries".to_string(), Value::Array(queries));
        rec_obj.insert("_financial_line_plans".to_string(), Value::Object(plans));
    }

    Ok(())
}

fn prepare_image_identifiers<B: ClairicBackend + ?Sized>(
    backend: &B,
    profile: ClairicProfile,
    req: &InternalRequest,
    encoded: &mut EncodedRequest,
) -> Result<()> {
    let selected = crops::identifier::fields(&req.fields);
    if selected.is_empty() || req.image_count != 1 || req.has_documents() {
        return Ok(());
    }
    let Some(image) = encoded.original_images.first().cloned() else {
        return Ok(());
    };
    let skip_focus =
        profile.is_flash() && profile.focus_sparse_images() && encoded.default_unfocused_single;
    let views: Vec<Arc<RgbImage>> =
        crops::identifier::content_views(&image, VISION_MAX_PIXELS, skip_focus)
            .into_iter()
            .map(Arc::new)
            .collect();
    let mut child_fields = Map::new();
    for name in &selected {
        if let Some(spec) = req.fields.get(name) {
            child_fields.insert(name.clone(), spec.clone());
        }
    }
    let child_req = req.with_fields(child_fields);
    let child = encode_for_profile(
        backend,
        profile,
        &child_req,
        EncodeOptions {
            identifier_view: true,
            image_views: Some(views),
            unfocused_default_image: if skip_focus { Some(&image) } else { None },
            ..EncodeOptions::default()
        },
    )?;
    replace_selected_tasks(encoded, &selected, child, false);
    Ok(())
}

fn prepare_party_name_tasks<B: ClairicBackend + ?Sized>(
    backend: &B,
    profile: ClairicProfile,
    req: &InternalRequest,
    encoded: &mut EncodedRequest,
) -> Result<()> {
    let selected = crops::party_names::fields(&req.fields);
    if selected.is_empty() || req.image_count != 1 || req.has_documents() {
        return Ok(());
    }
    let Some(image) = encoded.original_images.first().cloned() else {
        return Ok(());
    };
    let views = crops::party_names::views(&image, VISION_MAX_PIXELS);
    if views.len() == 1 {
        return Ok(());
    }
    let mut child_fields = Map::new();
    for name in &selected {
        if let Some(spec) = req.fields.get(name).and_then(Value::as_object) {
            let mut updated = spec.clone();
            let existing = spec
                .get("instructions")
                .and_then(Value::as_str)
                .unwrap_or("");
            updated.insert(
                "instructions".to_string(),
                Value::String(format!(
                    "{existing}{}",
                    crops::party_names::VIEW_INSTRUCTION
                )),
            );
            child_fields.insert(name.clone(), Value::Object(updated));
        }
    }
    let child_req = req.with_fields(child_fields);
    let view_arcs: Vec<Arc<RgbImage>> = views.into_iter().map(Arc::new).collect();
    let child = encode_for_profile(
        backend,
        profile,
        &child_req,
        EncodeOptions {
            party_name_view: true,
            image_views: Some(view_arcs),
            image_view_key: Some(crops::party_names::VIEW_ID.to_string()),
            ..EncodeOptions::default()
        },
    )?;
    replace_selected_tasks(encoded, &selected, child, true);
    Ok(())
}

fn prepare_pdf_number_focus<B: ClairicBackend + ?Sized>(
    backend: &B,
    profile: ClairicProfile,
    req: &InternalRequest,
    encoded: &mut EncodedRequest,
    enabled: bool,
) -> Result<()> {
    if !enabled || !req.has_documents() {
        return Ok(());
    }
    let active: HashSet<String> = encoded.tasks.iter().map(|t| t.name.clone()).collect();
    let pdf_pages: Vec<RgbImage> = encoded
        .original_images
        .iter()
        .skip(req.image_count)
        .map(|arc| (**arc).clone())
        .collect();
    let selected = crops::pdf_number_focus::eligible_fields(
        enabled,
        req.has_documents(),
        &req.fields,
        &active,
        &pdf_pages,
    );
    if selected.is_empty() {
        return Ok(());
    }
    let mut child_fields = Map::new();
    for name in &selected {
        if let Some(spec) = req.fields.get(name) {
            child_fields.insert(name.clone(), spec.clone());
        }
    }
    let child = encode_for_profile(
        backend,
        profile,
        &req.with_fields(child_fields),
        EncodeOptions {
            identifier_view: true,
            image_views: Some(encoded.original_images.clone()),
            image_view_key: Some(crops::pdf_number_focus::VIEW_ID.to_string()),
            ..EncodeOptions::default()
        },
    )?;
    if child.typed.is_some() || child.eos != encoded.eos {
        return Err(invalid_request(
            "A literal PDF number view must retain the scalar AR contract",
        ));
    }
    replace_selected_tasks(encoded, &selected, child, false);
    Ok(())
}

fn prepare_identifier_spelling_pdf_tasks<B: ClairicBackend + ?Sized>(
    backend: &B,
    profile: ClairicProfile,
    req: &InternalRequest,
    encoded: &mut EncodedRequest,
) -> Result<()> {
    if !req.has_documents() {
        return Ok(());
    }
    let active: HashSet<String> = encoded.tasks.iter().map(|t| t.name.clone()).collect();
    let selected = identifier_spelling::selected_spelling_fields(&encoded.record, &active, true);
    let pdf_page_count = encoded
        .original_images
        .len()
        .saturating_sub(req.image_count);
    if selected.is_empty() || pdf_page_count == 0 {
        return Ok(());
    }
    let mut child_fields = Map::new();
    for name in &selected {
        if let Some(spec) = req.fields.get(name) {
            child_fields.insert(name.clone(), spec.clone());
        }
    }
    let child = encode_for_profile(
        backend,
        profile,
        &req.with_fields(child_fields),
        EncodeOptions {
            identifier_view: true,
            pdf_spelling_view: true,
            image_views: Some(encoded.original_images.clone()),
            image_view_key: Some(identifier_spelling::PDF_VIEW_ID.to_string()),
            ..EncodeOptions::default()
        },
    )?;
    if child.typed.is_some() || child.eos != encoded.eos {
        return Err(invalid_request(
            "A PDF PO spelling view must retain the scalar AR contract",
        ));
    }
    let meta_val = identifier_spelling::pdf_view_metadata(
        &selected,
        encoded.original_images.len(),
        pdf_page_count,
    );
    replace_selected_tasks(encoded, &selected, child, true);
    if let Some(meta_mut) = encoded
        .record
        .get_mut("_meta")
        .and_then(Value::as_object_mut)
    {
        meta_mut.insert("identifier_spelling_pdf_view".to_string(), meta_val);
    }
    Ok(())
}

fn prepare_identifier_spelling_many_image_tasks<B: ClairicBackend + ?Sized>(
    backend: &B,
    profile: ClairicProfile,
    req: &InternalRequest,
    encoded: &mut EncodedRequest,
) -> Result<()> {
    if req.has_documents() || !matches!(req.image_count, 3 | 4) {
        return Ok(());
    }
    let active: HashSet<String> = encoded.tasks.iter().map(|t| t.name.clone()).collect();
    let selected = identifier_spelling::selected_spelling_fields(&encoded.record, &active, true);
    if selected.is_empty() {
        return Ok(());
    }
    if encoded.original_images.len() != req.image_count {
        return Err(invalid_request(
            "PO detail view must retain every original image in order",
        ));
    }
    let mut child_fields = Map::new();
    for name in &selected {
        if let Some(spec) = req.fields.get(name) {
            child_fields.insert(name.clone(), spec.clone());
        }
    }
    let child = encode_for_profile(
        backend,
        profile,
        &req.with_fields(child_fields),
        EncodeOptions {
            identifier_view: true,
            po_detail_view: true,
            image_views: Some(encoded.original_images.clone()),
            image_view_key: Some(identifier_spelling::MANY_IMAGE_VIEW_ID.to_string()),
            ..EncodeOptions::default()
        },
    )?;
    if child.typed.is_some() || child.eos != encoded.eos {
        return Err(invalid_request(
            "A PO image detail view must retain the scalar AR contract",
        ));
    }
    let meta_val =
        identifier_spelling::many_image_view_metadata(&selected, encoded.original_images.len());
    replace_selected_tasks(encoded, &selected, child, true);
    if let Some(meta_mut) = encoded
        .record
        .get_mut("_meta")
        .and_then(Value::as_object_mut)
    {
        meta_mut.insert("identifier_spelling_image_view".to_string(), meta_val);
    }
    Ok(())
}

fn prepare_identifier_spelling_fallback<B: ClairicBackend + ?Sized>(
    backend: &B,
    profile: ClairicProfile,
    req: &InternalRequest,
    encoded: &mut EncodedRequest,
) -> Result<()> {
    let active: HashSet<String> = encoded.tasks.iter().map(|t| t.name.clone()).collect();
    let selected = identifier_spelling::selected_spelling_fields(&encoded.record, &active, false);
    if selected.is_empty() {
        return Ok(());
    }
    let mut child_fields = Map::new();
    for name in &selected {
        if let Some(spec) = req.fields.get(name) {
            child_fields.insert(name.clone(), spec.clone());
        }
    }
    let child = encode_for_profile(
        backend,
        profile,
        &req.with_fields(child_fields),
        EncodeOptions {
            identifier_spelling_disabled: true,
            ..EncodeOptions::default()
        },
    )?;
    if child.typed.is_some() || child.eos != encoded.eos {
        return Err(invalid_request(
            "Identifier spelling fallback must retain the original scalar contract",
        ));
    }
    encoded.identifier_spelling_fallback = Some(Box::new(child));
    Ok(())
}

fn apply_27b_literal_scalar_view<B: ClairicBackend + ?Sized>(
    backend: &B,
    profile: ClairicProfile,
    req: &InternalRequest,
    encoded: &mut EncodedRequest,
) -> Result<()> {
    let active: HashSet<String> = encoded.tasks.iter().map(|t| t.name.clone()).collect();
    let mut po_fields = Vec::new();
    if matches!(req.image_count, 1 | 2) && !req.has_documents() {
        po_fields = identifier_spelling::selected_spelling_fields(&encoded.record, &active, true);
    }
    let mut number_fields = Vec::new();
    if req.image_count > 1 || req.has_documents() {
        for (name, spec) in &req.fields {
            if active.contains(name)
                && spec.get("type").and_then(Value::as_str) == Some("number")
                && spec.get("source_only") == Some(&Value::Bool(true))
                && !is_truthy(spec.get("knowledge"))
                && !grounding::state_only(spec)
            {
                number_fields.push(name.clone());
            }
        }
    }
    let mut selected_names: Vec<String> = Vec::new();
    let mut child_fields = Map::new();
    for name in po_fields.iter().chain(number_fields.iter()) {
        if !child_fields.contains_key(name)
            && let Some(spec) = req.fields.get(name)
        {
            selected_names.push(name.clone());
            child_fields.insert(name.clone(), spec.clone());
        }
    }
    if selected_names.is_empty() {
        return Ok(());
    }
    let child = encode_base(
        backend,
        profile,
        &req.with_fields(child_fields),
        EncodeOptions {
            identifier_view: true,
            pdf_spelling_view: true,
            image_views: Some(encoded.original_images.clone()),
            image_view_key: Some("27b-literal-scalars-all-content2-v1".to_string()),
            ..EncodeOptions::default()
        },
    )?;
    if child.typed.is_some() || child.eos != encoded.eos {
        return Err(invalid_request(
            "27B scalar detail must retain the original scalar AR contract",
        ));
    }
    let img_count = encoded.original_images.len();
    replace_selected_tasks(encoded, &selected_names, child, true);
    if let Some(meta_mut) = encoded
        .record
        .get_mut("_meta")
        .and_then(Value::as_object_mut)
    {
        meta_mut.insert(
            "model_document_scalar_view".to_string(),
            json!({
                "version": "27b-literal-scalars-all-content2-v1",
                "fields": selected_names,
                "po_fields": po_fields,
                "number_fields": number_fields,
                "image_count": img_count,
            }),
        );
    }
    Ok(())
}

fn plain_po_task(
    task: &EncodedArTask,
    wire_spec: &Value,
    public_spec: &Value,
    tokenizer: &Tokenizer,
    max_length: usize,
) -> Result<(EncodedArTask, Value, GrammarSpec, Value)> {
    let text = tokenizer
        .decode(&task.input_ids, false)
        .map_err(|e| invalid_request(format!("PO scalar decode failed: {e}")))?;
    let rt = typed_prompt::tokenize_ids(tokenizer, &text)?;
    if rt != task.input_ids {
        return Err(invalid_request(
            "PO scalar source prompt cannot make a lossless tokenizer roundtrip",
        ));
    }
    let before = format!("{}\n", identifier_spelling::INSTRUCTION);
    let wire_inst = wire_spec
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    if text.matches(&before).count() != 1 || wire_inst.matches(&before).count() != 1 {
        return Err(invalid_request(
            "PO scalar source must contain exactly one spelling instruction",
        ));
    }
    let mut scalar_obj = wire_spec.as_object().cloned().unwrap_or_default();
    scalar_obj.insert(
        "instructions".to_string(),
        Value::String(wire_inst.replacen(&before, "", 1)),
    );
    if let Some(pub_obj) = public_spec.as_object() {
        for key in ["minLength", "maxLength", "pattern"] {
            if let Some(v) = pub_obj.get(key) {
                scalar_obj.insert(key.to_string(), v.clone());
            }
        }
    }
    let scalar = Value::Object(scalar_obj);
    let old_schema = ar::schema(wire_spec)?;
    let new_schema = ar::schema(&scalar)?;
    let mut rewritten = text.replacen(&before, "", 1);
    if new_schema != old_schema {
        let old = format!("\nSchema: {}", ar::compact_json(&old_schema));
        let new = format!("\nSchema: {}", ar::compact_json(&new_schema));
        if rewritten.matches(&old).count() != 1 {
            return Err(invalid_request(
                "PO scalar source must contain exactly one native schema",
            ));
        }
        rewritten = rewritten.replacen(&old, &new, 1);
    }
    let plain_ids = typed_prompt::tokenize_ids(tokenizer, &rewritten)?;
    if plain_ids.len() > max_length {
        return Err(invalid_request(
            "PO scalar detail exceeds the complete model context",
        ));
    }
    let grammar = GrammarSpec::JsonSchema(new_schema);
    let c_json = constraint_to_json(&grammar)?;
    let mut updated = task.clone();
    updated.input_ids = plain_ids;
    updated.constraint = grammar.clone();
    updated.constraint_json = c_json.clone();
    Ok((updated, scalar, grammar, c_json))
}

fn apply_27b_po_supplement<B: ClairicBackend + ?Sized>(
    backend: &B,
    profile: ClairicProfile,
    req: &InternalRequest,
    encoded: &mut EncodedRequest,
) -> Result<()> {
    let active_media: HashSet<String> = encoded
        .tasks
        .iter()
        .filter(|t| t.media.is_some())
        .map(|t| t.name.clone())
        .collect();
    let spelling_wire: HashMap<String, String> = encoded
        .record
        .get("_meta")
        .and_then(|m| m.get("identifier_spelling"))
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| {
                    v.get("wire")
                        .and_then(Value::as_str)
                        .map(|w| (k.clone(), w.to_string()))
                })
                .collect()
        })
        .unwrap_or_default();

    let selected = crops::po_supplement::selected_fields(
        &req.fields,
        req.image_count,
        req.has_documents(),
        &active_media,
        &spelling_wire,
    );
    if selected.is_empty() {
        return Ok(());
    }
    if encoded.original_images.len() != req.image_count {
        return Err(invalid_request(
            "PO supplement must retain every original image in order",
        ));
    }
    let raw_originals: Vec<RgbImage> = encoded
        .original_images
        .iter()
        .map(|arc| (**arc).clone())
        .collect();
    let supplements = crops::po_supplement::build_supplements(&raw_originals)?;
    let mut supp_rgbs: Vec<Arc<RgbImage>> = Vec::with_capacity(supplements.len());
    for s in &supplements {
        supp_rgbs.push(Arc::new(s.to_rgb_image()?));
    }

    let mut selected_map = Map::new();
    for name in &selected {
        if let Some(spec) = req.fields.get(name) {
            selected_map.insert(name.clone(), spec.clone());
        }
    }
    let preserved = crops::focus::preserve_scalar_fields(&selected_map);

    let mut child_images = req.decoded_images.clone();
    child_images.extend(supp_rgbs);
    let mut hasher = Sha256::new();
    if let Some(ref base) = req.source_media_key {
        hasher.update(base.as_bytes());
    }
    hasher.update(crops::po_supplement::VERSION.as_bytes());
    let child_req = InternalRequest {
        state: req.state.clone(),
        image_count: child_images.len(),
        pdf_page_count: 0,
        decoded_images: child_images,
        source_media_key: Some(hex_digest(&hasher.finalize())),
        fields: preserved,
        reference_date: req.reference_date.clone(),
    };

    let mut child = encode_for_profile(
        backend,
        profile,
        &child_req,
        EncodeOptions {
            po_supplement_disabled: true,
            ..EncodeOptions::default()
        },
    )?;
    if child.typed.is_some() || child.eos != encoded.eos {
        return Err(invalid_request(
            "PO supplement must retain the complete selected scalar task contract",
        ));
    }

    let plain = req.image_count == 1;
    let tokenizer_arc = backend.tokenizer()?;
    let tokenizer = tokenizer_arc.as_ref();

    if plain {
        for task in &mut child.tasks {
            if let (Some(wire_spec), Some(public_spec)) = (
                child
                    .record
                    .get("fields")
                    .and_then(|f| f.get(&task.name))
                    .cloned(),
                req.fields.get(&task.name),
            ) {
                let (plain_t, scalar_spec, _g, c_json) =
                    plain_po_task(task, &wire_spec, public_spec, tokenizer, MAX_LENGTH)?;
                *task = plain_t;
                if let Some(f_mut) = child
                    .record
                    .get_mut("fields")
                    .and_then(Value::as_object_mut)
                {
                    f_mut.insert(task.name.clone(), scalar_spec);
                }
                if let Some(c_mut) = child
                    .record
                    .get_mut("_ar_constraints")
                    .and_then(Value::as_object_mut)
                {
                    c_mut.insert(task.name.clone(), c_json);
                }
            }
        }
    }

    let native_image_count = child_req.image_count;
    replace_selected_tasks(encoded, &selected, child, true);

    let selected_set: HashSet<&str> = selected.iter().map(String::as_str).collect();
    if let Some(meta_mut) = encoded
        .record
        .get_mut("_meta")
        .and_then(Value::as_object_mut)
    {
        if plain
            && let Some(contracts) = meta_mut
                .get("identifier_spelling")
                .and_then(Value::as_object)
                .cloned()
        {
            let remaining: Map<String, Value> = contracts
                .into_iter()
                .filter(|(k, _)| !selected_set.contains(k.as_str()))
                .collect();
            if remaining.is_empty() {
                meta_mut.remove("identifier_spelling");
            } else {
                meta_mut.insert("identifier_spelling".to_string(), Value::Object(remaining));
            }
        }
        for key in [
            "model_document_scalar_view",
            "identifier_spelling_image_view",
        ] {
            if let Some(prev_obj) = meta_mut.get(key).and_then(Value::as_object).cloned() {
                let remaining: Vec<Value> = prev_obj
                    .get("fields")
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter(|v| v.as_str().is_none_or(|s| !selected_set.contains(s)))
                            .cloned()
                            .collect()
                    })
                    .unwrap_or_default();
                if remaining.is_empty() {
                    meta_mut.remove(key);
                } else {
                    let mut retained = prev_obj;
                    retained.insert("fields".to_string(), Value::Array(remaining));
                    if let Some(po_arr) = retained.get("po_fields").and_then(Value::as_array) {
                        let rem_po: Vec<Value> = po_arr
                            .iter()
                            .filter(|v| v.as_str().is_none_or(|s| !selected_set.contains(s)))
                            .cloned()
                            .collect();
                        retained.insert("po_fields".to_string(), Value::Array(rem_po));
                    }
                    meta_mut.insert(key.to_string(), Value::Object(retained));
                }
            }
        }
        meta_mut.insert(
            "model_po_supplement".to_string(),
            json!({
                "version": crops::po_supplement::VERSION,
                "fields": selected,
                "original_image_count": req.image_count,
                "native_image_count": native_image_count,
                "horizontal_scale": 2.0,
                "supplement_max_pixels": crops::po_supplement::MAX_PIXELS,
                "wire": if plain { "native_scalar" } else { identifier_spelling::VERSION },
            }),
        );
    }

    Ok(())
}

/// Top-level request encoder (`backend.py::_encode_request` + `large.py::_encode_request`).
fn encode_request<B: ClairicBackend + ?Sized>(
    backend: &B,
    profile: ClairicProfile,
    req: &InternalRequest,
    cancelled: Option<&Arc<AtomicBool>>,
) -> Result<EncodedRequest> {
    let share_desc = profile.is_flash() && req.image_count == 1 && !req.has_documents();
    let mut encoded = encode_for_profile(
        backend,
        profile,
        req,
        EncodeOptions {
            share_image_row_description: share_desc,
            cancelled,
            ..EncodeOptions::default()
        },
    )?;

    // Source date supplement (text-only requests)
    if !req.has_media() {
        let printed_temporal_set: HashSet<String> = encoded
            .record
            .get("_meta")
            .and_then(|m| m.get("printed_temporal"))
            .and_then(Value::as_object)
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        if let Some(plan) = source_dates::build_supplement_plan(
            &req.state,
            &req.fields,
            &printed_temporal_set,
            false,
        )? {
            let internal_req = req.with_fields(plan.internal_fields);
            if let Ok(mut dated) = encode_for_profile(
                backend,
                profile,
                &internal_req,
                EncodeOptions {
                    cancelled,
                    ..EncodeOptions::default()
                },
            ) {
                let date_src = Arc::new(plan.source);
                for task in &mut dated.tasks {
                    if let Some(spec) = plan.constraints.get(&task.name) {
                        task.constraint = spec.clone();
                        task.constraint_json = constraint_to_json(spec)?;
                        if let Some(cons_mut) = dated
                            .record
                            .get_mut("_ar_constraints")
                            .and_then(Value::as_object_mut)
                        {
                            cons_mut.insert(task.name.clone(), task.constraint_json.clone());
                        }
                    }
                }
                dated.ar_source = Some(EvidenceSource::Date(date_src.clone()));
                dated.ar_sources.clear();
                for name in &plan.names {
                    dated
                        .ar_sources
                        .insert(name.clone(), EvidenceSource::Date(date_src.clone()));
                }
                if let Some(rec_obj) = dated.record.as_object_mut() {
                    rec_obj.insert("_null_recovery".to_string(), Value::Bool(true));
                }
                encoded.date_supplement = Some(EncodedDateSupplement {
                    encoded: Box::new(dated),
                    source: date_src,
                    names: plan.names,
                });
            }
        }
    }

    // Flash text-only receipt sum fields
    if profile.is_flash() && !req.has_media() {
        let receipt_fields = receipt::totals::eligible_fields(&req.state, &req.fields, false);
        if !receipt_fields.is_empty() {
            if let Some(rec_obj) = encoded.record.as_object_mut() {
                rec_obj.insert(
                    "_receipt_sum_fields".to_string(),
                    Value::Array(receipt_fields.iter().cloned().map(Value::String).collect()),
                );
            }
            encoded.receipt_sum_fields = receipt_fields;
        }
    }

    // 27B media-state prompt rewriting & first-item list gating (`large.py:163-184`)
    if profile.is_large_27b() && req.has_media() {
        if let Some(rec_obj) = encoded.record.as_object_mut() {
            rec_obj.insert("_first_item_list_gating".to_string(), Value::Bool(true));
        }
        if let Some(ref mut fb) = encoded.identifier_spelling_fallback
            && let Some(rec_obj) = fb.record.as_object_mut()
        {
            rec_obj.insert("_first_item_list_gating".to_string(), Value::Bool(true));
        }
        let tokenizer_arc = backend.tokenizer()?;
        let tokenizer = tokenizer_arc.as_ref();
        let raw_tasks: Vec<(String, Vec<u32>, Option<usize>)> = encoded
            .tasks
            .iter()
            .enumerate()
            .map(|(i, t)| {
                (
                    t.name.clone(),
                    t.input_ids.clone(),
                    t.media.as_ref().map(|_| i),
                )
            })
            .collect();
        if let Some(rewritten) = media_state_prompt::prepare_tasks(
            &req.state,
            &req.fields,
            req.image_count,
            usize::from(req.has_documents()),
            &encoded.record,
            &raw_tasks,
            tokenizer,
            MAX_LENGTH,
        ) {
            for (task, (_, new_ids, _)) in encoded.tasks.iter_mut().zip(rewritten) {
                task.input_ids = new_ids;
            }
        }
    }

    Ok(encoded)
}

// =============================================================================
// Path A result parsing (`backend.py::scalar_result`)
// =============================================================================

#[allow(clippy::too_many_arguments)]
fn scalar_result(
    result: &ArBatchItemResult,
    spec: &Value,
    eos: u32,
    null_id: Option<u32>,
    excluded_token_id: Option<u32>,
    tokenizer: &Tokenizer,
    source_text: &str,
    evidence: Option<&EvidenceSource>,
    provenance: Option<&grounding::Source>,
) -> Result<Value> {
    let complete = result.finish_reason_type.as_deref() == Some("stop")
        && result.finish_reason_matched == Some(eos);
    let raw = if let Some(ref ids) = result.output_ids {
        let slice = if ids.last() == Some(&eos) {
            &ids[..ids.len().saturating_sub(1)]
        } else {
            ids.as_slice()
        };
        tokenizer
            .decode(slice, false)
            .map_err(|e| inference_error(format!("failed to decode AR output: {e}")))?
    } else {
        result
            .text
            .strip_suffix(ar::EOS)
            .unwrap_or(&result.text)
            .to_string()
    };
    let raw = raw.trim().to_string();

    let mut value = Value::Null;
    let mut valid = false;
    let mut span: Option<[usize; 2]> = None;
    let mut failure_reason: Option<&'static str> = None;

    if complete {
        let schema_val = ar::schema(spec)?;
        if let Ok(mut parsed) = serde_json::from_str::<Value>(&raw) {
            let finite_ok = parsed.as_f64().is_none_or(f64::is_finite);
            if finite_ok && json_validate::is_valid(&schema_val, &parsed) {
                let mut ok = true;
                if !parsed.is_null()
                    && let Some(ev) = evidence
                {
                    let (found, found_span) = ev.span(&parsed, spec);
                    if !found {
                        failure_reason = Some("not_in_source");
                        ok = false;
                    } else {
                        span = found_span;
                    }
                }
                if ok
                    && !parsed.is_null()
                    && spec.get("type").and_then(Value::as_str) == Some("number")
                    && schema::allowed_decimals(spec).is_some()
                {
                    match ar::normalize_number(&Value::String(raw.clone()), spec) {
                        Ok(norm) if json_validate::is_valid(&schema_val, &norm) => {
                            parsed = norm;
                        }
                        _ => ok = false,
                    }
                }
                if ok
                    && !parsed.is_null()
                    && spec.get("type").and_then(Value::as_str) == Some("object")
                    && let (Some(parsed_obj), Some(props)) = (
                        parsed.as_object(),
                        spec.get("properties").and_then(Value::as_object),
                    )
                {
                    let mut norm_obj = Map::with_capacity(props.len());
                    for (prop_name, child_spec) in props {
                        let Some(prop_val) = parsed_obj.get(prop_name) else {
                            ok = false;
                            break;
                        };
                        if child_spec.get("type").and_then(Value::as_str) == Some("number") {
                            match ar::normalize_number(prop_val, child_spec) {
                                Ok(norm) => {
                                    norm_obj.insert(prop_name.clone(), norm);
                                }
                                Err(_) => {
                                    ok = false;
                                    break;
                                }
                            }
                        } else {
                            norm_obj.insert(prop_name.clone(), prop_val.clone());
                        }
                    }
                    if ok {
                        let candidate = Value::Object(norm_obj);
                        if json_validate::is_valid(&schema_val, &candidate) {
                            parsed = candidate;
                        } else {
                            ok = false;
                        }
                    }
                }
                if ok {
                    value = parsed;
                    valid = true;
                }
            }
        }
    }

    let mut pred = Map::new();
    pred.insert("value".to_string(), value.clone());
    pred.insert("valid".to_string(), Value::Bool(valid));
    pred.insert("complete".to_string(), Value::Bool(complete));
    pred.insert(
        "decoder".to_string(),
        Value::String(ar::VERSION.to_string()),
    );
    pred.insert("tokens".to_string(), json!(result.completion_tokens));
    pred.insert("raw".to_string(), Value::String(raw));
    pred.insert(
        "span".to_string(),
        if valid {
            span.map_or(Value::Null, |[s, e]| json!([s, e]))
        } else {
            Value::Null
        },
    );

    if let Some(reason) = failure_reason {
        pred.insert(
            "failure_reason".to_string(),
            Value::String(reason.to_string()),
        );
    }

    if valid
        && !value.is_null()
        && let Some(ev) = evidence
    {
        let ws_norm =
            ev.is_whitespace_normalized() && value.as_str().is_some_and(|s| !ev.text().contains(s));
        pred.insert(
            "grounding_method".to_string(),
            Value::String(
                if ws_norm {
                    "whitespace_normalized_source"
                } else {
                    "literal_source"
                }
                .to_string(),
            ),
        );
        if span.is_some() {
            pred.insert(
                "span_method".to_string(),
                Value::String(
                    if ws_norm {
                        "unique_whitespace_normalized_match"
                    } else {
                        "unique_literal_match"
                    }
                    .to_string(),
                ),
            );
        }
    }

    let first_scores = result.output_token_ids_logprobs.first();
    if let Some(excl_id) = excluded_token_id {
        let score = first_scores
            .and_then(|scores| scores.iter().find(|(_, tid)| *tid == excl_id))
            .map(|(lp, _)| *lp);
        let Some(score) = score else {
            return Err(inference_error(
                "Engine omitted a valid row-exclusion log probability",
            ));
        };
        if score.is_nan() || score > 1e-6 {
            return Err(inference_error(
                "Engine omitted a valid row-exclusion log probability",
            ));
        }
        pred.insert("p_excluded".to_string(), json!(score.min(0.0).exp()));
        pred.insert(
            "p_excluded_method".to_string(),
            Value::String("ar_first_token_false_probability".to_string()),
        );
    }

    if spec.get("nullable") == Some(&Value::Bool(false)) {
        pred.insert("p_null".to_string(), json!(0.0));
        pred.insert(
            "p_null_method".to_string(),
            Value::String("schema_disallows_null".to_string()),
        );
    } else if let (Some(nid), Some(scores)) = (null_id, first_scores) {
        let score = scores
            .iter()
            .find(|(_, tid)| *tid == nid)
            .map(|(lp, _)| *lp);
        let Some(score) = score else {
            return Err(inference_error(
                "Engine omitted a valid null-token log probability",
            ));
        };
        if score.is_nan() || score > 1e-6 {
            return Err(inference_error(
                "Engine omitted a valid null-token log probability",
            ));
        }
        pred.insert("p_null".to_string(), json!(score.min(0.0).exp()));
        pred.insert(
            "p_null_method".to_string(),
            Value::String("ar_first_token_null_probability".to_string()),
        );
    }

    if valid && !result.output_token_logprobs.is_empty() {
        let log_prob: f64 = result.output_token_logprobs.iter().map(|(lp, _)| *lp).sum();
        if !log_prob.is_finite() || log_prob > 1e-6 {
            return Err(inference_error(
                "Engine returned invalid token log probabilities",
            ));
        }
        pred.insert("confidence".to_string(), json!(log_prob.min(0.0).exp()));
        pred.insert("log_probability".to_string(), json!(log_prob));
        pred.insert(
            "confidence_method".to_string(),
            Value::String("ar_sequence_probability".to_string()),
        );
    } else if !valid {
        pred.insert("confidence".to_string(), json!(0.0));
    }

    // Provenance spans for ungrounded numbers & strings (`backend.py:365-381`)
    if valid
        && evidence.is_none()
        && !value.is_null()
        && spec.get("type").and_then(Value::as_str) == Some("number")
        && let Some(prov) = provenance
    {
        let (found, lit_span) = prov.span(&value, None);
        if found && let Some([s, e]) = lit_span {
            pred.insert("span".to_string(), json!([s, e]));
            pred.insert(
                "span_method".to_string(),
                Value::String("unique_literal_match".to_string()),
            );
        }
    }
    if valid
        && evidence.is_none()
        && !source_text.is_empty()
        && let Some(s) = value.as_str()
        && !s.is_empty()
        && let Some(byte_start) = source_text.find(s)
    {
        let next_byte = source_text[byte_start..]
            .chars()
            .next()
            .map_or(source_text.len(), |ch| byte_start + ch.len_utf8());
        if source_text[next_byte..].find(s).is_none() {
            let char_start = source_text[..byte_start].chars().count();
            let char_end = char_start + s.chars().count();
            pred.insert("span".to_string(), json!([char_start, char_end]));
            pred.insert(
                "span_method".to_string(),
                Value::String("unique_literal_match".to_string()),
            );
        }
    }

    Ok(Value::Object(pred))
}

// =============================================================================
// AR execution, completion schedulers, and recovery loops (`backend.py` + `text_ar.py` + `large.py`)
// =============================================================================

fn build_null_recovery(
    encoded: &EncodedRequest,
    fields: &HashSet<String>,
    tokenizer: &Tokenizer,
) -> Result<Option<EncodedRequest>> {
    let orig_tokens = typed_prompt::tokenize_ids(tokenizer, ar::EXTRACTION_INSTRUCTION)?;
    if orig_tokens.is_empty() {
        return Ok(None);
    }
    let mut repair_tasks: Vec<EncodedArTask> = Vec::new();
    for task in &encoded.tasks {
        if !fields.contains(&task.name) {
            continue;
        }
        let has_source = encoded
            .ar_sources
            .get(&task.name)
            .or(encoded.ar_source.as_ref())
            .is_some();
        let instruction = if has_source {
            ar::COMPUTATION_INSTRUCTION
        } else {
            ar::RECOVERY_INSTRUCTION
        };
        let replacement = typed_prompt::tokenize_ids(tokenizer, instruction)?;
        let matches: Vec<usize> = task
            .input_ids
            .windows(orig_tokens.len())
            .enumerate()
            .filter_map(|(i, win)| (win == orig_tokens.as_slice()).then_some(i))
            .collect();
        if matches.len() != 1 {
            continue;
        }
        let start = matches[0];
        let mut prompt =
            Vec::with_capacity(task.input_ids.len() - orig_tokens.len() + replacement.len());
        prompt.extend_from_slice(&task.input_ids[..start]);
        prompt.extend_from_slice(&replacement);
        prompt.extend_from_slice(&task.input_ids[start + orig_tokens.len()..]);
        if prompt.len() <= MAX_LENGTH {
            let mut updated = task.clone();
            updated.input_ids = prompt;
            repair_tasks.push(updated);
        }
    }
    if repair_tasks.is_empty() {
        return Ok(None);
    }
    let mut repair_record = encoded.record.clone();
    if let Some(obj) = repair_record.as_object_mut() {
        obj.insert("_null_recovery".to_string(), Value::Bool(true));
    }
    Ok(Some(EncodedRequest {
        record: repair_record,
        forecasts: Vec::new(),
        tasks: repair_tasks,
        typed: None,
        typed_media: None,
        eos: encoded.eos,
        ar_source: encoded.ar_source.clone(),
        ar_sources: encoded.ar_sources.clone(),
        ar_provenance: encoded.ar_provenance.clone(),
        ar_provenance_map: encoded.ar_provenance_map.clone(),
        original_images: encoded.original_images.clone(),
        default_unfocused_single: false,
        list_source_bounds: BTreeMap::new(),
        identifier_spelling_fallback: None,
        date_supplement: None,
        receipt_sum_fields: Vec::new(),
        cache_salt: encoded.cache_salt.clone(),
    }))
}

fn execute_ar_all<B: ClairicBackend + ?Sized>(
    backend: &mut B,
    profile: ClairicProfile,
    encoded: &mut EncodedRequest,
    cancelled: Option<&Arc<AtomicBool>>,
) -> Result<Map<String, Value>> {
    check_cancel_opt(cancelled)?;
    if encoded.tasks.is_empty() {
        return Ok(Map::new());
    }

    let tokenizer_arc = backend.tokenizer()?;
    let eos = encoded.eos;
    let row_gates: HashSet<String> = encoded
        .record
        .get("_source_row_gates")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    let null_id = single_token_id(&tokenizer_arc, "null");
    let false_id = if !row_gates.is_empty() {
        single_token_id(&tokenizer_arc, "false")
    } else {
        None
    };

    let mut score_ids: Vec<u32> = Vec::with_capacity(2);
    if let Some(nid) = null_id {
        score_ids.push(nid);
    }
    if let Some(fid) = false_id
        && !score_ids.contains(&fid)
    {
        score_ids.push(fid);
    }

    let descriptors: Vec<ArTaskDescriptor> = encoded
        .tasks
        .iter()
        .map(|t| ArTaskDescriptor {
            name: t.name.clone(),
            rid: format!("ar-{}", Uuid::new_v4().simple()),
            input_ids: t.input_ids.clone(),
            has_media: t.media.is_some(),
        })
        .collect();

    let max_out_default = encoded
        .record
        .get("_ar_max_output_tokens")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(MAX_OUTPUT_TOKENS);

    let decode_tok = tokenizer_arc.clone();
    let decode_tokens = move |ids: &[u32]| -> Result<String> {
        decode_tok
            .decode(ids, false)
            .map_err(|e| inference_error(format!("failed to decode tokens: {e}")))
    };

    let parse_tok = tokenizer_arc.clone();
    let parse_for_cohort = move |item: &ArBatchItemResult, spec: &Value, src_text: &str| -> Value {
        scalar_result(
            item, spec, eos, null_id, None, &parse_tok, src_text, None, None,
        )
        .unwrap_or_else(|_| json!({"value": null, "valid": false, "complete": false}))
    };
    let normalize_for_cohort = |rec: &Value, preds: &Map<String, Value>| -> Map<String, Value> {
        let after_temporal = printed_temporal::normalize_predictions(rec, preds);
        printed_numbers::normalize_predictions(rec, &after_temporal)
    };
    let cohort_hooks = CohortHooks {
        parse_result: &parse_for_cohort,
        normalize: &normalize_for_cohort,
    };

    let tasks_snapshot = encoded.tasks.clone();
    let fields_snapshot = encoded.record.get("fields").cloned().unwrap_or(Value::Null);
    let record_for_limits = encoded.record.clone();

    let run_batch = |indices: &[usize]| -> Result<Vec<ArBatchItemResult>> {
        check_cancel_opt(cancelled)?;
        let mut gen_tasks: Vec<ArGenTask> = Vec::with_capacity(indices.len());
        for &idx in indices {
            let task = &tasks_snapshot[idx];
            let is_inv = fields_snapshot
                .get(&task.name)
                .is_some_and(image_rows::is_inventory);
            let max_new = if is_inv {
                image_rows::output_limit(
                    &record_for_limits,
                    &task.name,
                    task.input_ids.len(),
                    MAX_LENGTH,
                )
            } else {
                max_out_default
            };
            gen_tasks.push(ArGenTask {
                id: descriptors[idx].rid.clone(),
                input_ids: task.input_ids.clone(),
                media: task.media.clone(),
                media_key: task.media_key.clone(),
                grammar: task.constraint.clone(),
                max_new_tokens: max_new,
                stop_token_ids: vec![eos],
            });
        }
        let outputs = backend.generate_batch(&gen_tasks, &score_ids, cancelled)?;
        Ok(outputs
            .into_iter()
            .map(|out| {
                let (finish_type, finish_matched) = match out.finish_reason {
                    FinishReason::Stop(tok) => (Some("stop".to_string()), Some(tok)),
                    FinishReason::Length => (Some("length".to_string()), None),
                };
                let completion_tokens = out.output_ids.len();
                let output_token_logprobs: Vec<(f64, u32)> = out
                    .output_token_logprobs
                    .into_iter()
                    .map(|(lp, tid)| (f64::from(lp), tid))
                    .collect();
                let first_step: Vec<(f64, u32)> = out
                    .first_step_token_logprobs
                    .into_iter()
                    .map(|(lp, tid)| (f64::from(lp), tid))
                    .collect();
                ArBatchItemResult {
                    id: out.id,
                    output_ids: Some(out.output_ids),
                    text: out.text,
                    completion_tokens,
                    finish_reason_type: finish_type,
                    finish_reason_matched: finish_matched,
                    output_token_logprobs,
                    output_token_ids_logprobs: vec![first_step],
                    unused_row_proposal: false,
                    omission_marker: None,
                }
            })
            .collect())
    };

    let use_text_ar =
        profile.is_large_27b() && completion::text_list_eligible(&encoded.record, &descriptors);

    let batch_results = if use_text_ar {
        completion::text_list_generate(
            &mut encoded.record,
            &descriptors,
            eos,
            &decode_tokens,
            Some(&cohort_hooks),
            run_batch,
        )?
    } else {
        let frontiers = if profile.is_flash() {
            OrdinaryListFrontiers::FLASH_DEFAULT
        } else {
            OrdinaryListFrontiers::DISABLED
        };
        completion::ordinary_list_generate(
            &mut encoded.record,
            &descriptors,
            eos,
            frontiers,
            &decode_tokens,
            Some(&cohort_hooks),
            run_batch,
        )?
    };

    let input_text = encoded
        .record
        .get("input")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    let mut predictions = Map::new();
    for (task, item) in encoded.tasks.iter().zip(batch_results.iter()) {
        if item.unused_row_proposal {
            predictions.insert(task.name.clone(), completion::unused_prediction(item));
            continue;
        }
        let Some(spec) = encoded.record.get("fields").and_then(|f| f.get(&task.name)) else {
            return Err(inference_error(format!(
                "missing field spec for {:?}",
                task.name
            )));
        };
        let ev = if grounding::applies(spec, task.media.is_some()) {
            encoded
                .ar_sources
                .get(&task.name)
                .or(encoded.ar_source.as_ref())
        } else {
            None
        };
        let prov = if task.media.is_none() && !is_truthy(spec.get("knowledge")) {
            encoded
                .ar_provenance_map
                .get(&task.name)
                .or(encoded.ar_provenance.as_ref())
                .map(Arc::as_ref)
        } else {
            None
        };
        let excl_id = if row_gates.contains(&task.name) {
            false_id
        } else {
            None
        };
        let pred = scalar_result(
            item,
            spec,
            eos,
            null_id,
            excl_id,
            &tokenizer_arc,
            &input_text,
            ev,
            prov,
        )?;
        predictions.insert(task.name.clone(), pred);
    }

    // Verify every non-unused prediction has `confidence` and `p_null`
    for pred in predictions.values() {
        if is_truthy(pred.get("_unused_row_proposal")) {
            continue;
        }
        if pred.get("confidence").is_none() {
            return Err(inference_error(
                "Engine omitted generation log probabilities",
            ));
        }
        if pred.get("p_null").is_none() {
            return Err(inference_error(
                "Engine omitted null-token log probabilities",
            ));
        }
    }

    // Recovery pass 1: Null / ungrounded recovery (`RECOVER_NULLS`)
    if profile.recover_nulls() && !is_truthy(encoded.record.get("_null_recovery")) {
        let mut recovery_fields: HashSet<String> = HashSet::new();
        let temporal_meta = encoded
            .record
            .get("_meta")
            .and_then(|m| m.get("printed_temporal"))
            .and_then(Value::as_object);
        let numbers_meta = encoded
            .record
            .get("_meta")
            .and_then(|m| m.get("printed_numbers"))
            .and_then(Value::as_object);
        let contract_fields = encoded
            .record
            .get("_meta")
            .and_then(|m| m.get("extraction_contract"))
            .and_then(|c| c.get("fields"))
            .and_then(Value::as_object);

        for (name, pred) in &predictions {
            let is_null_or_ungrounded = (is_truthy(pred.get("valid"))
                && pred.get("value").is_some_and(Value::is_null))
                || pred.get("failure_reason").and_then(Value::as_str) == Some("not_in_source");
            if !is_null_or_ungrounded {
                continue;
            }
            let Some(spec) = encoded.record.get("fields").and_then(|f| f.get(name)) else {
                continue;
            };
            if is_truthy(spec.get("knowledge")) {
                continue;
            }
            if temporal_meta.is_some_and(|m| m.contains_key(name))
                || numbers_meta.is_some_and(|m| m.contains_key(name))
            {
                continue;
            }
            let kind = spec.get("type").and_then(Value::as_str).unwrap_or("");
            if !(kind == "number" || (kind == "text" && !input_text.is_empty())) {
                continue;
            }
            let has_parent = contract_fields
                .and_then(|f| f.get(name))
                .and_then(|a| a.get("parent"))
                .and_then(Value::as_str)
                .is_some_and(|p| !p.is_empty());
            if has_parent {
                continue;
            }
            recovery_fields.insert(name.clone());
        }

        if !recovery_fields.is_empty()
            && let Some(mut repair) =
                build_null_recovery(encoded, &recovery_fields, &tokenizer_arc)?
        {
            let retry_lengths: HashMap<String, usize> = repair
                .tasks
                .iter()
                .map(|t| (t.name.clone(), t.input_ids.len()))
                .collect();
            let alternatives = execute_ar_all(backend, profile, &mut repair, cancelled)?;
            for (name, alt) in alternatives {
                let Some(orig) = predictions.get(&name).cloned() else {
                    continue;
                };
                let orig_tokens = orig.get("tokens").and_then(Value::as_u64).unwrap_or(0);
                let alt_tokens = alt.get("tokens").and_then(Value::as_u64).unwrap_or(0);
                let total_tokens = orig_tokens + alt_tokens;

                let kind = encoded
                    .record
                    .get("fields")
                    .and_then(|f| f.get(&name))
                    .and_then(|s| s.get("type"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let alt_gm = alt.get("grounding_method").and_then(Value::as_str);
                let accept = is_truthy(alt.get("valid"))
                    && alt.get("value").is_some_and(|v| !v.is_null())
                    && (matches!(
                        alt_gm,
                        Some("literal_source" | "whitespace_normalized_source")
                    ) || kind == "number"
                        || (alt.get("span").is_some_and(|v| !v.is_null())
                            && alt.get("confidence").and_then(Value::as_f64).unwrap_or(0.0)
                                > alt.get("p_null").and_then(Value::as_f64).unwrap_or(1.0)));

                let mut final_pred = if accept {
                    let mut m = alt.as_object().cloned().unwrap_or_default();
                    m.insert("null_recovered".to_string(), Value::Bool(true));
                    m
                } else {
                    orig.as_object().cloned().unwrap_or_default()
                };
                final_pred.insert("tokens".to_string(), json!(total_tokens));
                if let Some(&rlen) = retry_lengths.get(&name) {
                    final_pred.insert("retry_input_tokens".to_string(), json!(rlen));
                }
                predictions.insert(name, Value::Object(final_pred));
            }
        }
    }

    // Recovery pass 2: Image row omission recovery (`row_omission.py`, skipped on `text_ar` path)
    if !use_text_ar {
        let task_infos: Vec<row_omission::TaskInfo> = encoded
            .tasks
            .iter()
            .map(|t| row_omission::TaskInfo {
                name: t.name.clone(),
                input_len: t.input_ids.len(),
                has_media: t.media.is_some(),
            })
            .collect();
        if let Some(plan) = row_omission::prepare(&encoded.record, &predictions, &task_infos) {
            let task_by_name: HashMap<&str, &EncodedArTask> =
                encoded.tasks.iter().map(|t| (t.name.as_str(), t)).collect();
            let mut retry_tasks = Vec::with_capacity(plan.retry_task_names.len());
            for child in &plan.retry_task_names {
                if let Some(&orig_task) = task_by_name.get(child.as_str()) {
                    let mut updated = orig_task.clone();
                    if let Some(c_val) = plan.retry_constraints.get(child)
                        && let Some(schema_str) = c_val.get("json_schema").and_then(Value::as_str)
                        && let Ok(parsed_schema) = serde_json::from_str::<Value>(schema_str)
                    {
                        updated.constraint = GrammarSpec::JsonSchema(parsed_schema);
                        updated.constraint_json = c_val.clone();
                    }
                    retry_tasks.push(updated);
                }
            }
            if !retry_tasks.is_empty() {
                let mut repair = EncodedRequest {
                    record: plan.repair_record.clone(),
                    forecasts: Vec::new(),
                    tasks: retry_tasks,
                    typed: None,
                    typed_media: None,
                    eos: encoded.eos,
                    ar_source: encoded.ar_source.clone(),
                    ar_sources: encoded.ar_sources.clone(),
                    ar_provenance: encoded.ar_provenance.clone(),
                    ar_provenance_map: encoded.ar_provenance_map.clone(),
                    original_images: encoded.original_images.clone(),
                    default_unfocused_single: false,
                    list_source_bounds: BTreeMap::new(),
                    identifier_spelling_fallback: None,
                    date_supplement: None,
                    receipt_sum_fields: Vec::new(),
                    cache_salt: encoded.cache_salt.clone(),
                };
                let alternatives = execute_ar_all(backend, profile, &mut repair, cancelled)?;
                let (merged, _) = row_omission::merge(&predictions, &alternatives, &plan);
                predictions = merged;
            }
        }
    }

    Ok(predictions)
}

// =============================================================================
// Receipt totals boolean repair pass (`backend.py:1170-1291`)
// =============================================================================

struct ReceiptRepairOutcome {
    updates: Map<String, Value>,
    input_tokens: usize,
    task_count: usize,
    output_tokens: usize,
    max_tokens: usize,
}

fn execute_receipt_repairs<B: ClairicBackend + ?Sized>(
    backend: &mut B,
    profile: ClairicProfile,
    req: &InternalRequest,
    encoded: &EncodedRequest,
    predictions: &Map<String, Value>,
    cancelled: Option<&Arc<AtomicBool>>,
) -> Result<ReceiptRepairOutcome> {
    let mut outcome = ReceiptRepairOutcome {
        updates: Map::new(),
        input_tokens: 0,
        task_count: 0,
        output_tokens: 0,
        max_tokens: 0,
    };
    if encoded.receipt_sum_fields.is_empty() {
        return Ok(outcome);
    }

    let mut source_rec = encoded.record.as_object().cloned().unwrap_or_default();
    source_rec.insert("input".to_string(), Value::String(req.state.clone()));
    source_rec.insert("fields".to_string(), Value::Object(req.fields.clone()));
    let prepared = receipt::totals::prepare_repairs(&Value::Object(source_rec), predictions, false);
    if prepared.is_empty() {
        return Ok(outcome);
    }

    let tokenizer = backend.tokenizer()?;

    for repair in prepared {
        check_cancel_opt(cancelled)?;
        let mut all_tasks: Vec<EncodedArTask> = Vec::with_capacity(repair.tasks.len());
        let mut aux_fields = Map::new();
        let mut aux_constraints = Map::new();
        let salt = Uuid::new_v4().simple().to_string();

        for spec in &repair.tasks {
            let prompt_text = format!("{USER_HEADER}{}{ASSISTANT_FOOTER}", spec.prompt);
            let ids = typed_prompt::tokenize_ids(&tokenizer, &prompt_text)?;
            if ids.is_empty() || ids.len() > MAX_LENGTH {
                return Err(invalid_request(
                    "Receipt selector exceeds the model context; no input was truncated",
                ));
            }
            let c_json = constraint_to_json(&spec.grammar)?;
            aux_fields.insert(
                spec.occurrence_id.clone(),
                json!({ "type": "boolean", "nullable": true }),
            );
            aux_constraints.insert(spec.occurrence_id.clone(), c_json.clone());
            all_tasks.push(EncodedArTask {
                name: spec.occurrence_id.clone(),
                input_ids: ids,
                media: None,
                media_key: None,
                constraint: spec.grammar.clone(),
                constraint_json: c_json,
            });
        }

        let mut decisions = Map::new();
        for chunk in all_tasks.chunks(repair.batch_size) {
            check_cancel_opt(cancelled)?;
            for t in chunk {
                outcome.input_tokens += t.input_ids.len();
                outcome.task_count += 1;
            }
            let batch_record = json!({
                "input": req.state,
                "fields": Value::Object(aux_fields.clone()),
                "_ar_constraints": Value::Object(aux_constraints.clone()),
                "_cache_salt": salt,
                "_null_recovery": true,
                "_ar_max_output_tokens": repair.max_new_tokens,
                "_meta": { "extraction_contract": { "fields": {} } },
            });
            let mut batch_encoded = EncodedRequest {
                record: batch_record,
                forecasts: Vec::new(),
                tasks: chunk.to_vec(),
                typed: None,
                typed_media: None,
                eos: encoded.eos,
                ar_source: None,
                ar_sources: HashMap::new(),
                ar_provenance: None,
                ar_provenance_map: HashMap::new(),
                original_images: Vec::new(),
                default_unfocused_single: false,
                list_source_bounds: BTreeMap::new(),
                identifier_spelling_fallback: None,
                date_supplement: None,
                receipt_sum_fields: Vec::new(),
                cache_salt: salt.clone(),
            };
            let batch_preds = execute_ar_all(backend, profile, &mut batch_encoded, cancelled)?;
            for (k, v) in batch_preds {
                let toks = v.get("tokens").and_then(Value::as_u64).unwrap_or(0) as usize;
                outcome.output_tokens += toks;
                outcome.max_tokens = outcome.max_tokens.max(toks);
                decisions.insert(k, receipt::totals::build_decision_entry(&v));
            }
        }

        if let Some(primary) = predictions.get(&repair.field_name)
            && let Some(update) =
                receipt::totals::merge_receipt_repair(&repair.packet, &decisions, primary)
        {
            outcome.updates.insert(repair.field_name, update);
        }
    }

    Ok(outcome)
}

// =============================================================================
// Post-assembly & response serialization (`backend.py:1483-1645` + `api.py:170-201`)
// =============================================================================

fn assemble_geometric_rows(record: &Value, predictions: &Map<String, Value>) -> Map<String, Value> {
    let mut out = Map::new();
    let Some(anns) = record
        .get("_meta")
        .and_then(|m| m.get("extraction_contract"))
        .and_then(|c| c.get("fields"))
        .and_then(Value::as_object)
    else {
        return out;
    };
    for (name, ann) in anns {
        let Some(queries) = ann.get("geometric_row_fields").and_then(Value::as_array) else {
            continue;
        };
        if queries.is_empty() {
            continue;
        }
        let rows: Vec<&Value> = queries
            .iter()
            .filter_map(|q| q.as_str().and_then(|s| predictions.get(s)))
            .collect();
        if rows.len() == queries.len() {
            out.insert(name.clone(), crops::table_rows::assemble_rows(&rows));
        }
    }
    out
}

fn assemble_ragged_rows(record: &Value, predictions: &Map<String, Value>) -> Map<String, Value> {
    let mut out = Map::new();
    let Some(anns) = record
        .get("_meta")
        .and_then(|m| m.get("extraction_contract"))
        .and_then(|c| c.get("fields"))
        .and_then(Value::as_object)
    else {
        return out;
    };
    for (name, ann) in anns {
        let Some(pairs_arr) = ann.get("ragged_row_fields").and_then(Value::as_array) else {
            continue;
        };
        if pairs_arr.is_empty() {
            continue;
        }
        let max_items = ann
            .get("ragged_row_max_items")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(schema::MAX_LIST_ITEMS);
        let mut pairs: Vec<(&Value, &Value)> = Vec::with_capacity(pairs_arr.len());
        let mut all_ok = true;
        for entry in pairs_arr {
            let Some(pair) = entry.as_array() else {
                all_ok = false;
                break;
            };
            let (Some(gate_name), Some(copy_name)) = (
                pair.first().and_then(Value::as_str),
                pair.get(1).and_then(Value::as_str),
            ) else {
                all_ok = false;
                break;
            };
            let (Some(gate_pred), Some(copy_pred)) =
                (predictions.get(gate_name), predictions.get(copy_name))
            else {
                all_ok = false;
                break;
            };
            pairs.push((gate_pred, copy_pred));
        }
        if all_ok {
            out.insert(
                name.clone(),
                crops::ragged_rows::assemble_rows(&pairs, max_items),
            );
        }
    }
    out
}

fn assemble_financial_lines(
    record: &Value,
    predictions: &Map<String, Value>,
) -> Result<Map<String, Value>> {
    let mut out = Map::new();
    let Some(queries) = record
        .get("_financial_line_queries")
        .and_then(Value::as_array)
    else {
        return Ok(out);
    };
    if queries.is_empty() {
        return Ok(out);
    }
    let copies: Vec<&Value> = queries
        .iter()
        .filter_map(|q| q.as_str().and_then(|s| predictions.get(s)))
        .collect();
    if copies.len() != queries.len() {
        return Ok(out);
    }
    let Some(plans) = record
        .get("_financial_line_plans")
        .and_then(Value::as_object)
    else {
        return Ok(out);
    };
    for (name, plan) in plans {
        let Some(spec) = plan.get("spec") else {
            continue;
        };
        let Some(kind_str) = plan.get("kind").and_then(Value::as_str) else {
            continue;
        };
        let target_role = match kind_str {
            "tax" => crops::financial_lines::FinancialLineRole::Tax,
            "fee" => crops::financial_lines::FinancialLineRole::Fee,
            "charge" => crops::financial_lines::FinancialLineRole::Charge,
            "payment" => crops::financial_lines::FinancialLineRole::Payment,
            _ => crops::financial_lines::FinancialLineRole::Summary,
        };
        let Some(pres_name) = plan.get("presence").and_then(Value::as_str) else {
            continue;
        };
        let Some(presence) = predictions.get(pres_name) else {
            continue;
        };
        let assembled =
            crops::financial_lines::assemble_plan(spec, target_role, &copies, presence)?;
        out.insert(name.clone(), assembled);
    }
    Ok(out)
}

fn without_quantiles(answer: &Value) -> Value {
    let Some(obj) = answer.as_object() else {
        return answer.clone();
    };
    let mut out = Map::with_capacity(obj.len());
    for (k, v) in obj {
        if k == "quantiles" {
            continue;
        }
        if k == "steps"
            && let Some(arr) = v.as_array()
        {
            out.insert(
                k.clone(),
                Value::Array(arr.iter().map(without_quantiles).collect()),
            );
            continue;
        }
        out.insert(k.clone(), v.clone());
    }
    Value::Object(out)
}

/// RAII guard guaranteeing [`ClairicBackend::clear_request_caches`] runs on
/// every exit path and emitting the opt-in `[clairic-phase]` timing breakdown
/// once cache cleanup finishes.
struct CacheCleanupGuard<'a, B: ClairicBackend + ?Sized> {
    backend: &'a mut B,
}

impl<B: ClairicBackend + ?Sized> Drop for CacheCleanupGuard<'_, B> {
    fn drop(&mut self) {
        self.backend.clear_request_caches();
        self.backend.finish_request_timing();
    }
}

/// Full dual-path Clairic request entry point running on any [`ClairicBackend`]
/// (`backend.py::predict_async` + `api.py::serialize_response`).
pub fn extract_json<B: ClairicBackend + ?Sized>(
    backend: &mut B,
    raw_request: &str,
    rendered_pdf_pages: Vec<RenderedPdfPageData>,
    cancelled: Option<&Arc<AtomicBool>>,
) -> Result<String> {
    let request_start = backend.phase_timing_enabled().then(Instant::now);
    check_cancel_opt(cancelled)?;

    let parse_start = request_start.map(|_| Instant::now());
    let parsed = parse_request(raw_request, rendered_pdf_pages)?;
    let parse_ms = parse_start.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);

    if let Some(start) = request_start {
        backend.set_request_timing_start(start, parse_ms);
    }
    let guard = CacheCleanupGuard { backend };

    let profile = guard.backend.profile();

    let encode_start = request_start.map(|_| Instant::now());
    let mut encoded = encode_request(guard.backend, profile, &parsed.internal, cancelled)?;
    if let Some(t) = encode_start {
        guard
            .backend
            .set_request_encode_ms(t.elapsed().as_secs_f64() * 1000.0);
    }
    check_cancel_opt(cancelled)?;

    // Path B: Typed-head readout
    let mut predictions: Map<String, Value> = if let Some(ref typed_item) = encoded.typed {
        guard
            .backend
            .read_typed(typed_item, encoded.typed_media.as_deref(), cancelled)?
    } else {
        Map::new()
    };

    // Path A: Autoregressive grammar-constrained decoding
    let ar_predictions = if !encoded.tasks.is_empty() {
        execute_ar_all(guard.backend, profile, &mut encoded, cancelled)?
    } else {
        Map::new()
    };
    for (k, v) in &ar_predictions {
        predictions.insert(k.clone(), v.clone());
    }

    // Post-AR normalizations
    predictions = printed_temporal::normalize_predictions(&encoded.record, &predictions);
    predictions = printed_numbers::normalize_predictions(&encoded.record, &predictions);
    predictions = identifier_spelling::normalize_predictions(&encoded.record, &predictions);

    // Recovery pass 3: Identifier spelling fallback
    let mut identifier_fallback_results: Map<String, Value> = Map::new();
    let failed_spelling = identifier_spelling::failed_spelling_fields(&predictions);
    if !failed_spelling.is_empty()
        && let Some(mut fb) = encoded.identifier_spelling_fallback.take()
    {
        fb.tasks.retain(|t| failed_spelling.contains(&t.name));
        if !fb.tasks.is_empty() {
            if let Some(obj) = fb.record.as_object_mut() {
                obj.insert("_null_recovery".to_string(), Value::Bool(true));
            }
            let fb_lengths: HashMap<String, usize> = fb
                .tasks
                .iter()
                .map(|t| (t.name.clone(), t.input_ids.len()))
                .collect();
            identifier_fallback_results =
                execute_ar_all(guard.backend, profile, &mut fb, cancelled)?;
            predictions = identifier_spelling::merge_fallback(
                &predictions,
                &identifier_fallback_results,
                &fb_lengths,
            );
        }
    }

    // Recovery pass 4: Source date supplement
    let mut date_results: Map<String, Value> = Map::new();
    let mut date_task_input_tokens = 0usize;
    let mut date_task_count = 0usize;
    if let Some(mut supp) = encoded.date_supplement.take() {
        let needed: Vec<String> = supp
            .names
            .into_iter()
            .filter(|name| {
                predictions
                    .get(name)
                    .and_then(|p| p.get("value"))
                    .and_then(Value::as_str)
                    .is_some_and(|val| !supp.source.support.contains_key(val))
            })
            .collect();
        if !needed.is_empty() {
            let needed_set: HashSet<&str> = needed.iter().map(String::as_str).collect();
            supp.encoded
                .tasks
                .retain(|t| needed_set.contains(t.name.as_str()));
            if !supp.encoded.tasks.is_empty() {
                date_task_count = supp.encoded.tasks.len();
                date_task_input_tokens = supp.encoded.tasks.iter().map(|t| t.input_ids.len()).sum();
                if let Ok(raw_dates) =
                    execute_ar_all(guard.backend, profile, &mut supp.encoded, cancelled)
                {
                    let annotated = supp.source.annotate_predictions(&raw_dates, &needed);
                    predictions = supp
                        .source
                        .merge_predictions(&predictions, &annotated, &needed);
                    date_results = annotated;
                } else {
                    date_task_count = 0;
                    date_task_input_tokens = 0;
                }
            }
        }
    }

    // Recovery pass 5: Flash text-only receipt totals repair
    let receipt_outcome = if !encoded.receipt_sum_fields.is_empty() {
        match execute_receipt_repairs(
            guard.backend,
            profile,
            &parsed.internal,
            &encoded,
            &predictions,
            cancelled,
        ) {
            Ok(out) => {
                for (k, v) in &out.updates {
                    predictions.insert(k.clone(), v.clone());
                }
                out
            }
            Err(err)
                if err.is_cancelled() || err.to_string() == super::errors::CANCELLED_MESSAGE =>
            {
                return Err(err);
            }
            Err(_) => ReceiptRepairOutcome {
                updates: Map::new(),
                input_tokens: 0,
                task_count: 0,
                output_tokens: 0,
                max_tokens: 0,
            },
        }
    } else {
        ReceiptRepairOutcome {
            updates: Map::new(),
            input_tokens: 0,
            task_count: 0,
            output_tokens: 0,
            max_tokens: 0,
        }
    };

    // Structural row bounds
    if let Some(fields_obj) = encoded.record.get("fields").and_then(Value::as_object) {
        predictions = source_rows::bounded_predictions(
            &predictions,
            fields_obj,
            &encoded.list_source_bounds,
            schema::list_item_name,
        );
    }

    // Assemble all list types
    let mut merged = schema::assemble_list_predictions(&encoded.record, &predictions)?;
    for (k, v) in image_rows::assemble(&encoded.record, &predictions) {
        merged.insert(k, v);
    }
    for (k, v) in assemble_geometric_rows(&encoded.record, &predictions) {
        merged.insert(k, v);
    }
    for (k, v) in assemble_ragged_rows(&encoded.record, &predictions) {
        merged.insert(k, v);
    }
    for (k, v) in assemble_financial_lines(&encoded.record, &predictions)? {
        merged.insert(k, v);
    }
    for (k, v) in table_rows::assemble_record(&encoded.record, &predictions)? {
        merged.insert(k, v);
    }

    // Truncation & ordinary list scoring metadata (`backend.py:1503-1578`)
    let annotations = encoded
        .record
        .get("_meta")
        .and_then(|m| m.get("extraction_contract"))
        .and_then(|c| c.get("fields"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let mut truncated = encoded.tasks.iter().any(|t| {
        let has_parent = annotations
            .get(&t.name)
            .and_then(|a| a.get("parent"))
            .and_then(Value::as_str)
            .is_some_and(|p| !p.is_empty());
        !has_parent
            && !predictions
                .get(&t.name)
                .and_then(|p| p.get("valid"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
    });

    if let Some(plans) = encoded
        .record
        .get("_financial_line_plans")
        .and_then(Value::as_object)
    {
        for name in plans.keys() {
            let ok = merged
                .get(name)
                .is_some_and(|m| is_truthy(m.get("valid")) && is_truthy(m.get("complete")));
            truncated |= !ok;
        }
    }

    for (name, ann) in &annotations {
        if ann
            .get("source_fields")
            .and_then(Value::as_array)
            .is_some_and(|a| !a.is_empty())
        {
            truncated |= !merged
                .get(name)
                .and_then(|m| m.get("complete"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
        }
        if ann
            .get("image_row_fields")
            .and_then(Value::as_array)
            .is_some_and(|a| !a.is_empty())
            || ann
                .get("geometric_row_fields")
                .and_then(Value::as_array)
                .is_some_and(|a| !a.is_empty())
            || ann
                .get("ragged_row_fields")
                .and_then(Value::as_array)
                .is_some_and(|a| !a.is_empty())
        {
            let ok = merged
                .get(name)
                .is_some_and(|m| is_truthy(m.get("valid")) && is_truthy(m.get("complete")));
            truncated |= !ok;
        }
        if let Some(queries) = ann.get("query_fields").and_then(Value::as_array)
            && !queries.is_empty()
        {
            let mut used: Vec<&Value> = Vec::new();
            for q in queries {
                if let Some(child_name) = q.as_str()
                    && let Some(child_pred) = predictions.get(child_name)
                {
                    used.push(child_pred);
                    if child_pred.get("value").is_none_or(Value::is_null) {
                        break;
                    }
                }
            }
            if !used.is_empty() {
                let valid = used
                    .iter()
                    .all(|c| c.get("valid").and_then(Value::as_bool).unwrap_or(true));
                let complete = valid
                    && used.last().is_some_and(|last| {
                        last.get("value").is_none_or(Value::is_null)
                            && last
                                .get("complete")
                                .and_then(Value::as_bool)
                                .unwrap_or(true)
                    });
                if let Some(m_obj) = merged.get_mut(name).and_then(Value::as_object_mut) {
                    m_obj.insert("valid".to_string(), Value::Bool(valid));
                    m_obj.insert("complete".to_string(), Value::Bool(complete));
                    if let Some(p_null) = used[0].get("p_null").cloned() {
                        m_obj.insert("p_null".to_string(), p_null);
                    }
                    m_obj.insert(
                        "p_null_method".to_string(),
                        Value::String("first_item_null_probability".to_string()),
                    );
                    let p_nulls: Vec<Value> = used
                        .iter()
                        .filter(|c| c.get("value").is_some_and(|v| !v.is_null()))
                        .map(|c| c.get("p_null").cloned().unwrap_or(Value::Null))
                        .collect();
                    m_obj.insert("p_nulls".to_string(), Value::Array(p_nulls));

                    if used
                        .iter()
                        .all(|c| c.get("decoder").and_then(Value::as_str) == Some(ar::VERSION))
                    {
                        m_obj.insert("confidence".to_string(), json!(0.0));
                        if used
                            .iter()
                            .all(|c| c.get("log_probability").and_then(Value::as_f64).is_some())
                        {
                            let lp: f64 = used
                                .iter()
                                .filter_map(|c| c.get("log_probability").and_then(Value::as_f64))
                                .sum();
                            let conf = if complete { lp.min(0.0).exp() } else { 0.0 };
                            let structural_end = used.iter().any(|c| {
                                c.get("grounding_method").and_then(Value::as_str)
                                    == Some("source_row_bound")
                            });
                            m_obj.insert("confidence".to_string(), json!(conf));
                            m_obj.insert("log_probability".to_string(), json!(lp));
                            m_obj.insert(
                                "confidence_method".to_string(),
                                Value::String(
                                    if structural_end {
                                        "ar_sequence_probability_with_structural_end"
                                    } else {
                                        "ar_sequence_probability"
                                    }
                                    .to_string(),
                                ),
                            );
                        }
                        let spans: Vec<Value> = used
                            .iter()
                            .filter(|c| c.get("value").is_some_and(|v| !v.is_null()))
                            .map(|c| c.get("span").cloned().unwrap_or(Value::Null))
                            .collect();
                        m_obj.insert("spans".to_string(), Value::Array(spans));
                    }
                }
                truncated |= !complete;
            }
        }
    }

    // Merge forecast horizons
    if !encoded.forecasts.is_empty() {
        merged = typed_decode::merge_forecasts(&merged, &encoded.forecasts)?;
    }

    // Build public `answers` preserving request field order (`backend.py:1580-1583` + `api.py:177-201`)
    let mut answers = Map::with_capacity(parsed.internal.fields.len());
    for (name, spec) in &parsed.internal.fields {
        let mut ans = merged
            .get(name)
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        if let Some(ty) = spec.get("type").cloned() {
            ans.insert("type".to_string(), ty);
        }
        let ans_val = Value::Object(ans);
        answers.insert(
            name.clone(),
            if parsed.options.include_quantiles {
                ans_val
            } else {
                without_quantiles(&ans_val)
            },
        );
    }

    // Token usage accounting (`backend.py:1584-1645`)
    let tokenizer_arc = guard.backend.tokenizer()?;
    let tokenizer = tokenizer_arc.as_ref();
    let state_tokens = typed_prompt::tokenize_ids(tokenizer, &parsed.internal.state)?.len();

    let mut input_tokens: usize = encoded.tasks.iter().map(|t| t.input_ids.len()).sum();
    if let Some(ref item) = encoded.typed {
        input_tokens += item.input_ids.len();
    }
    for v in ar_predictions.values() {
        input_tokens += v
            .get("retry_input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
    }
    for v in predictions.values() {
        if is_truthy(v.get("identifier_spelling_fallback")) {
            let total_retry = v
                .get("retry_input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            let prev_retry = v
                .get("identifier_spelling_attempt")
                .and_then(|a| a.get("retry_input_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            input_tokens += total_retry.saturating_sub(prev_retry);
        }
    }
    input_tokens += date_task_input_tokens;
    input_tokens += receipt_outcome.input_tokens;

    let unexecuted_map = encoded
        .record
        .get("_unexecuted_list_items")
        .and_then(Value::as_object);
    let unexecuted_tokens: usize = unexecuted_map
        .map(|m| {
            m.values()
                .filter_map(Value::as_u64)
                .map(|v| v as usize)
                .sum()
        })
        .unwrap_or(0);
    let unexecuted_count = unexecuted_map.map_or(0, Map::len);
    input_tokens = input_tokens.saturating_sub(unexecuted_tokens);

    let input_work = encoded
        .tasks
        .len()
        .saturating_add(usize::from(encoded.typed.is_some()))
        .saturating_sub(unexecuted_count)
        .saturating_add(identifier_fallback_results.len())
        .saturating_add(date_task_count)
        .saturating_add(receipt_outcome.task_count);

    let image_token_id = guard.backend.image_token_id();
    let has_media_payload = is_truthy(encoded.record.get("_media_payload"));
    let image_tokens = if has_media_payload {
        if let Some(first_task) = encoded.tasks.first() {
            first_task
                .input_ids
                .iter()
                .filter(|&&tok| tok == image_token_id)
                .count()
        } else if let Some(ref item) = encoded.typed {
            item.input_ids
                .iter()
                .filter(|&&tok| tok == image_token_id)
                .count()
        } else {
            0
        }
    } else {
        0
    };

    let schema_tokens = input_tokens.saturating_sub(
        state_tokens
            .saturating_add(image_tokens)
            .saturating_mul(input_work),
    );

    let output_tokens: usize = ar_predictions
        .values()
        .map(|v| v.get("tokens").and_then(Value::as_u64).unwrap_or(0) as usize)
        .sum::<usize>()
        + identifier_fallback_results
            .values()
            .map(|v| v.get("tokens").and_then(Value::as_u64).unwrap_or(0) as usize)
            .sum::<usize>()
        + date_results
            .values()
            .map(|v| v.get("tokens").and_then(Value::as_u64).unwrap_or(0) as usize)
            .sum::<usize>()
        + receipt_outcome.output_tokens;

    let mut passes = 1usize;
    for v in ar_predictions.values() {
        passes = passes.max(v.get("tokens").and_then(Value::as_u64).unwrap_or(0) as usize);
    }
    for name in identifier_fallback_results.keys() {
        if let Some(p) = predictions.get(name) {
            passes = passes.max(p.get("tokens").and_then(Value::as_u64).unwrap_or(0) as usize);
        }
    }
    for v in date_results.values() {
        passes = passes.max(v.get("tokens").and_then(Value::as_u64).unwrap_or(0) as usize);
    }
    passes = passes.max(receipt_outcome.max_tokens);

    let mut response = Map::new();
    response.insert(
        "model".to_string(),
        Value::String(profile.public_model_id().to_string()),
    );
    response.insert("answers".to_string(), Value::Object(answers));
    response.insert("truncated".to_string(), Value::Bool(truncated));
    response.insert(
        "usage".to_string(),
        json!({
            "state_tokens": state_tokens,
            "image_tokens": image_tokens,
            "schema_tokens": schema_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "passes": passes,
        }),
    );
    if parsed.options.return_state_text {
        response.insert(
            "state_text".to_string(),
            Value::String(parsed.internal.state),
        );
    }

    let _ = settings::PUBLIC_MODEL_ID;
    serde_json::to_string(&Value::Object(response))
        .map_err(|e| inference_error(format!("failed to serialize Clairic response: {e}")))
}

// =============================================================================
// Pure-Rust orchestrator unit tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{CANCELLED_MESSAGE, INVALID_REQUEST_PREFIX};

    #[test]
    fn rejects_excessive_single_list_expansion_and_cancels_cleanly() {
        // Direct `expand_request` guard on single-list budget > MAX_LIST_EXPANSION_BUDGET
        // (e.g., unhinted list over >512 lexical units).
        let many_words = (0..600)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let single_over_budget = InternalRequest {
            state: many_words,
            image_count: 0,
            pdf_page_count: 0,
            decoded_images: Vec::new(),
            source_media_key: None,
            fields: json!({
                "items": { "type": "list", "items": "text" }
            })
            .as_object()
            .unwrap()
            .clone(),
            reference_date: None,
        };
        let err = expand_request(&single_over_budget, ClairicProfile::Flash, None).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.starts_with(INVALID_REQUEST_PREFIX),
            "expected invalid request for >MAX_LIST_EXPANSION_BUDGET, got: {msg}"
        );

        // Cancellation inside list expansion aborts cleanly.
        let cancelled = Arc::new(AtomicBool::new(true));
        let cancel_req = InternalRequest {
            state: "a b c".to_string(),
            image_count: 0,
            pdf_page_count: 0,
            decoded_images: Vec::new(),
            source_media_key: None,
            fields: json!({
                "items": { "type": "list", "items": "text", "max_items": 4 }
            })
            .as_object()
            .unwrap()
            .clone(),
            reference_date: None,
        };
        let err = expand_request(&cancel_req, ClairicProfile::Flash, Some(&cancelled)).unwrap_err();
        assert_eq!(err.to_string(), CANCELLED_MESSAGE);
        assert!(err.is_cancelled());
    }
}
