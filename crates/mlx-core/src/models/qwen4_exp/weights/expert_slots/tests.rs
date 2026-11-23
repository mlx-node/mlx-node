use super::*;

#[test]
fn invalid_device_route_mapping_does_not_commit_earlier_layer_ages() {
    let mut store = Store::open_metadata(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
        None,
    )
    .unwrap();
    let weight = Arc::new(Weight {
        values: MxArray::zeros(&[1, 1], Some(DType::Float32)).unwrap(),
        dense_bf16: None,
        scales: None,
        biases: None,
        group: 0,
        bits: 0,
        mode: String::new(),
        q8_symmetric_128: false,
        qlinear: None,
    });
    for (layer, mapping) in [vec![Some(0)], vec![None, Some(2)]].into_iter().enumerate() {
        store.slots.insert(
            layer,
            ExpertSlots {
                banks: [weight.clone(), weight.clone(), weight.clone()],
                direct: false,
                names: ["gate".into(), "up".into(), "down".into()],
                rows: [1; 3],
                occupants: vec![Some(layer as u32)],
                mapping,
                device_mapping: None,
                ages: vec![0],
                tick: 0,
                readers: Vec::new(),
            },
        );
    }
    assert!(!store.commit_device_routes(&[0, 1], &[0, 1], 1).unwrap());
    assert_eq!(store.slot_hits, 0);
    for slots in store.slots.values() {
        assert_eq!(slots.tick, 0);
        assert_eq!(slots.ages, [0]);
    }
}
#[test]
fn completed_expert_window_releases_large_reader_storage() {
    let mut store = Store::open_metadata(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
        None,
    )
    .unwrap();
    let bank = Arc::new(Weight {
        dense_bf16: None,
        values: MxArray::zeros(&[1, 1], Some(DType::Float32)).unwrap(),
        scales: None,
        biases: None,
        group: 0,
        bits: 0,
        mode: String::new(),
        q8_symmetric_128: false,
        qlinear: None,
    });
    store.slots.insert(
        0,
        ExpertSlots {
            banks: [bank.clone(), bank.clone(), bank],
            direct: false,
            names: Default::default(),
            rows: [1; 3],
            occupants: vec![Some(0)],
            mapping: vec![Some(0)],
            device_mapping: None,
            ages: vec![0],
            tick: 0,
            readers: Vec::new(),
        },
    );
    crate::array::memory::synchronize();
    let before = crate::array::memory::get_active_memory();
    let reduced = {
        let large = MxArray::ones(&[1024, 1024], Some(DType::Float32)).unwrap();
        let reduced = large.sum(Some(&[0]), Some(false)).unwrap();
        store.finish_expert_slots(0, &large).unwrap();
        MxArray::eval_arrays(&[&large]).unwrap();
        assert!(unsafe { mlx_sys::mlx_array_is_available(large.as_raw_ptr()) });
        store.complete_expert_window(0, &reduced).unwrap();
        reduced
    };
    // Sibling tests allocate and free on the same process-wide counter
    // (a residency test holds fixture banks for seconds); what this
    // window retained is what is still there once they are done.
    let mut retained = f64::MAX;
    for _ in 0..1000 {
        crate::array::memory::synchronize();
        retained = crate::array::memory::get_active_memory() - before;
        if retained < 256. * 1024. {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        retained < 256. * 1024.,
        "completed window retained {retained} bytes"
    );
    assert!(store.slots[&0].readers.is_empty());
    assert!(reduced.to_float32().unwrap().iter().all(|&x| x == 1024.));
    let previous = store.slots[&0].banks.clone();
    store.slots.insert(
        1,
        ExpertSlots {
            banks: previous,
            direct: false,
            names: Default::default(),
            rows: [1; 3],
            occupants: vec![Some(0)],
            mapping: vec![Some(0)],
            device_mapping: None,
            ages: vec![0],
            tick: 0,
            readers: Vec::new(),
        },
    );
    store.defer_expert_reduction(0, &reduced).unwrap();
    let independent = reduced.add_scalar(1.).unwrap();
    store.defer_expert_reduction(1, &independent).unwrap();
    assert!(store.slots[&0].readers.is_empty());
    assert_eq!(store.slots[&1].readers.len(), 1);
    assert_eq!(store.deferred_reduction_layer, Some(1));
    // Join independent leases from both layers, including an unevaluated
    // reader that does not depend on the newest token's output.
    let old = store.slots[&0].banks[0].values.add_scalar(7.).unwrap();
    store.finish_expert_slots(0, &old).unwrap();
    store.complete_device_readers().unwrap();
    assert!(store.slots.values().all(|s| s.readers.is_empty()));
    assert_eq!(store.deferred_reduction_layer, None);
    assert_eq!(&*old.to_float32().unwrap(), &[7.]);
    assert!(
        independent
            .to_float32()
            .unwrap()
            .iter()
            .all(|&x| x == 1025.)
    );
}
#[test]
fn slot_budget_uses_attached_auxiliary_inventory_without_reading_weights() {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return;
    }
    let mut store = Store::open_metadata(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
        None,
    )
    .unwrap();
    store.tensors.clear();
    store.gguf = true;
    let mut routed = 0u64;
    for layer in 0..48 {
        for part in ["gate", "up", "down"] {
            let t = Tensor {
                path: PathBuf::new(),
                shape: if part == "down" {
                    vec![512, 2560, 640]
                } else {
                    vec![512, 640, 2560]
                },
                offset: 0,
                bytes: 0,
                encoding: Encoding::Gguf(if part == "down" {
                    GgufTensorType::Q5_1
                } else {
                    GgufTensorType::Q4K
                }),
            };
            routed += t.runtime_bytes().unwrap();
            store
                .tensors
                .insert(format!("blk.{layer}.ffn_{part}_exps.weight"), t);
        }
    }
    let fixed = 4 << 30;
    store.plan.hot_bytes = routed + fixed;
    let mut previous = 0;
    for gib in [16, 32, 48] {
        store.cache_limit = gib << 30;
        store.slot_capacity = None;
        store.tensors.remove("mtp.test.weight");
        store.tensors.remove("nextn.0.ffn_gate_exps.weight");
        let text = store.expert_slot_capacity(512, 10).unwrap();
        assert!(text > previous && text < 512);
        previous = text;
        assert!(routed * text as u64 / 512 + fixed + MAX_READ_BYTES * 6 <= store.cache_limit);
        assert!(
            (routed + routed / 48) * text as u64 / 512 + fixed + MAX_READ_BYTES * 6
                <= store.cache_limit,
            "largest-layer staging must also fit the admitted weight budget"
        );
        store.tensors.insert(
            "mtp.test.weight".into(),
            Tensor {
                path: PathBuf::new(),
                shape: vec![65536, 65536],
                offset: 0,
                bytes: 8 << 30,
                encoding: Encoding::Safe("BF16".into()),
            },
        );
        store.slot_capacity = None;
        let auxiliary = store.expert_slot_capacity(512, 10).unwrap();
        assert!(
            auxiliary < text,
            "attached auxiliary weights need their own cache allowance"
        );
        assert!(
            routed * auxiliary as u64 / 512 + fixed + (store.cache_limit / 8) <= store.cache_limit
        );
        let extra = store.tensors["mtp.test.weight"].clone();
        store
            .tensors
            .insert("nextn.0.ffn_gate_exps.weight".into(), extra);
        store.slot_capacity = None;
        assert_eq!(
            store.expert_slot_capacity(512, 10).unwrap(),
            auxiliary,
            "auxiliary experts must not be counted as main-network banks"
        );
    }
    assert_eq!(store.bytes_read, 0);
    assert_eq!(store.slot_bytes, 0);
}
const DRAFT_HEADS: [(&str, i64, i64); 2] = [
    ("mtp.layers.0.mlp.experts.gate_up_proj", 512 * 1280, 2560),
    ("mtp.layers.0.mlp.experts.down_proj", 512 * 2560, 640),
];
const FIXED_BANK_BYTES: u64 = 5_245_160_960;
const TEXT_ONLY_SLOTS_6_TO_70_GIB: [usize; 65] = [
    0, 11, 18, 25, 32, 39, 45, 52, 59, 66, 73, 80, 86, 93, 100, 107, 114, 120, 127, 134, 141, 148,
    154, 161, 168, 175, 182, 189, 195, 202, 209, 216, 223, 229, 236, 243, 250, 257, 264, 270, 277,
    284, 291, 298, 304, 311, 318, 325, 332, 339, 345, 352, 359, 366, 373, 379, 386, 393, 400, 407,
    413, 420, 427, 434, 441,
];

/// The released UD-Q4_K_XL checkpoint's routed experts (or one GGUF type
/// for all of them), its fixed bank bytes and the MTP head, on the slot
/// path, with each layer's routed bytes.
fn draft_bank_store(uniform: Option<GgufTensorType>) -> Option<(Store, Vec<u64>)> {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return None;
    }
    let mut store = Store::open_metadata(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
        None,
    )
    .unwrap();
    store.tensors.clear();
    store.gguf = true;
    store
        .metadata
        .insert("qwen4exp.expert_count".into(), GgufMetaValue::Uint32(512));
    let mut layers = Vec::new();
    for layer in 0..48 {
        let (gate_up, down) = match (uniform, layer) {
            (Some(t), _) => (t, t),
            (None, 2) => (GgufTensorType::Q5K, GgufTensorType::Q8_0),
            (None, 4 | 30 | 46 | 47) => (GgufTensorType::Q4K, GgufTensorType::Q8_0),
            (None, _) => (GgufTensorType::Q4K, GgufTensorType::Q5_1),
        };
        let mut bytes = 0;
        for part in ["gate", "up", "down"] {
            let t = Tensor {
                path: PathBuf::new(),
                shape: if part == "down" {
                    vec![512, 2560, 640]
                } else {
                    vec![512, 640, 2560]
                },
                offset: 0,
                bytes: 0,
                encoding: Encoding::Gguf(if part == "down" { down } else { gate_up }),
            };
            bytes += t.runtime_bytes().unwrap();
            store
                .tensors
                .insert(format!("blk.{layer}.ffn_{part}_exps.weight"), t);
        }
        layers.push(bytes);
    }
    store.plan.hot_bytes = layers.iter().sum::<u64>() + FIXED_BANK_BYTES;
    store.draft_bank_budget = 0;
    for (name, rows, width) in DRAFT_HEADS {
        store.tensors.insert(
            name.into(),
            Tensor {
                path: PathBuf::new(),
                shape: vec![512, rows as usize / 512, width as usize],
                offset: 0,
                bytes: rows as u64 * width as u64 * 2,
                encoding: Encoding::Safe("BF16".into()),
            },
        );
    }
    Some((store, layers))
}

/// The draft banks in the layout the admission takes, never evaluated.
fn install_draft_banks(store: &mut Store) -> u64 {
    let specs = store
        .mtp_bank_admission()
        .unwrap()
        .expect("the admission takes the draft banks");
    for (name, _, rows, width, sides, _) in specs {
        let (rows, width, sides) = (rows as i64, width as i64, sides.unwrap());
        let lazy = |columns: i64, dtype| MxArray::zeros(&[rows, columns], Some(dtype)).unwrap();
        store.banks.insert(
            name.into(),
            Arc::new(Weight {
                values: lazy(width / 4, DType::Uint32),
                dense_bf16: None,
                scales: Some(lazy(width / 32, sides)),
                biases: Some(lazy(width / 32, sides)),
                group: 32,
                bits: 8,
                mode: "affine".into(),
                q8_symmetric_128: false,
                qlinear: None,
            }),
        );
    }
    let draft = store.mtp_bank_bytes().unwrap();
    store.bank_bytes += draft;
    store.cache_bytes += draft;
    draft
}

/// One pass's slot-bank builds at `capacity`: the eviction `expert_slots`
/// runs before each layer's bank, with no LRU row to give way.
fn build_slot_banks(store: &mut Store, layers: &[u64], capacity: usize) -> Result<()> {
    for &layer in layers {
        let bytes = layer * capacity as u64 / 512;
        store.evict_to(store.cache_limit.saturating_sub(bytes + MAX_READ_BYTES * 3))?;
        store.slot_bytes += bytes;
        store.cache_bytes += bytes;
    }
    Ok(())
}

/// A direct MTP load installs the Q8 draft banks; partial slots then fit
/// beside them while they stay pinned. Every layer's
/// slot bank is built with only LRU rows evictable, so the last one needs
/// the banks, all slots and its staging inside the budget.
#[test]
fn fallback_slots_fit_beside_installed_draft_banks() {
    let Some((mut store, layers)) = draft_bank_store(None) else {
        return;
    };
    let routed: u64 = layers.iter().sum();
    assert_eq!(routed, 78_538_342_400);
    let fixed = FIXED_BANK_BYTES;
    store.cache_limit = 36 << 30;
    let draft = store.mtp_bank_estimate().unwrap();
    assert_eq!(draft, 3_040_870_400, "the f32-sided gate_up bank");
    let slots = |store: &mut Store, gib: u64| {
        store.cache_limit = gib << 30;
        store.slot_capacity = None;
        store.expert_slot_capacity(512, 10).unwrap()
    };
    let budgets = [10u64, 16, 20, 26, 30, 36];
    let text = budgets.map(|gib| slots(&mut store, gib));
    assert_eq!(
        [text[1], text[3]],
        [63, 125],
        "no installed draft bank: the slot count is unchanged"
    );
    store.bank_bytes = fixed;
    store.cache_bytes = fixed;
    assert_eq!(install_draft_banks(&mut store), draft);
    let banked = budgets.map(|gib| slots(&mut store, gib));
    let table = format!("slots at {budgets:?} GiB: {banked:?} with the banks, {text:?} without");
    for (k, gib) in budgets.into_iter().enumerate() {
        assert!(
            store.bank_bytes + routed * banked[k] as u64 / 512 + MAX_READ_BYTES * 3 <= gib << 30,
            "the slots overrun {gib} GiB beside the draft banks; {table}"
        );
        if gib >= 26 {
            assert_eq!(banked[k], text[k], "{table}");
        }
    }
    assert_eq!(store.bytes_read, 0);
    assert_eq!(store.slot_bytes, 0);
}

/// `prepare_hot` caches the slot count before the load admits the draft
/// banks, and the first turn's pass builds every layer's slot bank beside
/// them. Every 1/16 GiB from 6 to 70 GiB, on an all-Q5_1 geometry
/// and the released one: no layer runs out of staging, and the banks are
/// still admitted at every whole budget from 22 GiB.
#[test]
fn draft_banks_admitted_after_the_cached_slot_count_leave_every_layer_staging() {
    for (uniform, at_22) in [(Some(GgufTensorType::Q5_1), 85), (None, 98)] {
        let Some((mut store, layers)) = draft_bank_store(uniform) else {
            return;
        };
        let mut admitted = Vec::new();
        for sixteenths in 6 * 16..=70 * 16u64 {
            let limit = sixteenths << 26;
            store.banks.retain(|name, _| !name.starts_with("mtp."));
            store.bank_bytes = FIXED_BANK_BYTES;
            store.cache_bytes = FIXED_BANK_BYTES;
            store.slot_bytes = 0;
            store.slot_capacity = None;
            store.mtp_banks_declined = false;
            store.cache_limit = limit;
            let capacity = store.expert_slot_capacity(512, 10).unwrap();
            let fits = store.mtp_bank_admission().unwrap().is_some();
            if fits {
                install_draft_banks(&mut store);
            }
            let gib = limit as f64 / (1u64 << 30) as f64;
            if capacity > 0 {
                build_slot_banks(&mut store, &layers, capacity).unwrap_or_else(|e| {
                    panic!("{uniform:?} at {gib} GiB, {capacity} slots, banks {fits}: {e}")
                });
            }
            assert!(
                store.cache_bytes + MAX_READ_BYTES * 3 <= limit,
                "{uniform:?} at {gib} GiB"
            );
            if limit == 22 << 30 {
                assert_eq!(
                    (capacity, fits),
                    (at_22, true),
                    "{uniform:?} at 22 GiB: (slots, draft banks admitted)"
                );
            }
            if fits && limit % (1 << 30) == 0 {
                admitted.push(limit >> 30);
            }
        }
        assert_eq!(admitted, (22..=70).collect::<Vec<_>>(), "{uniform:?}");
        assert_eq!(store.bytes_read, 0);
    }
}

/// The draft reserve costs slots only at budgets whose admission will take
/// the banks; a decline gives them back, and a text-only load sizes
/// exactly as before.
#[test]
fn slot_count_reserves_only_draft_banks_the_admission_takes() {
    let Some((mut store, _)) = draft_bank_store(None) else {
        return;
    };
    let slots = |store: &mut Store, gib: u64| {
        store.cache_limit = gib << 30;
        store.slot_capacity = None;
        store.expert_slot_capacity(512, 10).unwrap()
    };
    let mut text_only = draft_bank_store(None).unwrap().0;
    text_only
        .tensors
        .retain(|name, _| !name.starts_with("mtp."));
    assert_eq!(text_only.mtp_bank_estimate().unwrap(), 0);
    let text: Vec<usize> = (6..=70).map(|gib| slots(&mut text_only, gib)).collect();
    assert_eq!(text, TEXT_ONLY_SLOTS_6_TO_70_GIB, "no MTP head");
    let budgets = [21u64, 22, 23, 24, 26, 30];
    assert_eq!(
        budgets.map(|gib| slots(&mut store, gib)),
        [94, 98, 104, 111, 125, 149],
        "MTP head attached"
    );
    store.mtp_banks_declined = true;
    assert_eq!(
        budgets.map(|gib| slots(&mut store, gib)),
        [94, 100, 106, 112, 125, 149],
        "draft banks declined"
    );
}

/// The store plan's cap, less the slots' staging, bounds partial banks on
/// the released 48-layer geometry beside the fixed and draft banks and the
/// row share. No bank payload is allocated by this check.
#[test]
fn gate_sized_partial_capacity_tracks_the_full_bank_shortfall() {
    let Some((mut store, layers)) = draft_bank_store(None) else {
        return;
    };
    let routed: u64 = layers.iter().sum();
    assert_eq!(routed, 78_538_342_400);
    let largest = store.largest_routed_layer_bytes().unwrap();
    let rows = super::super::super::load_plan::row_cache_share(
        &super::super::super::memory::flash_next_config(),
        MAX_READ_BYTES * 6,
    )
    .all;
    let anonymous = FIXED_BANK_BYTES + 3_040_870_400 + rows;
    let mut capacities = Vec::new();
    for cap in [routed - 1, 72 << 30, 40 << 30] {
        let bank_budget = slot_bank_budget_for(cap, routed, largest);
        store.cache_limit = anonymous.saturating_add(cap);
        store.plan.budget = store.cache_limit;
        store.slot_bank_budget = Some(bank_budget);
        store.slot_capacity = None;
        store.mtp_banks_declined = false;
        capacities.push(store.expert_slot_capacity(512, 10).unwrap());
        let planned = store.planned_slot_bytes().unwrap();
        assert!(planned <= bank_budget, "cap {cap}");
        assert!(
            planned > bank_budget.saturating_sub(routed / 512),
            "cap {cap}"
        );
        assert_eq!(store.plan.budget, anonymous + cap);
    }
    assert_eq!(capacities, [498, 490, 272]);
    assert_eq!(store.bytes_read, 0);
    assert_eq!(store.slot_bytes, 0);
}

#[test]
fn partial_gate_released_geometry_preserves_execution_reserves() {
    use super::super::super::memory;

    let Some((mut store, layers)) = draft_bank_store(None) else {
        return;
    };
    store.tensors.retain(|name, _| !name.starts_with("mtp."));
    let routed = layers.iter().sum::<u64>();
    assert_eq!(routed, 78_538_342_400);
    let host_at = |available: u64| memory::HostMemory {
        physical: 128 << 30,
        free: available,
        file: 0,
        anonymous: (128u64 << 30) - available - (20 << 30),
        purgeable: 0,
        wired: 13 << 30,
        compressor: 7 << 30,
    };
    let host = host_at(84 << 30);
    assert_eq!(host.available(), 84 << 30);
    assert_eq!(host.non_reclaimable(), 44 << 30);
    let pool = 1365u64 * 786_432;
    let rows = super::super::super::load_plan::row_cache_share(
        &memory::flash_next_config(),
        MAX_READ_BYTES * 6,
    )
    .all;
    let anonymous = FIXED_BANK_BYTES + rows;
    assert_eq!(anonymous, 6_989_991_424);
    let gate = memory::bank_gate(&host, anonymous + pool, memory::DIRECT_MARGIN);
    assert_eq!(gate.bank_budget, 77_835_874_816);
    assert!(gate.bank_budget < routed);
    let budget = anonymous + gate.bank_budget;
    assert_eq!(budget, 84_825_866_240);

    store.cache_limit = budget;
    store.plan.budget = budget;
    store.plan.available_bytes = Some(host.available());
    store.plan.physical_bytes = Some(host.physical);
    store.plan.bank_budget = gate.bank_budget;
    store.plan.live_non_reclaimable = gate.non_reclaimable;
    store.plan.partial_direct_gate = true;
    store.slot_bank_budget = Some(gate.bank_budget);
    store.slot_capacity = None;
    store.mtp_banks_declined = false;
    let capacity = store.expert_slot_capacity(512, 10).unwrap();
    assert_eq!(capacity, 502);
    store.plan.partial_expert_slots = capacity;
    let planned = store.planned_slot_bytes().unwrap();
    assert_eq!(planned, 77_004_390_400);
    assert!(!store.plan.direct);
    assert!(store.plan.use_direct_turn_floor());

    let c = memory::flash_next_config();
    let scratch = memory::prefill_scratch(&c, 1024, 32_768);
    assert_eq!(scratch, 1_988_100_096);
    let mut ordinary = store.plan.clone();
    ordinary.partial_direct_gate = false;
    assert!(!ordinary.use_direct_turn_floor());

    let post_available = host.available() - FIXED_BANK_BYTES - planned;
    assert_eq!(post_available, 7_944_761_856);
    let post = host_at(post_available);
    memory::check_growth_headroom_with(&post, store.plan.use_direct_turn_floor()).unwrap();
    let after_scratch = host_at(post_available - scratch);
    memory::check_growth_headroom_with(&after_scratch, store.plan.use_direct_turn_floor()).unwrap();
    let below_direct_floor = host_at(memory::DIRECT_TURN_FLOOR - 1);
    assert!(
        memory::check_growth_headroom_with(&below_direct_floor, store.plan.use_direct_turn_floor())
            .is_err()
    );
    assert!(memory::check_growth_headroom_with(&post, ordinary.use_direct_turn_floor()).is_err());
    memory::check_growth_headroom_with(&host_at(16 << 30), ordinary.use_direct_turn_floor())
        .unwrap();
    assert!(
        memory::check_growth_headroom_with(
            &host_at((16 << 30) - 1),
            ordinary.use_direct_turn_floor()
        )
        .is_err()
    );
    assert_eq!(store.bytes_read, 0);
    assert_eq!(store.slot_bytes, 0);
}

/// A 128 GiB Mac (a 115,448,725,504-byte Metal working set, nothing
/// else loaded) on the released text geometry, through the load's own
/// steps: the store plan, the host gate, the engine plan and the plan the
/// decoder adopts. A quiet host admits every expert bank; a busy one
/// (72 GiB it cannot reclaim) refuses them and the store keeps about 300
/// of 512 slots under the plan's cap. Both prefill 1,024 rows up to the
/// 65,536-token allowance the gate priced; past it only the refused
/// store halves its passes, to fit the scratch that gate reserved.
/// MUTATION: `adopt_plan` keeps no reservation on a refused gate → red
/// (1,024-row passes at 200,000 tokens on the busy host).
#[test]
fn busy_host_prefills_the_planned_chunk_on_a_partial_store() {
    use super::super::super::{Qwen4ExpContextLimits, load_plan, memory};
    use crate::engine::memory_plan::LiveMemory;
    if !gpu_repack_available() {
        return;
    }
    let Some((mut store, layers)) = draft_bank_store(None) else {
        return;
    };
    store.tensors.retain(|name, _| !name.starts_with("mtp."));
    let routed: u64 = layers.iter().sum();
    let shard = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/qwen4-exp/config.json")
        .canonicalize()
        .unwrap();
    for tensor in store.tensors.values_mut() {
        tensor.path = shard.clone();
    }
    store.gguf_source = Some(DirectGgufSource::new(std::slice::from_ref(&shard)).unwrap());
    store.direct_gguf = true;
    store.metadata.insert(
        "qwen4exp.expert_used_count".into(),
        GgufMetaValue::Uint32(10),
    );
    const WORKING_SET: u64 = 115_448_725_504;
    let gib = |n: f64| (n * (1u64 << 30) as f64) as u64;
    let quiet = memory::HostMemory {
        physical: 128 << 30,
        free: gib(77.3),
        file: gib(10.0),
        anonymous: gib(20.1),
        purgeable: gib(0.7),
        wired: gib(13.2),
        compressor: gib(7.5),
    };
    let busy = memory::HostMemory {
        physical: 128 << 30,
        free: 30 << 30,
        file: 25 << 30,
        anonymous: 51 << 30,
        purgeable: 0,
        wired: 13 << 30,
        compressor: 8 << 30,
    };
    let c = load_plan::tests::int8_config();
    let inventory = load_plan::LoadInventory {
        store_fixed_bytes: FIXED_BANK_BYTES,
        ..store.load_inventory(&c, false).unwrap()
    };
    assert_eq!(inventory.experts.all, routed);
    let live = |active: u64| LiveMemory {
        working_set_bytes: WORKING_SET,
        active_bytes: active,
        registered_pool_bytes: 0,
    };
    let decision = load_plan::plan_store(
        "Qwen4Exp",
        &c,
        &inventory,
        live(0),
        None,
        None,
        false,
        false,
        load_plan::DraftPrice::default(),
    )
    .unwrap();
    assert!(decision.store.whole, "the plan holds every expert");
    store.set_store_plan(decision.store, inventory.store_fixed_bytes);
    store.plan.physical_bytes = Some(128 << 30);
    let need = inventory.store_fixed_bytes + decision.store.row_cache_bytes + decision.allowance;
    let engine = |active: u64, held_back: u64| {
        load_plan::plan_engine(
            "Qwen4Exp",
            &c,
            live(active),
            held_back,
            None,
            false,
            false,
            load_plan::DraftPrice::default(),
            decision.states,
        )
    };
    let adopted = |plan: &memory::Plan,
                   live: &Arc<Live>,
                   engine_plan: &crate::engine::memory_plan::MemoryPlan| {
        let limits =
            Qwen4ExpContextLimits::from_plan(&c, engine_plan, c.served_context_tokens()).unwrap();
        let (mut decoder, _) = super::super::super::tests::tiny_decoder();
        decoder.config = Arc::new(c.clone());
        decoder.weights.plan = plan.clone();
        decoder.weights.live = Arc::clone(live);
        decoder.adopt_plan(
            limits.effective_window_tokens as usize,
            decision.allowance_tokens,
        );
        assert_eq!(
            decoder
                .weights
                .live
                .prefill_chunk_tokens
                .load(Ordering::Acquire),
            decoder.prefill_chunk_size as u64
        );
        let passes = [0, 64_000, 200_000].map(|position| {
            decoder.history = vec![0; position];
            decoder.prefill_pass_window()
        });
        (
            decoder.prefill_chunk_size,
            passes,
            limits.effective_window_tokens,
        )
    };

    let mut direct = store.plan.clone();
    {
        let _quiet = memory::scoped_test_host(quiet);
        memory::direct_gate(&mut direct, need, routed).unwrap();
    }
    direct.direct = true;
    direct.direct_bytes = routed;
    let direct_engine = engine(FIXED_BANK_BYTES + routed, decision.store.row_cache_bytes).unwrap();
    let (direct_window, direct_passes, direct_context) =
        adopted(&direct, &Arc::default(), &direct_engine);

    {
        let _busy = memory::scoped_test_host(busy);
        assert!(
            store
                .select_direct_residency_with(512, need, &mut memory::direct_gate, &mut |_| panic!(
                    "a refused gate must not allocate"
                ),)
                .unwrap()
        );
    }
    assert!(!store.plan.direct && store.plan.partial_direct_gate);
    let slots = store.plan.partial_expert_slots;
    assert!((250..=350).contains(&slots), "{slots} slots");
    let held_back = store.held_back_bytes();
    let partial_engine = load_plan::settle(
        &|held_back| engine(0, held_back),
        held_back,
        decision.allowance,
        &mut |bytes| store.lower_store_cap(bytes),
    )
    .unwrap();
    let (partial_window, partial_passes, partial_context) =
        adopted(&store.plan, &store.live, &partial_engine);

    let scratch = |rows, context| memory::prefill_scratch(&c, rows, context);
    eprintln!(
        "busy host: hard {} B; store plan {:?}; allowance {} B; gate need {} B; routed {} B; \
         quiet bank budget {} B; busy bank budget {} B, {} slots, planned slot bytes {} B, \
         weight budget {} B, available at gate {:?} B; direct engine dynamic {} B, window {}, \
         transient {:?} B, pool max blocks {:?}; partial engine dynamic {} B, window {}, \
         transient {:?} B, pool max blocks {:?}, passes at 0 / 64,000 / 200,000: {:?}; scratch 1024 rows at 65,536 / 262,144: {} / {} B; \
         2048 rows: {} / {} B",
        partial_engine.hard_budget_bytes,
        decision.store,
        decision.allowance,
        need,
        routed,
        direct.bank_budget,
        store.plan.bank_budget,
        slots,
        store.planned_slot_bytes().unwrap(),
        store.plan.budget,
        store.plan.available_bytes,
        direct_engine.dynamic_budget_bytes,
        direct_context,
        direct_engine.pool.map(|pool| pool.lane.transient_bytes),
        direct_engine.pool.map(|pool| pool.max_blocks),
        partial_engine.dynamic_budget_bytes,
        partial_context,
        partial_engine.pool.map(|pool| pool.lane.transient_bytes),
        partial_engine.pool.map(|pool| pool.max_blocks),
        partial_passes,
        scratch(1024, 65_536),
        scratch(1024, 262_144),
        scratch(2048, 65_536),
        scratch(2048, 262_144),
    );
    assert_eq!(direct_window, 1024);
    assert_eq!(partial_window, 1024);
    assert_eq!(direct_passes, [1024; 3]);
    assert_eq!(partial_passes[..2], [1024, 1024]);
    let long = partial_passes[2];
    assert!(long < 1024, "{partial_passes:?}");
    assert!(scratch(long, 200_000 + 1024) <= scratch(1024, 65_536));
    assert_eq!(store.slot_bytes, 0);
    assert_eq!(store.bytes_read, 0);
}

/// A slot count cached with no room for the draft banks beside it: the
/// admission declines them before reading a byte, the decline sticks, and
/// every layer still builds. Half the layers already built: the same.
#[test]
fn draft_bank_admission_declines_beside_a_cached_count_without_staging() {
    let Some((mut store, layers)) = draft_bank_store(Some(GgufTensorType::Q5_1)) else {
        return;
    };
    let routed: u64 = layers.iter().sum();
    let draft = store.mtp_bank_estimate().unwrap();
    store.cache_limit = 22 << 30;
    assert!(draft <= store.auxiliary_cache_allowance().unwrap());
    // The count sized without the draft reserve.
    let capacity = 87;
    assert!(
        FIXED_BANK_BYTES + routed * capacity as u64 / 512 + draft + MAX_READ_BYTES * 3
            > store.cache_limit
    );
    for built in [0, 24] {
        store.bank_bytes = FIXED_BANK_BYTES;
        store.cache_bytes = FIXED_BANK_BYTES;
        store.slot_bytes = 0;
        store.slot_capacity = Some(capacity);
        store.mtp_banks_declined = false;
        build_slot_banks(&mut store, &layers[..built], capacity).unwrap();
        assert!(
            store.mtp_bank_admission().unwrap().is_none(),
            "{built} layers built: the draft banks fit beside {capacity} slots"
        );
        let admitted = store
            .prepare_mtp_expert_banks_with(&mut |_: u64| -> Result<()> {
                panic!("the draft banks were admitted")
            })
            .unwrap();
        assert_eq!(admitted, 0);
        assert!(store.mtp_banks_declined);
        assert_eq!(store.mtp_bank_bytes().unwrap(), 0);
        assert_eq!(
            store
                .prepare_mtp_expert_banks_with(&mut |_| Ok(()))
                .unwrap(),
            0,
            "a declined build must not retry"
        );
        build_slot_banks(&mut store, &layers[built..], capacity).unwrap();
    }
    store.bank_bytes = FIXED_BANK_BYTES;
    store.cache_bytes = FIXED_BANK_BYTES;
    store.slot_bytes = 0;
    store.slot_capacity = Some(85);
    store.mtp_banks_declined = false;
    assert!(
        store.mtp_bank_admission().unwrap().is_some(),
        "85 slots leave the staging"
    );
    assert_eq!(store.bytes_read, 0);
}

/// The gate_up draft bank takes f32 sides (+0.2 GiB) only where they fit
/// and keeps the f16 layout otherwise, never declining a bank the f16
/// build would take: below a 22.66 GiB slot allowance, beside a direct
/// plan that charged the f16 layout, or with only the f16 staging left.
/// The reserve never holds less than the build.
/// MUTATIONS: only the f32 layout -> 22 GiB; the slot allowance beside a
/// direct charge -> the f16 charge; no f16 retry -> the f16 staging.
#[test]
fn draft_banks_take_f32_sides_only_where_they_fit() {
    let Some((mut store, _)) = draft_bank_store(None) else {
        return;
    };
    const F16_BANKS: u64 = 2_831_155_200;
    const F32_GATE_UP: u64 = 3_040_870_400;
    let layout = |store: &Store| {
        store.mtp_bank_admission().unwrap().map(|specs| {
            let bytes: u64 = specs.iter().map(|spec| spec.5).sum();
            (
                bytes,
                specs.iter().map(|spec| spec.4.unwrap()).collect::<Vec<_>>(),
            )
        })
    };
    let f16 = Some((F16_BANKS, vec![DType::Float16; 2]));
    let f32 = Some((F32_GATE_UP, vec![DType::Float32, DType::Float16]));
    store.bank_bytes = FIXED_BANK_BYTES;
    store.cache_bytes = FIXED_BANK_BYTES;
    // cache_limit / 8 holds the f32 layout from 22.66 GiB.
    for (sixteenths, want, built) in [
        (22 * 16, F16_BANKS, &f16),
        (22 * 16 + 10, F16_BANKS, &f16),
        (22 * 16 + 11, F32_GATE_UP, &f32),
        (23 * 16, F32_GATE_UP, &f32),
    ] {
        store.cache_limit = sixteenths << 26;
        let gib = sixteenths as f64 / 16.;
        assert_eq!(store.mtp_bank_estimate().unwrap(), want, "{gib} GiB");
        assert_eq!(&layout(&store), built, "{gib} GiB");
    }
    store.cache_limit = 40 << 30;
    for (charged, built) in [(F16_BANKS, &f16), (F32_GATE_UP, &f32)] {
        store.draft_bank_budget = charged;
        let why = format!("a direct plan charged {charged}");
        assert_eq!(store.mtp_bank_estimate().unwrap(), charged, "{why}");
        assert_eq!(&layout(&store), built, "{why}");
    }
    store.draft_bank_budget = 0;
    // Pinned slot banks leave two copies of the f16 gate_up bank and the
    // staging reads: the reserve holds the f32 build, the admission
    // takes the f16 one.
    store.cache_limit = 30 << 30;
    store.slot_bytes =
        store.cache_limit - FIXED_BANK_BYTES - 2 * 1_887_436_800 - MAX_READ_BYTES * 3;
    assert_eq!(store.mtp_bank_estimate().unwrap(), F32_GATE_UP);
    assert_eq!(layout(&store), f16, "the f16 staging");
    store.slot_bytes += 1;
    assert_eq!(layout(&store), None);
    assert_eq!(store.bytes_read, 0);
}
/// The released checkpoint loaded on the slot path at a forced budget:
/// `prepare_hot` caches the slot count, the load's call admits
/// the draft banks, and a turn builds every layer's slot bank beside
/// them. 22 GiB is a budget the product picks; at 22.25 GiB a count
/// cached without the draft reserve overruns the last layer's staging.
/// Opt-in, the same variables as `real_fallback_slots_beside_draft_banks`:
/// `cargo test -p mlx-core --lib real_slot_load_admits_draft_banks_beside_the_cached_count
/// -- --ignored --nocapture`
#[test]
#[ignore = "real checkpoint: QWEN4_EXP_REAL_MODEL, QWEN4_EXP_REAL_INPUTS, QWEN4_EXP_REAL_AUX; --test-threads=1"]
fn real_slot_load_admits_draft_banks_beside_the_cached_count() {
    use super::super::super::tests::{argmax, real_prompt};
    use super::super::super::{auxiliary, decoder::Decoder, gguf};
    let model = std::env::var("QWEN4_EXP_REAL_MODEL").expect("QWEN4_EXP_REAL_MODEL");
    let aux = std::env::var("QWEN4_EXP_REAL_AUX").expect("QWEN4_EXP_REAL_AUX");
    let inputs = std::env::var("QWEN4_EXP_REAL_INPUTS").expect("QWEN4_EXP_REAL_INPUTS");
    let mut prompt = real_prompt(Path::new(&inputs), "6k");
    prompt.truncate(8);
    let mut first: Option<Vec<u32>> = None;
    for limit in [22u64 << 30, (89u64 << 30) / 4] {
        let gib = limit as f64 / (1u64 << 30) as f64;
        let mut store = Store::open(Path::new(&model)).unwrap();
        let mut config = gguf::config(&store).unwrap();
        auxiliary::attach(&mut store, &mut config, Path::new(&aux)).unwrap();
        let mut decoder = Decoder::new(config, store).unwrap();
        decoder.weights.set_cache_limit(limit);
        decoder.register_cache_limit();
        decoder.weights.prepare_fixture_hot().unwrap();
        decoder.weights.publish_live();
        decoder.residency_changed();
        assert_eq!(decoder.weights.cache_limit, limit);
        let slots = decoder.weights.plan.partial_expert_slots;
        assert_eq!(decoder.weights.slot_capacity, Some(slots));
        let draft = decoder.weights.prepare_mtp_expert_banks().unwrap();
        eprintln!(
            "{gib} GiB: {slots} slots cached, draft banks {:.3} GiB, banks {:.3} GiB",
            draft as f64 / (1u64 << 30) as f64,
            decoder.weights.bank_bytes as f64 / (1u64 << 30) as f64
        );
        assert_eq!(draft, 2_831_155_200, "{gib} GiB admits the draft banks");
        decoder.reset();
        decoder.weights.begin_turn().unwrap();
        let mut logits = decoder
            .prefill_chunk(&prompt, None, true)
            .unwrap_or_else(|e| panic!("the slot turn failed at {slots} slots, {gib} GiB: {e}"));
        let mut text = Vec::new();
        for _ in 0..16 {
            let id = argmax(&logits);
            text.push(id);
            logits = decoder.step(id).unwrap();
        }
        assert_eq!(
            decoder.weights.slots.len(),
            48,
            "every layer built its slots"
        );
        assert!(decoder.weights.cache_bytes <= decoder.weights.cache_limit);
        match &first {
            None => first = Some(text),
            Some(first) => assert_eq!(&text, first, "{gib} GiB gives the 22 GiB text"),
        }
        eprintln!(
            "{gib} GiB: slot bytes {:.3} GiB, cache {:.3} GiB, peak {:.3} GiB",
            decoder.weights.slot_bytes as f64 / (1u64 << 30) as f64,
            decoder.weights.cache_bytes as f64 / (1u64 << 30) as f64,
            decoder.weights.peak_cache_bytes as f64 / (1u64 << 30) as f64
        );
        drop(decoder);
        crate::array::memory::synchronize_and_clear_cache();
    }
}
#[test]
fn slot_replacement_completes_all_lazy_readers_and_retains_warm_codes() {
    let dir = std::env::temp_dir().join(format!("qwen4-slots-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(dir.clone());
    let path = dir.join("experts.gguf");
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(3u64.to_le_bytes());
    b.extend(1u64.to_le_bytes());
    let string = |b: &mut Vec<u8>, s: &str| {
        b.extend((s.len() as u64).to_le_bytes());
        b.extend(s.bytes());
    };
    string(&mut b, "general.architecture");
    b.extend(8u32.to_le_bytes());
    string(&mut b, "qwen4exp");
    for (part, name) in ["gate", "up", "down"].iter().enumerate() {
        string(&mut b, &format!("blk.0.ffn_{name}_exps.weight"));
        b.extend(3u32.to_le_bytes());
        for d in [32u64, 32, 5] {
            b.extend(d.to_le_bytes());
        }
        b.extend(7u32.to_le_bytes());
        b.extend((part as u64 * 5 * 32 * 24).to_le_bytes());
    }
    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    for row in 0..3 * 5 * 32 {
        b.extend(half::f16::from_f32(0.015625).to_bits().to_le_bytes());
        b.extend(half::f16::from_f32(-0.25).to_bits().to_le_bytes());
        b.extend((row as u32 * 7919).to_le_bytes());
        b.extend([row as u8; 16]);
    }
    std::fs::write(&path, b).unwrap();
    let mut store = Store::open_fixture(&path, Some(&dir)).unwrap();
    store.slot_capacity = Some(2);
    let x = MxArray::from_float32(
        &(0..64).map(|n| (n as f32 - 32.) / 64.).collect::<Vec<_>>(),
        &[2, 1, 32],
    )
    .unwrap()
    .astype(DType::BFloat16)
    .unwrap();
    let mut expected = Vec::new();
    for ids in [[0u32, 1], [1, 0], [2, 3], [0, 4], [4, 0]] {
        let rows = ids
            .iter()
            .enumerate()
            .map(|(r, &id)| {
                store
                    .read("blk.0.ffn_gate_exps.weight", id as usize * 32, 32)
                    .unwrap()
                    .linear(&x.slice_axis(0, r as i64, r as i64 + 1).unwrap())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        expected.push(
            MxArray::concatenate_many(rows.iter().collect(), Some(0))
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .to_vec(),
        );
    }
    let mut lazy = Vec::new();
    for (n, ids) in [[0u32, 1], [1, 0], [2, 3], [0, 4], [4, 0]]
        .iter()
        .enumerate()
    {
        let uploads = store.slot_upload_bytes;
        let (banks, local, count) = store.expert_slots(0, 5, 2, ids).unwrap().unwrap();
        let (_, device_local) = store
            .device_expert_slots(0, &MxArray::from_uint32(ids, &[2]).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(&*device_local.to_uint32().unwrap(), &local);
        assert!(store.commit_device_routes(&[0], ids, 2).unwrap());
        let absent = (0..5u32).find(|e| !ids.contains(e)).unwrap();
        let old_ages = store.slots[&0].ages.clone();
        let (_, missing) = store
            .device_expert_slots(0, &MxArray::from_uint32(&[absent], &[1]).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(&*missing.to_uint32().unwrap(), &[count as u32]);
        let (_, safe, safe_count) = store
            .prefill_device_expert_slots(
                0,
                &MxArray::from_uint32(&[ids[0], absent, ids[1]], &[3]).unwrap(),
            )
            .unwrap();
        assert_eq!(safe_count, count);
        assert_eq!(&*safe.to_uint32().unwrap(), &[local[0], 0, local[1]]);
        assert!(
            !store
                .commit_device_routes(&[0], &[ids[0], absent, ids[1]], 3)
                .unwrap()
        );
        assert!(!store.commit_device_routes(&[0], &[absent], 1).unwrap());
        assert_eq!(store.slots[&0].ages, old_ages);
        let output = banks[0]
            .gather_rows(
                &x,
                &MxArray::from_uint32(&local, &[2]).unwrap(),
                count,
                false,
            )
            .unwrap();
        if n == 1 || n == 3 {
            // Replaces earlier independent readers on a hit, then
            // protects an asynchronously submitted result across a miss.
            store.defer_expert_reduction(0, &output).unwrap();
            assert_eq!(store.slots[&0].readers.len(), 1);
        } else {
            store.finish_expert_slots(0, &output).unwrap();
        }
        if n == 1 || n == 4 {
            assert_eq!(
                store.slot_upload_bytes, uploads,
                "warm slots must not upload"
            );
        }
        lazy.push(output);
    }
    assert_eq!(
        [
            store.live.slot_hits.load(Ordering::Relaxed),
            store.live.slot_misses.load(Ordering::Relaxed),
        ],
        [store.slot_hits, store.slot_misses],
        "slot and device-route hits reach the live counters"
    );
    let reduced = lazy.last().unwrap().sum(Some(&[-1]), Some(false)).unwrap();
    store.complete_expert_window(0, &reduced).unwrap();
    assert!(store.slots[&0].readers.is_empty());
    store.expert_slots(0, 5, 2, &[1, 2]).unwrap().unwrap();
    for (output, want) in lazy.iter().zip(expected) {
        let got = output.astype(DType::Float32).unwrap().to_float32().unwrap();
        assert_eq!(&*got, &want, "reusing a slot changed a prior lazy output");
    }
    assert_eq!(store.slots[&0].occupants.len(), 2);
    assert!(store.cache_bytes >= store.slot_bytes);
    assert!(store.peak_cache_bytes <= store.cache_limit);
    let reads = store.bytes_read;
    assert!(store.expert_slots(0, 5, 2, &[5]).is_err());
    assert_eq!(
        reads, store.bytes_read,
        "bad ids must fail before reading weights"
    );
}
