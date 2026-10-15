/**
 * Parameter Management Utilities
 *
 * Provides utilities to convert between different parameter representations:
 * - Structured dictionaries (HashMap<String, MxArray>) for functional forward pass
 *
 * Note: The actual gradient flow uses HashMap-based lookup (not positional ordering):
 * 1. model.get_parameters() returns HashMap<String, MxArray>
 * 2. autograd.rs sorts keys alphabetically for flat param vector
 * 3. Gradients are mapped back to HashMap by name
 * 4. apply_gradients() looks up parameters by name
 */
use crate::array::MxArray;
use napi::bindgen_prelude::*;
use std::collections::HashMap;

/// Map flat parameter vector to structured dictionary
///
/// Converts a flat vector of parameters (from autograd) back to a structured
/// dictionary that can be used by the functional forward pass.
///
/// # Arguments
/// * `params` - Flat vector of parameter arrays
/// * `param_names` - Names corresponding to each parameter
///
/// # Returns
/// * HashMap mapping parameter names to arrays
///
/// # Panics
/// * If params.len() != param_names.len()
pub fn map_params_to_dict(
    params: &[MxArray],
    param_names: &[String],
) -> Result<HashMap<String, MxArray>> {
    if params.len() != param_names.len() {
        return Err(Error::from_reason(format!(
            "Parameter count mismatch: {} arrays but {} names",
            params.len(),
            param_names.len()
        )));
    }

    let mut param_dict = HashMap::new();

    for (param, name) in params.iter().zip(param_names.iter()) {
        // Use clone() not copy() - clone just increments Arc reference count (O(1))
        // while copy() creates a full tensor copy with eval() (O(n))
        param_dict.insert(name.clone(), param.clone());
    }

    Ok(param_dict)
}
