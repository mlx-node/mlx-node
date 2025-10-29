// Advantage computation for GRPO
// Reference: trl/trl/trainer/grpo_trainer.py lines 1567-1588
//
// This module implements group-based advantage computation for GRPO training.
// Advantages are computed by:
// 1. Grouping rewards by prompt
// 2. Computing zero-mean per group (reward - mean_group_reward)
// 3. Normalizing by std (group, batch, or none)

use crate::array::MxArray;
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Compute advantages for GRPO from rewards
///
/// Reference: TRL grpo_trainer.py:1567-1588
///
/// Algorithm:
/// 1. Reshape rewards from (B*G,) to (B, G) where B=batch, G=num_generations
/// 2. Compute mean reward per group (per prompt): mean_grouped_rewards
/// 3. Advantages = rewards - mean_grouped_rewards (zero-mean per group)
/// 4. Normalize by std based on scale_rewards:
///    - "group": Normalize by std within each group
///    - "batch": Normalize by global std across all rewards
///    - "none": No normalization (but still zero-mean)
///
/// # Arguments
/// * `rewards` - Reward values, shape (B*G,) where B=batch_size, G=num_generations
/// * `num_generations` - Number of completions per prompt (G)
/// * `scale_rewards` - How to normalize: "group", "batch", or "none"
///
/// # Returns
/// Advantages, shape (B*G,)
#[napi]
pub fn compute_advantages(
    rewards: &MxArray,
    num_generations: i32,
    scale_rewards: String,
) -> Result<MxArray> {
    // Validate inputs
    let rewards_shape = rewards.shape()?;
    if rewards_shape.len() != 1 {
        let shape_vec: Vec<i64> = rewards_shape.iter().copied().collect();
        return Err(Error::new(
            Status::InvalidArg,
            format!("rewards must be 1D, got shape: {:?}", shape_vec),
        ));
    }

    let total_size = rewards_shape[0];
    if total_size % (num_generations as i64) != 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "rewards size {} must be divisible by numGenerations {}",
                total_size, num_generations
            ),
        ));
    }

    let batch_size = total_size / (num_generations as i64);

    // Step 1: Reshape rewards to (B, G)
    let rewards_reshaped = rewards.reshape(&[batch_size, num_generations as i64])?;

    // Step 2: Compute mean reward per group (per prompt)
    // mean_grouped_rewards shape: (B,)
    let mean_grouped_rewards = rewards_reshaped.mean(Some(&[1]), Some(false))?;

    // Step 3: Repeat mean to (B*G,) for broadcasting
    // In PyTorch: mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
    // In our case: reshape (B,) -> (B, 1) then repeat along axis 1, then flatten
    let mean_expanded = mean_grouped_rewards.reshape(&[batch_size, 1])?;
    let mean_repeated = mean_expanded.broadcast_to(&[batch_size, num_generations as i64])?;
    let mean_flat = mean_repeated.reshape(&[total_size])?;

    // Step 4: Compute advantages = rewards - mean
    let mut advantages = rewards.sub(&mean_flat)?;

    // Step 5: Normalize based on scale_rewards
    match scale_rewards.as_str() {
        "group" => {
            // Compute std per group
            // std_rewards = rewards.view(-1, num_generations).std(dim=1)
            let std_per_group = rewards_reshaped.std(Some(&[1]), Some(false), Some(0))?;

            // Repeat std to (B*G,)
            let std_expanded = std_per_group.reshape(&[batch_size, 1])?;
            let std_repeated = std_expanded.broadcast_to(&[batch_size, num_generations as i64])?;
            let std_flat = std_repeated.reshape(&[total_size])?;

            // Normalize: advantages / (std + epsilon)
            // Match TRL's epsilon of 1e-4 (no clamping - TRL doesn't clamp advantages)
            let epsilon = MxArray::scalar_float(1e-4)?;
            let std_plus_eps = std_flat.add(&epsilon)?;
            advantages = advantages.div(&std_plus_eps)?;
        }
        "batch" => {
            // Compute global std across all rewards
            let all_axes: Vec<i32> = (0..rewards_shape.len() as i32).collect();
            let std_global = rewards.std(Some(&all_axes), Some(false), Some(0))?;

            // Normalize: advantages / (std + epsilon)
            // Match TRL's epsilon of 1e-4 (no clamping - TRL doesn't clamp advantages)
            let std_plus_eps = std_global.add_scalar(1e-4)?;
            let std_broadcasted = std_plus_eps.broadcast_to(&[total_size])?;
            advantages = advantages.div(&std_broadcasted)?;
        }
        "none" => {
            // No normalization, just return zero-meaned advantages
        }
        _ => {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "Invalid scale_rewards: {}. Must be \"group\", \"batch\", or \"none\"",
                    scale_rewards
                ),
            ));
        }
    }

    Ok(advantages)
}

/// Group statistics result
pub struct GroupStats {
    pub mean: MxArray,
    pub std: MxArray,
}

/// Compute group-level statistics for rewards
///
/// # Arguments
/// * `rewards` - Reward values, shape (B*G,)
/// * `num_generations` - Number of completions per prompt
///
/// # Returns
/// Object with mean and std per group
pub fn compute_group_stats(rewards: &MxArray, num_generations: i32) -> Result<GroupStats> {
    let rewards_shape = rewards.shape()?;
    let total_size = rewards_shape[0];
    let batch_size = total_size / (num_generations as i64);

    // Reshape to (B, G)
    let rewards_reshaped = rewards.reshape(&[batch_size, num_generations as i64])?;

    // Compute mean and std per group
    let mean = rewards_reshaped.mean(Some(&[1]), Some(false))?; // shape: (B,)
    let std = rewards_reshaped.std(Some(&[1]), Some(false), Some(0))?; // shape: (B,)

    Ok(GroupStats { mean, std })
}

/// Normalize rewards to zero-mean and unit variance
///
/// This is a simpler whitening operation that normalizes globally
///
/// # Arguments
/// * `rewards` - Input rewards
///
/// # Returns
/// Normalized rewards
pub fn normalize_rewards(rewards: &MxArray) -> Result<MxArray> {
    let mean = rewards.mean(None, Some(false))?;

    // Compute std across all axes
    let rewards_shape = rewards.shape()?;
    let all_axes: Vec<i32> = (0..rewards_shape.len() as i32).collect();
    let std = rewards.std(Some(&all_axes), Some(false), Some(0))?;

    let centered = rewards.sub(&mean.broadcast_to(&rewards_shape)?)?;
    let std_plus_eps = std.add_scalar(1e-4)?;

    centered.div(&std_plus_eps.broadcast_to(&rewards_shape)?)
}
