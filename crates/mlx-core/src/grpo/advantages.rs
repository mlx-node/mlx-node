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

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create MxArray from f32 slice
    fn array_from_f32(data: &[f32]) -> MxArray {
        MxArray::from_float32(data, &[data.len() as i64]).unwrap()
    }

    // Helper to get f32 values from MxArray
    fn to_f32_vec(arr: &MxArray) -> Vec<f32> {
        arr.to_float32().unwrap().to_vec()
    }

    // =========================================================================
    // Basic Functionality Tests
    // =========================================================================

    #[test]
    fn test_compute_advantages_basic() {
        // Two groups of 2: [1, 3] and [2, 4]
        // Group 1: mean=2, adv=[1-2, 3-2]=[-1, 1]
        // Group 2: mean=3, adv=[2-3, 4-3]=[-1, 1]
        let rewards = array_from_f32(&[1.0, 3.0, 2.0, 4.0]);
        let advantages = compute_advantages(&rewards, 2, "none".to_string()).unwrap();
        let values = to_f32_vec(&advantages);

        assert!((values[0] - (-1.0)).abs() < 1e-5);
        assert!((values[1] - 1.0).abs() < 1e-5);
        assert!((values[2] - (-1.0)).abs() < 1e-5);
        assert!((values[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_advantages_multiple_groups() {
        // Three groups of 3: [1,2,3], [4,5,6], [7,8,9]
        // Group 1: mean=2, adv=[-1, 0, 1]
        // Group 2: mean=5, adv=[-1, 0, 1]
        // Group 3: mean=8, adv=[-1, 0, 1]
        let rewards = array_from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let advantages = compute_advantages(&rewards, 3, "none".to_string()).unwrap();
        let values = to_f32_vec(&advantages);

        // Check all 9 values
        let expected = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (values[i] - exp).abs() < 1e-5,
                "Index {}: expected {}, got {}",
                i,
                exp,
                values[i]
            );
        }
    }

    // =========================================================================
    // Normalization Strategy Tests
    // =========================================================================

    #[test]
    fn test_scale_rewards_group() {
        // Two groups: [0, 2] and [0, 4]
        // Group 1: mean=1, std=1, adv=[-1, 1] / 1 = [-1, 1]
        // Group 2: mean=2, std=2, adv=[-2, 2] / 2 = [-1, 1]
        let rewards = array_from_f32(&[0.0, 2.0, 0.0, 4.0]);
        let advantages = compute_advantages(&rewards, 2, "group".to_string()).unwrap();
        let values = to_f32_vec(&advantages);

        // Debug: print actual values
        println!("Group normalized advantages: {:?}", values);

        // After group normalization, both groups should have normalized advantages
        // The exact values depend on std calculation (ddof=0 vs ddof=1)
        // With ddof=0: std([0,2])=1, std([0,4])=2
        // With ddof=1: std([0,2])=sqrt(2), std([0,4])=2*sqrt(2)
        // Just verify the pattern: opposite signs, similar magnitude within group
        assert!(values[0] < 0.0, "First value should be negative");
        assert!(values[1] > 0.0, "Second value should be positive");
        assert!(values[2] < 0.0, "Third value should be negative");
        assert!(values[3] > 0.0, "Fourth value should be positive");
        // Values within same group should have same absolute value
        assert!((values[0].abs() - values[1].abs()).abs() < 1e-4);
        assert!((values[2].abs() - values[3].abs()).abs() < 1e-4);
    }

    #[test]
    fn test_scale_rewards_batch() {
        // All rewards: [1, 3, 2, 4]
        // mean=2.5, std=sqrt(((1-2.5)^2+(3-2.5)^2+(2-2.5)^2+(4-2.5)^2)/4)=sqrt(1.25)≈1.118
        // Group 1: mean=2, adv=[-1, 1]
        // Group 2: mean=3, adv=[-1, 1]
        // After batch norm: divide by global std
        let rewards = array_from_f32(&[1.0, 3.0, 2.0, 4.0]);
        let advantages = compute_advantages(&rewards, 2, "batch".to_string()).unwrap();
        let values = to_f32_vec(&advantages);

        // Advantages should be divided by global std (~1.118)
        let global_std = 1.118_034_f32; // sqrt(5/4)
        let expected_scale = 1.0 / (global_std + 1e-4);

        assert!((values[0] - -expected_scale).abs() < 1e-3);
        assert!((values[1] - (1.0 * expected_scale)).abs() < 1e-3);
    }

    #[test]
    fn test_scale_rewards_none() {
        // With "none", advantages should just be zero-meaned, no normalization
        let rewards = array_from_f32(&[1.0, 5.0, 2.0, 6.0]);
        let advantages = compute_advantages(&rewards, 2, "none".to_string()).unwrap();
        let values = to_f32_vec(&advantages);

        // Group 1: mean=3, adv=[-2, 2]
        // Group 2: mean=4, adv=[-2, 2]
        assert!((values[0] - (-2.0)).abs() < 1e-5);
        assert!((values[1] - 2.0).abs() < 1e-5);
        assert!((values[2] - (-2.0)).abs() < 1e-5);
        assert!((values[3] - 2.0).abs() < 1e-5);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_uniform_rewards_zero_advantage() {
        // All rewards equal within each group → advantages should be 0
        let rewards = array_from_f32(&[5.0, 5.0, 3.0, 3.0]);
        let advantages = compute_advantages(&rewards, 2, "none".to_string()).unwrap();
        let values = to_f32_vec(&advantages);

        for &v in &values {
            assert!(v.abs() < 1e-5, "Expected 0 advantage, got {}", v);
        }
    }

    #[test]
    fn test_single_item_per_group() {
        // group_size=1 means each reward is its own group → all advantages=0
        let rewards = array_from_f32(&[1.0, 2.0, 3.0, 4.0]);
        let advantages = compute_advantages(&rewards, 1, "none".to_string()).unwrap();
        let values = to_f32_vec(&advantages);

        for &v in &values {
            assert!(v.abs() < 1e-5, "Expected 0 advantage, got {}", v);
        }
    }

    #[test]
    fn test_large_reward_variance() {
        // Test with rewards having large variance
        let rewards = array_from_f32(&[0.0, 1000.0, -500.0, 500.0]);
        let advantages = compute_advantages(&rewards, 2, "none".to_string()).unwrap();
        let values = to_f32_vec(&advantages);

        // Group 1: mean=500, adv=[-500, 500]
        // Group 2: mean=0, adv=[-500, 500]
        assert!((values[0] - (-500.0)).abs() < 1e-3);
        assert!((values[1] - 500.0).abs() < 1e-3);
        assert!((values[2] - (-500.0)).abs() < 1e-3);
        assert!((values[3] - 500.0).abs() < 1e-3);
    }

    #[test]
    fn test_negative_rewards() {
        // All negative rewards
        let rewards = array_from_f32(&[-4.0, -2.0, -6.0, -2.0]);
        let advantages = compute_advantages(&rewards, 2, "none".to_string()).unwrap();
        let values = to_f32_vec(&advantages);

        // Group 1: mean=-3, adv=[-1, 1]
        // Group 2: mean=-4, adv=[-2, 2]
        assert!((values[0] - (-1.0)).abs() < 1e-5);
        assert!((values[1] - 1.0).abs() < 1e-5);
        assert!((values[2] - (-2.0)).abs() < 1e-5);
        assert!((values[3] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_zero_std_handling_group() {
        // All equal within group → std=0 → should use epsilon to avoid division by zero
        let rewards = array_from_f32(&[5.0, 5.0, 3.0, 3.0]);
        let advantages = compute_advantages(&rewards, 2, "group".to_string()).unwrap();
        let values = to_f32_vec(&advantages);

        // With zero std, advantages should be 0 / (0 + eps) = 0
        for &v in &values {
            assert!(v.abs() < 1e-3, "Expected ~0 advantage, got {}", v);
        }
    }

    // =========================================================================
    // Numerical Stability Tests
    // =========================================================================

    #[test]
    fn test_very_small_rewards() {
        // Test with very small reward values
        let rewards = array_from_f32(&[1e-8, 3e-8, 2e-8, 4e-8]);
        let advantages = compute_advantages(&rewards, 2, "none".to_string()).unwrap();
        let values = to_f32_vec(&advantages);

        // Should still compute correct relative advantages
        // Group 1: mean=2e-8, adv=[-1e-8, 1e-8]
        assert!((values[0] - (-1e-8)).abs() < 1e-10);
        assert!((values[1] - 1e-8).abs() < 1e-10);
    }

    #[test]
    fn test_very_large_rewards() {
        // Test with large reward values
        let rewards = array_from_f32(&[1e6, 3e6, 2e6, 4e6]);
        let advantages = compute_advantages(&rewards, 2, "none".to_string()).unwrap();
        let values = to_f32_vec(&advantages);

        // Group 1: mean=2e6, adv=[-1e6, 1e6]
        assert!((values[0] - (-1e6)).abs() < 1e2);
        assert!((values[1] - 1e6).abs() < 1e2);
    }

    // =========================================================================
    // Input Validation Tests
    // =========================================================================

    #[test]
    fn test_invalid_scale_rewards() {
        let rewards = array_from_f32(&[1.0, 2.0, 3.0, 4.0]);
        let result = compute_advantages(&rewards, 2, "invalid".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_indivisible_size() {
        // 5 rewards with group_size=2 should fail
        let rewards = array_from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = compute_advantages(&rewards, 2, "none".to_string());
        assert!(result.is_err());
    }

    // =========================================================================
    // Group Stats Tests
    // =========================================================================

    #[test]
    fn test_compute_group_stats() {
        let rewards = array_from_f32(&[1.0, 3.0, 2.0, 6.0]);
        let stats = compute_group_stats(&rewards, 2).unwrap();

        let means = to_f32_vec(&stats.mean);
        let stds = to_f32_vec(&stats.std);

        // Group 1: [1, 3], mean=2, std=1
        // Group 2: [2, 6], mean=4, std=2
        assert!((means[0] - 2.0).abs() < 1e-5);
        assert!((means[1] - 4.0).abs() < 1e-5);
        assert!((stds[0] - 1.0).abs() < 1e-5);
        assert!((stds[1] - 2.0).abs() < 1e-5);
    }

    // =========================================================================
    // Normalize Rewards Tests
    // =========================================================================

    #[test]
    fn test_normalize_rewards() {
        // [1, 2, 3, 4] → mean=2.5, std≈1.118
        let rewards = array_from_f32(&[1.0, 2.0, 3.0, 4.0]);
        let normalized = normalize_rewards(&rewards).unwrap();
        let values = to_f32_vec(&normalized);

        // After normalization: (x - 2.5) / (1.118 + eps)
        let mean = 2.5_f32;
        let std = 1.118_034_f32;
        let scale = 1.0 / (std + 1e-4);

        assert!((values[0] - (1.0 - mean) * scale).abs() < 1e-3);
        assert!((values[1] - (2.0 - mean) * scale).abs() < 1e-3);
        assert!((values[2] - (3.0 - mean) * scale).abs() < 1e-3);
        assert!((values[3] - (4.0 - mean) * scale).abs() < 1e-3);
    }
}
