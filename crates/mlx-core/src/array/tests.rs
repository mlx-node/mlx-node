use super::*;

#[test]
fn test_basic_array_creation() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let arr = MxArray::from_float32(&data, &[2, 2]).unwrap();
    assert_eq!(arr.size().unwrap(), 4);
}

// Helper to compare float arrays with tolerance
fn assert_arrays_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "Array lengths differ: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tolerance,
            "Arrays differ at index {}: {} vs {} (diff: {})",
            i,
            a,
            e,
            (a - e).abs()
        );
    }
}

// Helper to convert BigInt64Array to Vec<i64>
fn shape_to_vec(shape: napi::bindgen_prelude::BigInt64Array) -> Vec<i64> {
    shape.as_ref().to_vec()
}

// Helper to convert Int32Array to Vec<i32>
fn int32_to_vec(arr: napi::bindgen_prelude::Int32Array) -> Vec<i32> {
    arr.as_ref().to_vec()
}

// ========================================
// Array Creation Operations
// ========================================

mod creation {
    use super::*;

    #[test]
    fn test_zeros() {
        let x = MxArray::zeros(&[2, 3], None).unwrap();
        assert_eq!(shape_to_vec(x.shape().unwrap()), vec![2, 3]);
        let values = x.to_float32().unwrap();
        assert_eq!(values.len(), 6);
        assert!(values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_ones() {
        let x = MxArray::ones(&[2, 3], None).unwrap();
        assert_eq!(shape_to_vec(x.shape().unwrap()), vec![2, 3]);
        let values = x.to_float32().unwrap();
        assert_eq!(values.len(), 6);
        assert!(values.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_full_scalar() {
        let x = MxArray::full(&[2, 3], Either::A(3.5), None).unwrap();
        assert_eq!(shape_to_vec(x.shape().unwrap()), vec![2, 3]);
        let values = x.to_float32().unwrap();
        assert_arrays_close(&values, &[3.5, 3.5, 3.5, 3.5, 3.5, 3.5], 1e-5);
    }

    #[test]
    fn test_full_with_int_dtype() {
        let x = MxArray::full(&[2, 2], Either::A(7.0), Some(DType::Int32)).unwrap();
        assert_eq!(shape_to_vec(x.shape().unwrap()), vec![2, 2]);
        let values = int32_to_vec(x.to_int32().unwrap());
        assert_eq!(values, vec![7, 7, 7, 7]);
    }

    #[test]
    fn test_arange() {
        let x = MxArray::arange(0.0, 5.0, None, None).unwrap();
        assert_eq!(shape_to_vec(x.shape().unwrap()), vec![5]);
        let values = x.to_float32().unwrap();
        assert_arrays_close(&values, &[0.0, 1.0, 2.0, 3.0, 4.0], 1e-5);
    }

    #[test]
    fn test_arange_with_step() {
        let x = MxArray::arange(0.0, 10.0, Some(2.0), None).unwrap();
        assert_eq!(shape_to_vec(x.shape().unwrap()), vec![5]);
        let values = x.to_float32().unwrap();
        assert_arrays_close(&values, &[0.0, 2.0, 4.0, 6.0, 8.0], 1e-5);
    }

    #[test]
    fn test_linspace_default() {
        let x = MxArray::linspace(0.0, 1.0, None, None).unwrap();
        assert_eq!(shape_to_vec(x.shape().unwrap()), vec![50]);
        let values = x.to_float32().unwrap();
        assert!((values[0] - 0.0).abs() < 1e-5);
        assert!((values[49] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_linspace_custom() {
        let x = MxArray::linspace(-2.0, 2.0, Some(5), Some(DType::Int32)).unwrap();
        assert_eq!(shape_to_vec(x.shape().unwrap()), vec![5]);
        let values = int32_to_vec(x.to_int32().unwrap());
        assert_eq!(values, vec![-2, -1, 0, 1, 2]);
    }

    #[test]
    fn test_eye_identity() {
        let eye = MxArray::eye(3, None, None, None).unwrap();
        assert_eq!(shape_to_vec(eye.shape().unwrap()), vec![3, 3]);
        let values = eye.to_float32().unwrap();
        assert_arrays_close(
            &values,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            1e-5,
        );
    }

    #[test]
    fn test_eye_rectangular_offset() {
        let eye = MxArray::eye(3, Some(4), Some(1), None).unwrap();
        assert_eq!(shape_to_vec(eye.shape().unwrap()), vec![3, 4]);
        let values = eye.to_float32().unwrap();
        let expected = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        assert_arrays_close(&values, &expected, 1e-5);
    }

    #[test]
    fn test_eye_negative_offset() {
        let eye = MxArray::eye(4, Some(3), Some(-1), None).unwrap();
        assert_eq!(shape_to_vec(eye.shape().unwrap()), vec![4, 3]);
        let values = eye.to_float32().unwrap();
        let expected = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_arrays_close(&values, &expected, 1e-5);
    }
}

// ========================================
// Arithmetic Operations
// ========================================

mod arithmetic {
    use super::*;

    #[test]
    fn test_add() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = MxArray::from_float32(&[4.0, 5.0, 6.0], &[3]).unwrap();
        let c = a.add(&b).unwrap();
        let values = c.to_float32().unwrap();
        assert_arrays_close(&values, &[5.0, 7.0, 9.0], 1e-5);
    }

    #[test]
    fn test_add_scalar() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = a.add_scalar(10.0).unwrap();
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[11.0, 12.0, 13.0], 1e-5);
    }

    #[test]
    fn test_sub() {
        let a = MxArray::from_float32(&[5.0, 6.0, 7.0], &[3]).unwrap();
        let b = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let c = a.sub(&b).unwrap();
        let values = c.to_float32().unwrap();
        assert_arrays_close(&values, &[4.0, 4.0, 4.0], 1e-5);
    }

    #[test]
    fn test_sub_scalar() {
        let a = MxArray::from_float32(&[5.0, 6.0, 7.0], &[3]).unwrap();
        let b = a.sub_scalar(2.0).unwrap();
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[3.0, 4.0, 5.0], 1e-5);
    }

    #[test]
    fn test_mul() {
        let a = MxArray::from_float32(&[2.0, 3.0, 4.0], &[3]).unwrap();
        let b = MxArray::from_float32(&[5.0, 6.0, 7.0], &[3]).unwrap();
        let c = a.mul(&b).unwrap();
        let values = c.to_float32().unwrap();
        assert_arrays_close(&values, &[10.0, 18.0, 28.0], 1e-5);
    }

    #[test]
    fn test_mul_scalar() {
        let a = MxArray::from_float32(&[2.0, 3.0, 4.0], &[3]).unwrap();
        let b = a.mul_scalar(3.0).unwrap();
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[6.0, 9.0, 12.0], 1e-5);
    }

    #[test]
    fn test_div() {
        let a = MxArray::from_float32(&[10.0, 20.0, 30.0], &[3]).unwrap();
        let b = MxArray::from_float32(&[2.0, 4.0, 5.0], &[3]).unwrap();
        let c = a.div(&b).unwrap();
        let values = c.to_float32().unwrap();
        assert_arrays_close(&values, &[5.0, 5.0, 6.0], 1e-5);
    }

    #[test]
    fn test_div_scalar() {
        let a = MxArray::from_float32(&[10.0, 20.0, 30.0], &[3]).unwrap();
        let b = a.div_scalar(10.0).unwrap();
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 2.0, 3.0], 1e-5);
    }

    #[test]
    fn test_negative_numbers() {
        let a = MxArray::from_float32(&[-1.0, -2.0, -3.0], &[3]).unwrap();
        let b = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let c = a.add(&b).unwrap();
        let values = c.to_float32().unwrap();
        assert_arrays_close(&values, &[0.0, 0.0, 0.0], 1e-5);
    }

    #[test]
    fn test_floor_divide() {
        let a = MxArray::from_float32(&[7.0, 8.0, 9.0], &[3]).unwrap();
        let b = MxArray::from_float32(&[2.0, 3.0, 4.0], &[3]).unwrap();
        let c = a.floor_divide(&b).unwrap();
        let values = c.to_float32().unwrap();
        assert_arrays_close(&values, &[3.0, 2.0, 2.0], 1e-5);
    }

    #[test]
    fn test_remainder() {
        let a = MxArray::from_float32(&[7.0, 8.0, 9.0], &[3]).unwrap();
        let b = MxArray::from_float32(&[3.0, 3.0, 4.0], &[3]).unwrap();
        let c = a.remainder(&b).unwrap();
        let values = c.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 2.0, 1.0], 1e-5);
    }

    #[test]
    fn test_power() {
        let a = MxArray::from_float32(&[2.0, 3.0, 4.0], &[3]).unwrap();
        let b = MxArray::from_float32(&[2.0, 2.0, 2.0], &[3]).unwrap();
        let c = a.power(&b).unwrap();
        let values = c.to_float32().unwrap();
        assert_arrays_close(&values, &[4.0, 9.0, 16.0], 1e-5);
    }
}

// ========================================
// Comparison Operations
// ========================================

mod comparison {
    use super::*;

    #[test]
    fn test_equal() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let c = a.equal(&b).unwrap();
        let values = c.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 1.0, 1.0], 1e-5);
    }

    #[test]
    fn test_not_equal() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = MxArray::from_float32(&[1.0, 0.0, 3.0], &[3]).unwrap();
        let c = a.not_equal(&b).unwrap();
        let values = c.to_float32().unwrap();
        assert_arrays_close(&values, &[0.0, 1.0, 0.0], 1e-5);
    }

    #[test]
    fn test_less() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = MxArray::from_float32(&[2.0, 2.0, 2.0], &[3]).unwrap();
        let c = a.less(&b).unwrap();
        let values = c.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 0.0, 0.0], 1e-5);
    }

    #[test]
    fn test_greater() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = MxArray::from_float32(&[0.0, 2.0, 4.0], &[3]).unwrap();
        let c = a.greater(&b).unwrap();
        let values = c.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 0.0, 0.0], 1e-5);
    }

    #[test]
    fn test_less_equal() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = MxArray::from_float32(&[1.0, 2.0, 4.0], &[3]).unwrap();
        let c = a.less_equal(&b).unwrap();
        let values = c.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 1.0, 1.0], 1e-5);
    }

    #[test]
    fn test_greater_equal() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = MxArray::from_float32(&[0.0, 2.0, 3.0], &[3]).unwrap();
        let c = a.greater_equal(&b).unwrap();
        let values = c.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 1.0, 1.0], 1e-5);
    }
}

// ========================================
// Reduction Operations
// ========================================

mod reduction {
    use super::*;

    #[test]
    fn test_sum() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let sum = a.sum(None, None).unwrap();
        let values = sum.to_float32().unwrap();
        assert!((values[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_sum_along_axis() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let sum0 = a.sum(Some(&[0]), None).unwrap();
        let values0 = sum0.to_float32().unwrap();
        assert_arrays_close(&values0, &[5.0, 7.0, 9.0], 1e-5);

        let sum1 = a.sum(Some(&[1]), None).unwrap();
        let values1 = sum1.to_float32().unwrap();
        assert_arrays_close(&values1, &[6.0, 15.0], 1e-5);
    }

    #[test]
    fn test_mean() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let mean = a.mean(None, None).unwrap();
        let values = mean.to_float32().unwrap();
        assert!((values[0] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_mean_along_axis() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let mean0 = a.mean(Some(&[0]), None).unwrap();
        let values0 = mean0.to_float32().unwrap();
        assert_arrays_close(&values0, &[2.5, 3.5, 4.5], 1e-5);

        let mean1 = a.mean(Some(&[1]), None).unwrap();
        let values1 = mean1.to_float32().unwrap();
        assert_arrays_close(&values1, &[2.0, 5.0], 1e-5);
    }

    #[test]
    fn test_prod() {
        let a = MxArray::from_float32(&[2.0, 3.0, 4.0], &[3]).unwrap();
        let prod = a.prod(None, None).unwrap();
        let values = prod.to_float32().unwrap();
        assert!((values[0] - 24.0).abs() < 1e-5);
    }

    #[test]
    fn test_max() {
        let a = MxArray::from_float32(&[1.0, 5.0, 3.0, 2.0], &[4]).unwrap();
        let max_val = a.max(None, None).unwrap();
        let values = max_val.to_float32().unwrap();
        assert!((values[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_min() {
        let a = MxArray::from_float32(&[1.0, 5.0, 3.0, 2.0], &[4]).unwrap();
        let min_val = a.min(None, None).unwrap();
        let values = min_val.to_float32().unwrap();
        assert!((values[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_argmax() {
        let a = MxArray::from_float32(&[1.0, 5.0, 3.0, 2.0], &[4]).unwrap();
        let idx = a.argmax(0, None).unwrap();
        let values = int32_to_vec(idx.to_int32().unwrap());
        assert_eq!(values[0], 1);
    }

    #[test]
    fn test_argmin() {
        let a = MxArray::from_float32(&[1.0, 5.0, 3.0, 2.0], &[4]).unwrap();
        let idx = a.argmin(0, None).unwrap();
        let values = int32_to_vec(idx.to_int32().unwrap());
        assert_eq!(values[0], 0);
    }
}

// ========================================
// Shape Operations
// ========================================

mod shape_ops {
    use super::*;

    #[test]
    fn test_reshape() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).unwrap();
        let b = a.reshape(&[2, 3]).unwrap();
        assert_eq!(shape_to_vec(b.shape().unwrap()), vec![2, 3]);
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5);
    }

    #[test]
    fn test_transpose_2d() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = a.transpose(None).unwrap();
        assert_eq!(shape_to_vec(b.shape().unwrap()), vec![3, 2]);
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-5);
    }

    #[test]
    fn test_expand_dims() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = a.expand_dims(0).unwrap();
        assert_eq!(shape_to_vec(b.shape().unwrap()), vec![1, 3]);
    }

    #[test]
    fn test_squeeze() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[1, 3, 1]).unwrap();
        let b = a.squeeze(None).unwrap();
        assert_eq!(shape_to_vec(b.shape().unwrap()), vec![3]);
    }

    #[test]
    fn test_broadcast_to() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = a.broadcast_to(&[2, 3]).unwrap();
        assert_eq!(shape_to_vec(b.shape().unwrap()), vec![2, 3]);
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], 1e-5);
    }
}

// ========================================
// Mathematical Functions
// ========================================

mod math_funcs {
    use super::*;

    #[test]
    fn test_exp() {
        use std::f32::consts::E;
        let a = MxArray::from_float32(&[0.0, 1.0, 2.0], &[3]).unwrap();
        let b = a.exp().unwrap();
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, E, E * E], 1e-4);
    }

    #[test]
    fn test_log() {
        use std::f32::consts::E;
        let a = MxArray::from_float32(&[1.0, E, E * E], &[3]).unwrap();
        let b = a.log().unwrap();
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[0.0, 1.0, 2.0], 1e-4);
    }

    #[test]
    fn test_sqrt() {
        let a = MxArray::from_float32(&[1.0, 4.0, 9.0, 16.0], &[4]).unwrap();
        let b = a.sqrt().unwrap();
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 2.0, 3.0, 4.0], 1e-5);
    }

    #[test]
    fn test_square() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = a.square().unwrap();
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 4.0, 9.0, 16.0], 1e-5);
    }

    #[test]
    fn test_abs() {
        let a = MxArray::from_float32(&[-1.0, -2.0, 3.0, -4.0], &[4]).unwrap();
        let b = a.abs().unwrap();
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 2.0, 3.0, 4.0], 1e-5);
    }

    #[test]
    fn test_sin() {
        let a = MxArray::from_float32(&[0.0, std::f32::consts::PI / 2.0], &[2]).unwrap();
        let b = a.sin().unwrap();
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[0.0, 1.0], 1e-5);
    }

    #[test]
    fn test_cos() {
        let a = MxArray::from_float32(&[0.0, std::f32::consts::PI], &[2]).unwrap();
        let b = a.cos().unwrap();
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, -1.0], 1e-5);
    }

    #[test]
    fn test_floor() {
        let a = MxArray::from_float32(&[1.2, 2.7, 3.5], &[3]).unwrap();
        let b = a.floor().unwrap();
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 2.0, 3.0], 1e-5);
    }

    #[test]
    fn test_ceil() {
        let a = MxArray::from_float32(&[1.2, 2.7, 3.5], &[3]).unwrap();
        let b = a.ceil().unwrap();
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[2.0, 3.0, 4.0], 1e-5);
    }

    #[test]
    fn test_round() {
        let a = MxArray::from_float32(&[1.2, 2.7, 3.5], &[3]).unwrap();
        let b = a.round().unwrap();
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 3.0, 4.0], 1e-5);
    }
}

// ========================================
// Logical Operations
// ========================================

mod logical {
    use super::*;

    #[test]
    fn test_logical_and() {
        let a = MxArray::from_float32(&[1.0, 0.0, 1.0], &[3]).unwrap();
        let b = MxArray::from_float32(&[1.0, 1.0, 0.0], &[3]).unwrap();
        let c = a.logical_and(&b).unwrap();
        let values = c.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 0.0, 0.0], 1e-5);
    }

    #[test]
    fn test_logical_or() {
        let a = MxArray::from_float32(&[1.0, 0.0, 1.0], &[3]).unwrap();
        let b = MxArray::from_float32(&[1.0, 1.0, 0.0], &[3]).unwrap();
        let c = a.logical_or(&b).unwrap();
        let values = c.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 1.0, 1.0], 1e-5);
    }

    #[test]
    fn test_logical_not() {
        let a = MxArray::from_float32(&[1.0, 0.0, 1.0], &[3]).unwrap();
        let b = a.logical_not().unwrap();
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[0.0, 1.0, 0.0], 1e-5);
    }
}

// ========================================
// Indexing and Slicing
// ========================================

mod indexing {
    use super::*;

    #[test]
    fn test_slice_basic() {
        let a = MxArray::from_float32(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5]).unwrap();
        let b = a.slice(&[1], &[4]).unwrap();
        assert_eq!(shape_to_vec(b.shape().unwrap()), vec![3]);
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 2.0, 3.0], 1e-5);
    }

    #[test]
    fn test_slice_2d() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = a.slice(&[0, 1], &[2, 3]).unwrap();
        assert_eq!(shape_to_vec(b.shape().unwrap()), vec![2, 2]);
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[2.0, 3.0, 5.0, 6.0], 1e-5);
    }

    #[test]
    fn test_take() {
        let a = MxArray::from_float32(&[10.0, 20.0, 30.0, 40.0], &[4]).unwrap();
        let indices = MxArray::from_int32(&[0, 2, 3], &[3]).unwrap();
        let b = a.take(&indices, 0).unwrap();
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[10.0, 30.0, 40.0], 1e-5);
    }
}

// ========================================
// Linear Algebra
// ========================================

mod linalg {
    use super::*;

    #[test]
    fn test_matmul_2d() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = MxArray::from_float32(&[5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(shape_to_vec(c.shape().unwrap()), vec![2, 2]);
        let values = c.to_float32().unwrap();
        // [[1,2], [3,4]] @ [[5,6], [7,8]] = [[19,22], [43,50]]
        assert_arrays_close(&values, &[19.0, 22.0, 43.0, 50.0], 1e-4);
    }

    #[test]
    fn test_matmul_vector() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(shape_to_vec(c.shape().unwrap()), vec![2]);
        let values = c.to_float32().unwrap();
        // [[1,2,3], [4,5,6]] @ [1,2,3] = [14, 32]
        assert_arrays_close(&values, &[14.0, 32.0], 1e-4);
    }
}

// ========================================
// Ordering Operations
// ========================================

mod ordering {
    use super::*;

    #[test]
    fn test_sort() {
        let a = MxArray::from_float32(&[3.0, 1.0, 4.0, 2.0], &[4]).unwrap();
        let b = a.sort(None).unwrap();
        let values = b.to_float32().unwrap();
        assert_arrays_close(&values, &[1.0, 2.0, 3.0, 4.0], 1e-5);
    }

    #[test]
    fn test_argsort() {
        let a = MxArray::from_float32(&[3.0, 1.0, 4.0, 2.0], &[4]).unwrap();
        let indices = a.argsort(None).unwrap();
        let values = int32_to_vec(indices.to_int32().unwrap());
        assert_eq!(values, vec![1, 3, 0, 2]);
    }
}

// ========================================
// Metadata and Utilities
// ========================================

mod metadata {
    use super::*;

    #[test]
    fn test_shape() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(shape_to_vec(a.shape().unwrap()), vec![2, 3]);
    }

    #[test]
    fn test_ndim() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(a.ndim().unwrap(), 2);
    }

    #[test]
    fn test_size() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(a.size().unwrap(), 6);
    }

    #[test]
    fn test_dtype() {
        let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
        assert_eq!(a.dtype().unwrap(), DType::Float32);

        let b = MxArray::from_int32(&[1, 2, 3], &[3]).unwrap();
        assert_eq!(b.dtype().unwrap(), DType::Int32);
    }

    #[test]
    fn test_astype() {
        let a = MxArray::from_float32(&[1.5, 2.7, 3.2], &[3]).unwrap();
        let b = a.astype(DType::Int32).unwrap();
        assert_eq!(b.dtype().unwrap(), DType::Int32);
        let values = int32_to_vec(b.to_int32().unwrap());
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_nbytes_float32() {
        // Float32 = 4 bytes per element
        let arr = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        // 5 elements * 4 bytes = 20 bytes
        assert_eq!(arr.nbytes(), 20);
    }

    #[test]
    fn test_nbytes_int32() {
        // Int32 = 4 bytes per element
        let arr = MxArray::from_int32(&[1, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        // 6 elements * 4 bytes = 24 bytes
        assert_eq!(arr.nbytes(), 24);
    }

    #[test]
    fn test_nbytes_uint32() {
        // Uint32 = 4 bytes per element
        let arr = MxArray::from_uint32(&[1, 2, 3, 4], &[2, 2]).unwrap();
        // 4 elements * 4 bytes = 16 bytes
        assert_eq!(arr.nbytes(), 16);
    }

    #[test]
    fn test_nbytes_2d_array() {
        // Create a 3x4 Float32 array = 12 elements
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let arr = MxArray::from_float32(&data, &[3, 4]).unwrap();
        // 12 elements * 4 bytes = 48 bytes
        assert_eq!(arr.nbytes(), 48);
    }

    #[test]
    fn test_nbytes_3d_array() {
        // Create a 2x3x4 Float32 array = 24 elements
        let data = vec![1.0f32; 24];
        let arr = MxArray::from_float32(&data, &[2, 3, 4]).unwrap();
        // 24 elements * 4 bytes = 96 bytes
        assert_eq!(arr.nbytes(), 96);
    }

    #[test]
    fn test_nbytes_zeros() {
        let arr = MxArray::zeros(&[10], Some(DType::Float32)).unwrap();
        // 10 elements * 4 bytes = 40 bytes
        assert_eq!(arr.nbytes(), 40);
    }

    #[test]
    fn test_nbytes_ones() {
        let arr = MxArray::ones(&[5, 5], Some(DType::Float32)).unwrap();
        // 25 elements * 4 bytes = 100 bytes
        assert_eq!(arr.nbytes(), 100);
    }

    #[test]
    fn test_nbytes_scalar() {
        let arr = MxArray::scalar_float(42.5).unwrap();
        // Scalar has at least some bytes allocated
        assert!(arr.nbytes() > 0);
    }

    #[test]
    fn test_nbytes_large_array() {
        // Create a larger array
        let size = 100000;
        let arr = MxArray::zeros(&[size], Some(DType::Float32)).unwrap();
        // 100,000 elements * 4 bytes = 400,000 bytes
        assert_eq!(arr.nbytes(), 400000);
    }
}

/// Edge case tests for functional components
///
/// Tests boundary conditions, extreme values, and error handling
/// to ensure robustness in production use.
mod edge_cases {
    use super::*;

    fn assert_shape_eq(arr: &MxArray, expected: &[i64]) {
        let shape = arr.shape().unwrap();
        let shape_vec: Vec<i64> = shape.to_vec();
        assert_eq!(shape_vec, expected, "Shape mismatch");
    }

    fn assert_all_finite(arr: &MxArray) {
        let data = arr.to_float32().unwrap();
        for (i, &val) in data.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Value at index {} is not finite: {}",
                i,
                val
            );
        }
    }

    // Empty and Single Element Inputs
    #[test]
    fn test_single_token_sequences() {
        let vocab_size = 100i64;
        let hidden_size = 64i64;

        let embedding_weight =
            MxArray::random_normal(&[vocab_size, hidden_size], 0.0, 0.02, None).unwrap();
        let input_ids = MxArray::from_int32(&[42], &[1]).unwrap();

        let embeddings = embedding_weight.take(&input_ids, 0).unwrap();

        assert_shape_eq(&embeddings, &[1, hidden_size]);

        assert_all_finite(&embeddings);
    }

    #[test]
    fn test_single_batch_single_token() {
        let vocab_size = 50i64;
        let hidden_size = 32i64;

        let embedding_weight =
            MxArray::random_normal(&[vocab_size, hidden_size], 0.0, 0.02, None).unwrap();
        let input_ids = MxArray::from_int32(&[0], &[1, 1]).unwrap();

        let embeddings = embedding_weight.take(&input_ids, 0).unwrap();

        assert_shape_eq(&embeddings, &[1, 1, hidden_size]);
    }

    #[test]
    fn test_minimum_model_dimensions() {
        let batch_size = 1i64;
        let seq_len = 1i64;
        let hidden_size = 4i64;

        let input =
            MxArray::random_normal(&[batch_size, seq_len, hidden_size], 0.0, 0.02, None).unwrap();
        let weight = MxArray::random_normal(&[hidden_size, hidden_size], 0.0, 0.02, None).unwrap();

        let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

        assert_shape_eq(&output, &[batch_size, seq_len, hidden_size]);
    }

    // Very Long Sequences
    #[test]
    fn test_sequence_length_1024() {
        let batch_size = 1i64;
        let seq_len = 1024i64;
        let hidden_size = 128i64;

        let input =
            MxArray::random_normal(&[batch_size, seq_len, hidden_size], 0.0, 0.02, None).unwrap();
        let weight = MxArray::random_normal(&[hidden_size, hidden_size], 0.0, 0.02, None).unwrap();

        let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

        assert_shape_eq(&output, &[batch_size, seq_len, hidden_size]);

        // Check numerical stability
        let max_val = output
            .abs()
            .unwrap()
            .max(None, None)
            .unwrap()
            .to_float32()
            .unwrap()[0];
        assert!(max_val.is_finite());
        assert!(max_val < 100.0, "Value should not explode, got {}", max_val);
    }

    #[test]
    fn test_sequence_length_2048() {
        let batch_size = 1i64;
        let seq_len = 2048i64;
        let hidden_size = 64i64;

        let input =
            MxArray::random_normal(&[batch_size, seq_len, hidden_size], 0.0, 0.02, None).unwrap();
        let weight = MxArray::random_normal(&[hidden_size, hidden_size], 0.0, 0.02, None).unwrap();

        let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

        assert_shape_eq(&output, &[batch_size, seq_len, hidden_size]);
    }

    // Large Batch Sizes
    #[test]
    fn test_batch_size_32() {
        let batch_size = 32i64;
        let seq_len = 16i64;
        let hidden_size = 128i64;

        let input =
            MxArray::random_normal(&[batch_size, seq_len, hidden_size], 0.0, 0.02, None).unwrap();
        let weight = MxArray::random_normal(&[hidden_size, hidden_size], 0.0, 0.02, None).unwrap();

        let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

        assert_shape_eq(&output, &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_batch_size_64() {
        let batch_size = 64i64;
        let seq_len = 8i64;
        let hidden_size = 64i64;

        let input =
            MxArray::random_normal(&[batch_size, seq_len, hidden_size], 0.0, 0.02, None).unwrap();
        let weight = MxArray::random_normal(&[hidden_size, hidden_size], 0.0, 0.02, None).unwrap();

        let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

        assert_shape_eq(&output, &[batch_size, seq_len, hidden_size]);
    }

    // Extreme Values
    #[test]
    fn test_all_zeros_input() {
        let input = MxArray::zeros(&[2, 5, 64], None).unwrap();
        let weight = MxArray::random_normal(&[64, 64], 0.0, 0.02, None).unwrap();

        let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

        // Output should be zero (or near zero)
        let max_val = output
            .abs()
            .unwrap()
            .max(None, None)
            .unwrap()
            .to_float32()
            .unwrap()[0];
        assert!(max_val < 1e-5);
    }

    #[test]
    fn test_all_ones_input() {
        let input = MxArray::ones(&[2, 5, 64], None).unwrap();
        let weight = MxArray::random_normal(&[64, 64], 0.0, 0.02, None).unwrap();

        let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

        // Output should be sum of weight columns
        assert_shape_eq(&output, &[2, 5, 64]);

        let max_val = output
            .abs()
            .unwrap()
            .max(None, None)
            .unwrap()
            .to_float32()
            .unwrap()[0];
        assert!(max_val.is_finite());
    }

    #[test]
    fn test_negative_values() {
        let input = MxArray::full(&[2, 5, 64], Either::A(-1.0), None).unwrap();
        let weight = MxArray::random_normal(&[64, 64], 0.0, 0.02, None).unwrap();

        let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

        assert_shape_eq(&output, &[2, 5, 64]);
        assert_all_finite(&output);
    }

    #[test]
    fn test_very_small_values() {
        let input = MxArray::full(&[2, 5, 64], Either::A(1e-8), None).unwrap();
        let weight = MxArray::random_normal(&[64, 64], 0.0, 0.02, None).unwrap();

        let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

        // Output should be very small but finite
        let max_val = output
            .abs()
            .unwrap()
            .max(None, None)
            .unwrap()
            .to_float32()
            .unwrap()[0];
        assert!(max_val.is_finite());
        assert!(max_val < 1e-5);
    }

    #[test]
    fn test_rms_norm_zero_mean() {
        // Input with zero mean but non-zero values
        let input = MxArray::from_float32(&[1.0, -1.0, 2.0, -2.0], &[4]).unwrap();
        let weight = MxArray::ones(&[4], None).unwrap();
        let eps = 1e-6;

        let squared = input.square().unwrap();
        let mean_squared = squared.mean(Some(&[-1]), Some(true)).unwrap();
        let eps_array = MxArray::full(&[], Either::A(eps), None).unwrap();
        let variance = mean_squared.add(&eps_array).unwrap();
        let rms = variance.sqrt().unwrap();
        let normalized = input.div(&rms).unwrap();
        let output = normalized.mul(&weight).unwrap();

        assert_all_finite(&output);
    }

    // Boundary Token IDs
    #[test]
    fn test_token_id_zero() {
        let vocab_size = 100i64;
        let hidden_size = 64i64;

        let embedding_weight =
            MxArray::random_normal(&[vocab_size, hidden_size], 0.0, 0.02, None).unwrap();
        let input_ids = MxArray::from_int32(&[0, 0, 0], &[3]).unwrap();

        let embeddings = embedding_weight.take(&input_ids, 0).unwrap();

        assert_shape_eq(&embeddings, &[3, hidden_size]);

        // All three should be identical (same token)
        let data = embeddings.to_float32().unwrap();
        let first = &data[0..hidden_size as usize];
        let second = &data[hidden_size as usize..2 * hidden_size as usize];

        let max_diff = first
            .iter()
            .zip(second.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(max_diff < 1e-6);
    }

    #[test]
    fn test_maximum_valid_token_id() {
        let vocab_size = 100i64;
        let hidden_size = 64i64;

        let embedding_weight =
            MxArray::random_normal(&[vocab_size, hidden_size], 0.0, 0.02, None).unwrap();
        let input_ids = MxArray::from_int32(&[(vocab_size - 1) as i32], &[1]).unwrap();

        let embeddings = embedding_weight.take(&input_ids, 0).unwrap();

        assert_shape_eq(&embeddings, &[1, hidden_size]);
        assert_all_finite(&embeddings);
    }

    #[test]
    fn test_repeated_tokens() {
        let vocab_size = 50i64;
        let hidden_size = 32i64;

        let embedding_weight =
            MxArray::random_normal(&[vocab_size, hidden_size], 0.0, 0.02, None).unwrap();
        let input_ids = MxArray::from_int32(&[5, 5, 5, 5, 5], &[5]).unwrap();

        let embeddings = embedding_weight.take(&input_ids, 0).unwrap();

        assert_shape_eq(&embeddings, &[5, hidden_size]);

        // All should be identical
        let data = embeddings.to_float32().unwrap();
        for i in 1..5 {
            let current = &data[(i * hidden_size as usize)..((i + 1) * hidden_size as usize)];
            let first = &data[0..hidden_size as usize];
            let max_diff = current
                .iter()
                .zip(first.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_diff < 1e-6);
        }
    }

    // Dimension Preservation
    #[test]
    fn test_preserve_shapes_through_operations() {
        let batch = 2i64;
        let seq = 5i64;
        let hidden = 64i64;

        let input = MxArray::random_normal(&[batch, seq, hidden], 0.0, 0.02, None).unwrap();
        let weight = MxArray::random_normal(&[hidden, hidden], 0.0, 0.02, None).unwrap();

        let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

        // Should maintain batch and sequence dimensions
        let result_shape = output.shape().unwrap();
        assert_eq!(result_shape[0], batch);
        assert_eq!(result_shape[1], seq);
        assert_eq!(result_shape[2], hidden);
    }

    #[test]
    fn test_asymmetric_weight_matrices() {
        let batch = 2i64;
        let seq = 5i64;
        let in_dim = 64i64;
        let out_dim = 128i64;

        let input = MxArray::random_normal(&[batch, seq, in_dim], 0.0, 0.02, None).unwrap();
        let weight = MxArray::random_normal(&[out_dim, in_dim], 0.0, 0.02, None).unwrap();

        let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

        assert_shape_eq(&output, &[batch, seq, out_dim]);
    }

    // Numerical Precision
    #[test]
    fn test_precision_through_multiple_operations() {
        let input = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let identity = MxArray::from_float32(
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            &[4, 4],
        )
        .unwrap();

        // Multiply by identity 10 times
        let mut result = input.reshape(&[1, 4]).unwrap();
        for _ in 0..10 {
            result = result.matmul(&identity).unwrap();
        }

        // Should still be close to original
        let final_result = result.reshape(&[4]).unwrap();
        let diff = final_result.sub(&input).unwrap();
        let max_diff = diff
            .abs()
            .unwrap()
            .max(None, None)
            .unwrap()
            .to_float32()
            .unwrap()[0];
        assert!(max_diff < 1e-5);
    }

    #[test]
    fn test_repeated_normalization() {
        let mut input = MxArray::random_normal(&[2, 8], 0.0, 1.0, None).unwrap();
        let weight = MxArray::ones(&[8], None).unwrap();
        let eps = 1e-6;

        // Normalize 5 times
        for _ in 0..5 {
            let squared = input.square().unwrap();
            let mean_squared = squared.mean(Some(&[-1]), Some(true)).unwrap();
            let eps_array = MxArray::full(&[], Either::A(eps), None).unwrap();
            let variance = mean_squared.add(&eps_array).unwrap();
            let rms = variance.sqrt().unwrap();
            input = input.div(&rms).unwrap().mul(&weight).unwrap();
        }

        // RMS should still be close to 1
        let final_squared = input.square().unwrap();
        let final_rms = final_squared
            .mean(Some(&[-1]), Some(false))
            .unwrap()
            .sqrt()
            .unwrap();
        let avg_rms = final_rms.mean(None, None).unwrap().to_float32().unwrap()[0];
        assert!(
            (avg_rms - 1.0).abs() < 0.1,
            "Expected ~1.0, got {}",
            avg_rms
        );
    }

    // Memory and Performance
    #[test]
    fn test_moderate_size_efficiency() {
        let batch_size = 8i64;
        let seq_len = 128i64;
        let hidden_size = 256i64;

        let input =
            MxArray::random_normal(&[batch_size, seq_len, hidden_size], 0.0, 0.02, None).unwrap();
        let weight = MxArray::random_normal(&[hidden_size, hidden_size], 0.0, 0.02, None).unwrap();

        let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

        assert_shape_eq(&output, &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_large_model_dimensions() {
        let batch_size = 4i64;
        let seq_len = 64i64;
        let hidden_size = 1024i64;

        let input =
            MxArray::random_normal(&[batch_size, seq_len, hidden_size], 0.0, 0.02, None).unwrap();
        let weight = MxArray::random_normal(&[hidden_size, hidden_size], 0.0, 0.02, None).unwrap();

        let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

        assert_shape_eq(&output, &[batch_size, seq_len, hidden_size]);

        // Check numerical stability with large dimensions
        let max_val = output
            .abs()
            .unwrap()
            .max(None, None)
            .unwrap()
            .to_float32()
            .unwrap()[0];
        assert!(max_val.is_finite());
        assert!(max_val < 10.0, "Should stay bounded, got {}", max_val);
    }

    // SwiGLU Edge Cases
    #[test]
    fn test_swiglu_zero_gate_values() {
        let batch_size = 2i64;
        let seq_len = 5i64;
        let hidden_size = 32i64;
        let intermediate_size = 128i64;

        let input =
            MxArray::random_normal(&[batch_size, seq_len, hidden_size], 0.0, 0.02, None).unwrap();
        let gate_weight = MxArray::zeros(&[intermediate_size, hidden_size], None).unwrap();
        let up_weight =
            MxArray::random_normal(&[intermediate_size, hidden_size], 0.0, 0.02, None).unwrap();
        let down_weight =
            MxArray::random_normal(&[hidden_size, intermediate_size], 0.0, 0.02, None).unwrap();

        let gate = input.matmul(&gate_weight.transpose(None).unwrap()).unwrap();
        let up = input.matmul(&up_weight.transpose(None).unwrap()).unwrap();

        // SiLU of zero is zero
        let neg_gate = gate.negative().unwrap();
        let exp_neg_gate = neg_gate.exp().unwrap();
        let one = MxArray::full(&[], Either::A(1.0), None).unwrap();
        let one_plus_exp = one.add(&exp_neg_gate).unwrap();
        let sigmoid = one.div(&one_plus_exp).unwrap();
        let gate_act = gate.mul(&sigmoid).unwrap();

        let gated = gate_act.mul(&up).unwrap();
        let output = gated.matmul(&down_weight.transpose(None).unwrap()).unwrap();

        // Output should be near zero
        let max_val = output
            .abs()
            .unwrap()
            .max(None, None)
            .unwrap()
            .to_float32()
            .unwrap()[0];
        assert!(max_val < 1e-4);
    }

    #[test]
    fn test_swiglu_extreme_activation_values() {
        let batch = 2i64;
        let seq = 3i64;
        let hidden = 16i64;
        let intermediate = 64i64;

        let input = MxArray::full(&[batch, seq, hidden], Either::A(10.0), None).unwrap();
        let gate_weight = MxArray::random_normal(&[intermediate, hidden], 0.0, 0.02, None).unwrap();
        let up_weight = MxArray::random_normal(&[intermediate, hidden], 0.0, 0.02, None).unwrap();
        let down_weight = MxArray::random_normal(&[hidden, intermediate], 0.0, 0.02, None).unwrap();

        let gate = input.matmul(&gate_weight.transpose(None).unwrap()).unwrap();
        let up = input.matmul(&up_weight.transpose(None).unwrap()).unwrap();
        let neg_gate = gate.negative().unwrap();
        let exp_neg_gate = neg_gate.exp().unwrap();
        let one = MxArray::full(&[], Either::A(1.0), None).unwrap();
        let one_plus_exp = one.add(&exp_neg_gate).unwrap();
        let sigmoid = one.div(&one_plus_exp).unwrap();
        let gate_act = gate.mul(&sigmoid).unwrap();
        let gated = gate_act.mul(&up).unwrap();
        let output = gated.matmul(&down_weight.transpose(None).unwrap()).unwrap();

        // Should not overflow
        let max_val = output
            .abs()
            .unwrap()
            .max(None, None)
            .unwrap()
            .to_float32()
            .unwrap()[0];
        assert!(max_val.is_finite());
    }
}

// ========================================
// Metal Buffer Extraction Tests
// ========================================

mod metal_buffer {
    use super::*;

    // Basic as_raw_ptr tests work on all platforms
    #[test]
    fn test_as_raw_ptr_not_null() {
        let arr = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        arr.eval();
        let ptr = arr.as_raw_ptr();
        assert!(!ptr.is_null(), "Raw pointer should not be null");
    }

    #[test]
    fn test_as_raw_ptr_different_arrays() {
        let arr1 = MxArray::from_float32(&[1.0, 2.0], &[2]).unwrap();
        let arr2 = MxArray::from_float32(&[3.0, 4.0], &[2]).unwrap();
        arr1.eval();
        arr2.eval();

        let ptr1 = arr1.as_raw_ptr();
        let ptr2 = arr2.as_raw_ptr();

        // Different arrays should have different handles
        assert_ne!(ptr1, ptr2, "Different arrays should have different handles");
    }

    #[test]
    fn test_as_raw_ptr_after_eval() {
        let arr = MxArray::random_normal(&[100, 100], 0.0, 1.0, None).unwrap();
        arr.eval();

        let ptr = arr.as_raw_ptr();
        assert!(!ptr.is_null());

        // The array should still be usable after getting ptr
        let data = arr.to_float32().unwrap();
        assert_eq!(data.len(), 10000);
    }

    // Metal-specific tests only run on macOS
    #[cfg(target_os = "macos")]
    mod macos_metal_tests {
        use super::*;

        #[test]
        fn test_metal_buffer_extraction() {
            use mlx_paged_attn::metal::{MlxMetalBuffer, synchronize_mlx};

            let arr = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
            arr.eval();
            synchronize_mlx();

            let ptr = arr.as_raw_ptr();
            let buffer_info = unsafe { MlxMetalBuffer::from_mlx_array(ptr) };

            assert!(
                buffer_info.is_some(),
                "Should extract Metal buffer from evaluated array"
            );

            let info = buffer_info.unwrap();
            assert!(
                !info.buffer_ptr.is_null(),
                "Buffer pointer should not be null"
            );
            assert!(info.data_size > 0, "Data size should be positive");
            assert_eq!(info.itemsize, 4, "Float32 should have itemsize 4");
        }

        #[test]
        fn test_metal_buffer_offset() {
            use mlx_paged_attn::metal::{MlxMetalBuffer, synchronize_mlx};

            // Create a sliced array to test offset
            let arr = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).unwrap();
            arr.eval();
            synchronize_mlx();

            let ptr = arr.as_raw_ptr();
            let buffer_info = unsafe { MlxMetalBuffer::from_mlx_array(ptr) };

            assert!(buffer_info.is_some());
            let info = buffer_info.unwrap();

            // Full array should have offset 0
            assert_eq!(info.offset, 0, "Full array should have zero offset");
        }

        #[test]
        fn test_metal_buffer_data_size() {
            use mlx_paged_attn::metal::{MlxMetalBuffer, synchronize_mlx};

            // 8 float32 elements
            let arr =
                MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8]).unwrap();
            arr.eval();
            synchronize_mlx();

            let ptr = arr.as_raw_ptr();
            let buffer_info = unsafe { MlxMetalBuffer::from_mlx_array(ptr) };

            assert!(buffer_info.is_some());
            let info = buffer_info.unwrap();

            // data_size returns number of elements, not bytes
            assert_eq!(info.data_size, 8, "Data size should be 8 elements");
            assert_eq!(info.itemsize, 4, "Float32 itemsize should be 4 bytes");
            assert_eq!(info.data_size_bytes(), 32, "Total size should be 32 bytes");
        }

        #[test]
        fn test_metal_synchronize() {
            use mlx_paged_attn::metal::synchronize_mlx;

            // Create and evaluate a large array to ensure GPU work is queued
            let arr = MxArray::random_normal(&[1000, 1000], 0.0, 1.0, None).unwrap();
            let result = arr.matmul(&arr.transpose(None).unwrap()).unwrap();
            result.eval();

            // Should not panic
            synchronize_mlx();

            // Array should be fully evaluated now
            let data = result.to_float32().unwrap();
            assert_eq!(data.len(), 1000 * 1000);
        }

        #[test]
        fn test_metal_extraction_supported() {
            use mlx_paged_attn::metal::is_metal_extraction_supported;

            // On macOS, Metal extraction is typically supported but may not be
            // available on headless CI, VMs, or systems without Metal GPU.
            // We just verify the function runs and returns a valid boolean.
            let supported = is_metal_extraction_supported();
            eprintln!("Metal extraction supported: {}", supported);
            // Don't assert - headless CI may not have Metal
        }

        #[test]
        fn test_sliced_array_metal_buffer() {
            use mlx_paged_attn::metal::{MlxMetalBuffer, synchronize_mlx};

            // Create array and take a slice
            let full_arr =
                MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8]).unwrap();
            let sliced = full_arr.slice(&[2], &[6]).unwrap();
            sliced.eval();
            synchronize_mlx();

            let ptr = sliced.as_raw_ptr();
            let buffer_info = unsafe { MlxMetalBuffer::from_mlx_array(ptr) };

            assert!(
                buffer_info.is_some(),
                "Should extract buffer from sliced array"
            );
        }

        #[test]
        fn test_2d_array_metal_buffer() {
            use mlx_paged_attn::metal::{MlxMetalBuffer, synchronize_mlx};

            let arr = MxArray::random_normal(&[16, 32], 0.0, 1.0, None).unwrap();
            arr.eval();
            synchronize_mlx();

            let ptr = arr.as_raw_ptr();
            let buffer_info = unsafe { MlxMetalBuffer::from_mlx_array(ptr) };

            assert!(buffer_info.is_some());
            let info = buffer_info.unwrap();

            // 16 * 32 = 512 elements (data_size returns element count, not bytes)
            assert_eq!(info.data_size, 512, "Data size should be 512 elements");
            assert_eq!(info.data_size_bytes(), 2048, "Total bytes should be 2048");
        }
    }
}

mod batched_generation_helpers {
    use super::*;

    #[test]
    fn test_repeat_along_axis_basic() {
        // Test: [1, 2, 3] repeated 3 times along axis 0
        let arr = MxArray::from_int32(&[1, 2, 3], &[1, 3]).unwrap();
        let repeated = arr.repeat_along_axis(0, 3).unwrap();

        repeated.eval();
        let shape = shape_to_vec(repeated.shape().unwrap());
        assert_eq!(shape, vec![3, 3]);

        let values = int32_to_vec(repeated.to_int32().unwrap());
        // Each row should be [1, 2, 3]
        assert_eq!(values, vec![1, 2, 3, 1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn test_repeat_along_axis_4d() {
        // Simulate KV cache expansion: [1, heads, seq, dim] -> [G, heads, seq, dim]
        let arr = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 1, 2]).unwrap();
        let repeated = arr.repeat_along_axis(0, 4).unwrap();

        repeated.eval();
        let shape = shape_to_vec(repeated.shape().unwrap());
        assert_eq!(shape, vec![4, 2, 1, 2]);

        let values = repeated.to_float32().unwrap();
        // All 4 batches should have the same values
        assert_eq!(values.len(), 16);
        for i in 0..4 {
            assert_arrays_close(&values[i * 4..(i + 1) * 4], &[1.0, 2.0, 3.0, 4.0], 1e-5);
        }
    }

    #[test]
    fn test_repeat_along_axis_no_repeat() {
        let arr = MxArray::from_int32(&[1, 2], &[2]).unwrap();
        let repeated = arr.repeat_along_axis(0, 1).unwrap();

        repeated.eval();
        let shape = shape_to_vec(repeated.shape().unwrap());
        assert_eq!(shape, vec![2]);
    }

    #[test]
    fn test_item_at_float32_2d() {
        // Create a 3x4 array
        let values: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let arr = MxArray::from_float32(&values, &[3, 4]).unwrap();
        arr.eval();

        // Test various indices
        assert_eq!(arr.item_at_float32_2d(0, 0).unwrap(), 0.0);
        assert_eq!(arr.item_at_float32_2d(0, 3).unwrap(), 3.0);
        assert_eq!(arr.item_at_float32_2d(1, 0).unwrap(), 4.0);
        assert_eq!(arr.item_at_float32_2d(1, 2).unwrap(), 6.0);
        assert_eq!(arr.item_at_float32_2d(2, 3).unwrap(), 11.0);
    }
}
