/**
 * Unit test for the F32 -> IEEE-754 half (F16) encoder + decoder used by the
 * f16 shipped-pack export (Task T4.0).
 *
 * This is a PURE-MATH test — it imports ONLY the `f32ToF16` / `f16ToF32` helpers
 * from `export-shipped-pack.mts` (whose module top-level runs the actual export
 * ONLY when invoked as the entry script, guarded by an `isMain` check), so this
 * test loads NO model and touches NO GPU. It is therefore SAFE to run while the
 * controller owns the serial GPU.
 *
 * Run with: env PATH="/opt/homebrew/bin:$PATH" oxnode \
 *   packages/browser/scripts/jlens/export-shipped-pack.test.mts
 *   (NOT tsx/ts-node — repo convention.)
 *
 * Prints `ENCODER TEST OK` + exits 0 on success; exits 1 (naming the case) on any
 * mismatch.
 */
import { f16ToF32, f32ToF16 } from './export-shipped-pack.mts';

let failures = 0;
function check(label: string, got: number, want: number): void {
  const ok = Object.is(got, want);
  if (!ok) {
    failures++;
    const fmt = (x: number) =>
      Number.isInteger(x) && x >= 0 && x <= 0xffff ? `0x${x.toString(16).padStart(4, '0')}` : String(x);
    console.error(`FAIL  ${label}: got ${fmt(got)} want ${fmt(want)}`);
  } else {
    console.log(`PASS  ${label}`);
  }
}
function checkPred(label: string, pred: boolean): void {
  if (!pred) {
    failures++;
    console.error(`FAIL  ${label}`);
  } else {
    console.log(`PASS  ${label}`);
  }
}

// ---- exact IEEE half bit patterns (encode) ------------------------------------
// hex ref: the standard binary16 encoding. u16 = sign(1)|exp(5)|mantissa(10).
const P2 = (n: number) => 2 ** n;
check('encode 0', f32ToF16(0), 0x0000);
check('encode -0 (sign preserved)', f32ToF16(-0), 0x8000);
check('encode 1.0', f32ToF16(1.0), 0x3c00);
check('encode -1.0', f32ToF16(-1.0), 0xbc00);
check('encode 65504 (max half normal)', f32ToF16(65504), 0x7bff);
check('encode 65520 -> +Inf (overflow, tie rounds up past max)', f32ToF16(65520), 0x7c00);
check('encode -65520 -> -Inf', f32ToF16(-65520), 0xfc00);
check('encode 2^-14 (min normal half)', f32ToF16(P2(-14)), 0x0400); // 6.103515625e-5
check('encode 2^-24 (min subnormal half)', f32ToF16(P2(-24)), 0x0001); // 5.9604644775e-8
check('encode 2^-25 (halfway 0..min-subnormal, ties to even 0)', f32ToF16(P2(-25)), 0x0000);
check('encode +Inf', f32ToF16(Infinity), 0x7c00);
check('encode -Inf', f32ToF16(-Infinity), 0xfc00);
// round-to-nearest-EVEN, both directions, in the NORMAL range:
//   1 + 2^-11   = 1.00048828125   is exactly halfway between 0x3c00 (1.0) and
//                 0x3c01; LSB of 0x3c00 is 0 (even) -> ties DOWN to 0x3c00.
check('encode 1+2^-11 (tie -> even, rounds DOWN to 0x3c00)', f32ToF16(1 + P2(-11)), 0x3c00);
//   1 + 12288/2^23 = 1.00146484375 is exactly halfway between 0x3c01 and 0x3c02;
//                 LSB of 0x3c01 is 1 (odd) -> ties UP to the even 0x3c02.
check('encode 1+12288/2^23 (tie -> even, rounds UP to 0x3c02)', f32ToF16(1 + 12288 / P2(23)), 0x3c02);

// ---- NaN preservation (encode): must stay a half-NaN, never collapse to Inf ---
const nanBits = f32ToF16(NaN);
checkPred('encode NaN -> exponent all ones (0x7c00 bits set)', (nanBits & 0x7c00) === 0x7c00);
checkPred('encode NaN -> nonzero mantissa (NaN, not Inf)', (nanBits & 0x03ff) !== 0);

// ---- decode (f16 -> f32) -------------------------------------------------------
check('decode 0x0000 -> 0', f16ToF32(0x0000), 0);
check('decode 0x8000 -> -0 (sign preserved)', f16ToF32(0x8000), -0);
check('decode 0x3c00 -> 1.0', f16ToF32(0x3c00), 1.0);
check('decode 0xbc00 -> -1.0', f16ToF32(0xbc00), -1.0);
check('decode 0x7bff -> 65504', f16ToF32(0x7bff), 65504);
check('decode 0x0400 -> 2^-14', f16ToF32(0x0400), P2(-14));
check('decode 0x0001 -> 2^-24', f16ToF32(0x0001), P2(-24));
check('decode 0x7c00 -> +Inf', f16ToF32(0x7c00), Infinity);
check('decode 0xfc00 -> -Inf', f16ToF32(0xfc00), -Infinity);
checkPred('decode 0x7e00 -> NaN', Number.isNaN(f16ToF32(0x7e00)));

// ---- round-trip identity for values EXACTLY representable in half --------------
for (const x of [0, 1, -1, 0.5, -0.5, 2, -2, 65504, -65504, P2(-14), P2(-24), P2(-24) * 3]) {
  const rt = f16ToF32(f32ToF16(x));
  check(`round-trip exact ${x}`, rt, x);
}
// -0 round-trips to -0 (sign bit survives both directions).
checkPred('round-trip -0 stays -0', Object.is(f16ToF32(f32ToF16(-0)), -0));

if (failures > 0) {
  console.error(`\nENCODER TEST FAILED: ${failures} case(s)`);
  process.exit(1);
}
console.log('\nENCODER TEST OK');
process.exit(0);
