#include <metal_stdlib>

using namespace metal;

// Metal applies a residency set's endResidency at its next GPU operation;
// this one-thread dispatch is that operation (Splash residency_kick).
kernel void residency_kick(device uint *target [[buffer(0)]]) {
  target[0] = 0;
}
