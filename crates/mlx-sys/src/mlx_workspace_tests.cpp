// Bounded real-encoder regression, invoked only by explicit integration tests.
#include "mlx/allocator.h"
#include "mlx_common.h"
#include "mlx_splash_moe_workspace.h"

#include <future>
#include <limits>
#include <thread>

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/metal/device.h"
#include "mlx/event.h"

namespace {
using namespace mlx::core;
namespace workspace = splash_moe::persistent_workspace;
using Spec = workspace::Spec;
constexpr uint64_t identity = 0x57534b5350414345ULL;
constexpr uint64_t failure_identity = 0x57534b5350414346ULL;
const std::vector<Spec> specs{{{32}, uint32}, {{32}, uint32}};

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

struct Settle {
  metal::CommandEncoder &enc;
  ~Settle() {
    try {
      enc.synchronize();
    } catch (...) {
    }
  }
};

array allocation() {
  array a({32}, uint32, nullptr, {});
  a.set_data(allocator::malloc(a.nbytes()));
  return a;
}

const std::vector<array> &acquire(Stream stream, metal::CommandEncoder &enc) {
  const auto *slots = workspace::try_workspace(stream, enc, identity, specs);
  require(slots && slots->size() == 2, "workspace acquisition failed");
  return *slots;
}

void kernel(metal::CommandEncoder &enc, Stream s, const char *name) {
  auto &d = metal::device(s.device);
  auto *lib = d.get_library("persistent_workspace_test_v1", [] {
    return R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void workspace_fill(device uint* a[[buffer(0)]],constant uint& seed[[buffer(1)]],uint i[[thread_position_in_grid]]) {a[i]=seed+i;}
kernel void workspace_read(device const uint* a[[buffer(0)]],device uint* b[[buffer(1)]],uint i[[thread_position_in_grid]]) {b[i]=a[i];}
)METAL";
  });
  enc.set_compute_pipeline_state(d.get_kernel(name, lib));
}

void fill(metal::CommandEncoder &enc, Stream s, array a, uint32_t value) {
  kernel(enc, s, "workspace_fill");
  enc.set_output_array(a, 0);
  enc.set_bytes(value, 1);
  enc.dispatch_threads(MTL::Size(32, 1, 1), MTL::Size(32, 1, 1));
}

void read(metal::CommandEncoder &enc, Stream s, const array &a, array &out) {
  kernel(enc, s, "workspace_read");
  enc.set_input_array(a, 0);
  enc.set_output_array(out, 1);
  enc.dispatch_threads(MTL::Size(32, 1, 1), MTL::Size(32, 1, 1));
}

void check(const array &a, uint32_t seed) {
  for (size_t i = 0; i < a.size(); ++i)
    require(a.data<uint32_t>()[i] == seed + i,
            "workspace overwrite/fence mismatch");
}

void rejected_requests(Stream stream, metal::CommandEncoder &enc,
                       bool populated) {
  auto try_workspace = [&](uint64_t id, const std::vector<Spec> &request) {
    return workspace::try_workspace(stream, enc, id, request);
  };
  require(!try_workspace(0, specs), "zero identity accepted");
  require(!try_workspace(identity, {}), "empty profile accepted");
  require(!try_workspace(identity, {{{0}, uint32}}), "zero dimension accepted");
  require(!try_workspace(identity, {{{-1}, uint32}}),
          "negative dimension accepted");
  require(!try_workspace(identity, {{{std::numeric_limits<int>::max(),
                                      std::numeric_limits<int>::max()},
                                     float32}}),
          "overflow/oversize accepted");
  require(!try_workspace(identity, {{{8 * 1024 * 1024 + 1}, uint8}}),
          "oversize bytes accepted");
  require(!try_workspace(identity, std::vector<Spec>(17, Spec{{1}, uint8})),
          "too many slots accepted");
  require(!try_workspace(identity, {{{1}, Dtype(Dtype::Val::uint8, 0)}}),
          "zero item size accepted");
  require(!try_workspace(identity, {{{4 * 1024 * 1024}, uint8},
                                    {{4 * 1024 * 1024 + 1}, uint8}}),
          "cumulative profile oversize accepted");
  if (populated) {
    require(!try_workspace(identity + 1, specs), "changed identity accepted");
    require(!try_workspace(identity, {{{64}, uint32}, {{32}, uint32}}),
            "changed shape accepted");
    require(!try_workspace(identity, {{{32}, int32}, {{32}, uint32}}),
            "changed dtype accepted");
    require(!try_workspace(identity, {{{32}, uint32}}),
            "changed slot count accepted");
  }
}

void boundary_regression() {
  const auto stream = new_stream(Device::gpu);
  auto &enc = metal::get_command_encoder(stream);
  Settle settle{enc};
  std::vector<Spec> maximum;
  for (int i = 0; i < 16; ++i)
    maximum.push_back({{131072}, uint32});
  const auto *slots = workspace::try_workspace(stream, enc, identity, maximum);
  require(slots && slots->size() == 16, "16-slot/8-MiB profile rejected");
  require(workspace::testing::has_profile(stream),
          "boundary profile not published");
  require(!workspace::try_workspace(stream, enc, identity,
                                    [&] {
                                      auto changed = maximum;
                                      changed.back().shape = {131073};
                                      return changed;
                                    }()),
          "profile over 8 MiB accepted");
  workspace::drain(stream);
  require(!workspace::testing::has_profile(stream),
          "boundary drain retained profile");

  const auto *exact = workspace::try_workspace(
      stream, enc, identity, std::vector<Spec>{{{8 * 1024 * 1024}, uint8}});
  require(exact && exact->size() == 1, "exact 8-MiB slot rejected");
  workspace::drain(stream);
}

void failure_atomicity() {
  const auto stream = new_stream(Device::gpu);
  auto &enc = metal::get_command_encoder(stream);
  Settle settle{enc};
  auto unrelated = allocation();
  fill(enc, stream, unrelated, 4321);

  for (int fail_after = 0; fail_after <= 2; ++fail_after) {
    for (int repeat = 0; repeat < 3; ++repeat) {
      std::vector<std::weak_ptr<array::Data>> weak;
      bool failed = false;
      try {
        workspace::testing::try_workspace_with_retention(
            stream, enc, failure_identity, specs,
            [&](metal::CommandEncoder &encoder, const array &owner) {
              for (const auto &slot : owner.inputs())
                weak.push_back(slot.data_shared_ptr());
              int pins = 0;
              for (const auto &slot : owner.inputs()) {
                if (fail_after == 0)
                  throw std::runtime_error("injected before pin");
                array single(Shape{}, uint8, nullptr, std::vector<array>{slot});
                encoder.retain_inputs(single);
                if (++pins == fail_after)
                  throw std::runtime_error("injected after pin");
              }
            });
      } catch (const std::runtime_error &) {
        failed = true;
        enc.synchronize();
      }
      require(failed, "injected publication failure was not propagated");
      require(!workspace::testing::has_profile(stream),
              "failed profile was published");
      require(weak.size() == specs.size(),
              "failure test did not observe prepared slots");
      require(std::all_of(weak.begin(), weak.end(),
                          [](const auto &p) { return p.expired(); }),
              "failed first publication retained storage");
    }
  }

  auto unrelated_out = allocation();
  read(enc, stream, unrelated, unrelated_out);
  enc.synchronize();
  check(unrelated_out, 4321);

  const auto &slots = acquire(stream, enc);
  const auto *storage = slots[0].data_shared_ptr().get();
  bool reuse_failed = false;
  try {
    workspace::testing::try_workspace_with_retention(
        stream, enc, identity, specs,
        [](metal::CommandEncoder &, const array &) {
          throw std::runtime_error("injected reuse failure");
        });
  } catch (const std::runtime_error &) {
    reuse_failed = true;
    enc.synchronize();
  }
  require(reuse_failed, "injected reuse failure was not propagated");
  require(workspace::testing::has_profile(stream),
          "failed reuse evicted profile");
  require(acquire(stream, enc)[0].data_shared_ptr().get() == storage,
          "failed reuse changed storage");
  workspace::drain(stream);
  std::fprintf(stderr, "[workspace-encoder] injected_publication_failures=9 "
                       "rollback_observable PASS\n");
}

void retention_failure_ordering_regression() {
  const auto producer = new_stream(Device::gpu);
  const auto consumer = new_stream(Device::gpu);
  auto &producer_enc = metal::get_command_encoder(producer);
  auto &consumer_enc = metal::get_command_encoder(consumer);
  auto producer_out = allocation();
  fill(producer_enc, producer, producer_out, 6789);

  Event fence(producer);
  constexpr uint64_t fence_value = 1;
  producer_enc.signal_event(fence, fence_value);
  consumer_enc.wait_event(fence, fence_value);

  std::vector<std::weak_ptr<array::Data>> weak;
  bool failed = false;
  try {
    workspace::testing::try_workspace_with_retention(
        consumer, consumer_enc, failure_identity, specs,
        [&](metal::CommandEncoder &encoder, const array &owner) {
          for (const auto &slot : owner.inputs())
            weak.push_back(slot.data_shared_ptr());
          array single(Shape{}, uint8, nullptr,
                       std::vector<array>{owner.inputs().front()});
          encoder.retain_inputs(single);
          throw std::runtime_error("injected cross-stream retention failure");
        });
  } catch (const std::runtime_error &) {
    failed = true;
  }

  // Match eval_impl cleanup: finalize the producer before the consumer wait.
  producer_enc.synchronize();
  consumer_enc.synchronize();

  require(failed, "cross-stream retention failure was not propagated");
  require(!workspace::testing::has_profile(consumer),
          "cross-stream failure published profile");
  require(weak.size() == specs.size(),
          "cross-stream failure did not observe prepared slots");
  require(std::all_of(weak.begin(), weak.end(),
                      [](const auto &p) { return p.expired(); }),
          "coordinated cleanup retained failed prepared storage");
  std::fprintf(stderr,
               "[workspace-encoder] cross_stream_retention_rollback PASS\n");
}

void clear_streams_reaping_regression() {
  using WeakData = std::weak_ptr<array::Data>;
  auto weak_data = [](const std::vector<array> &slots) {
    std::vector<WeakData> weak;
    for (const auto &slot : slots)
      weak.push_back(slot.data_shared_ptr());
    return weak;
  };
  auto all_live = [](const std::vector<WeakData> &weak) {
    return std::all_of(weak.begin(), weak.end(),
                       [](const auto &p) { return !p.expired(); });
  };
  auto all_expired = [](const std::vector<WeakData> &weak) {
    return std::all_of(weak.begin(), weak.end(),
                       [](const auto &p) { return p.expired(); });
  };

  std::vector<WeakData> previous;
  std::vector<Stream> streams;
  for (unsigned generation = 0; generation < 4; ++generation) {
    mlx::core::clear_streams();
    const auto stream = new_stream(Device::gpu);
    auto &enc = metal::get_command_encoder(stream);
    if (generation != 0)
      require(all_live(previous),
              "ordinary clear_streams released profile before reaping");
    const auto &slots = acquire(stream, enc);
    if (generation != 0)
      require(all_expired(previous), "stale profile survived next acquisition");
    fill(enc, stream, slots[0], 5000 + generation);
    enc.synchronize();
    previous = weak_data(slots);
    streams.push_back(stream);
  }

  const auto stale_drain_stream = streams.back();
  mlx::core::clear_streams();
  require(all_live(previous),
          "ordinary clear_streams released profile before stale drain");
  workspace::drain(stale_drain_stream);
  require(all_expired(previous), "stale drain retained profile");

  const auto combined_stream = new_stream(Device::gpu);
  std::vector<WeakData> combined;
  {
    auto &enc = metal::get_command_encoder(combined_stream);
    const auto &slots = acquire(combined_stream, enc);
    fill(enc, combined_stream, slots[0], 7000);
    enc.synchronize();
    combined = weak_data(slots);
  }
  mlx::core::clear_streams();
  require(all_live(combined),
          "ordinary clear_streams released profile before combined cleanup");
  workspace::clear_streams_and_workspaces();
  require(all_expired(combined), "combined cleanup retained stale profile");
  std::fprintf(stderr,
               "[workspace-encoder] clear_streams_generations=4 stale_reaps=3 "
               "stale_drain combined_cleanup PASS\n");
}

void encoder_regression() {
  const auto stream = new_stream(Device::gpu);
  auto &enc = metal::get_command_encoder(stream);
  std::vector<array> out;
  Settle settle{enc};
  rejected_requests(stream, enc, false);
  const auto &first = acquire(stream, enc);
  const auto *storage0 = first[0].data_shared_ptr().get();
  const auto *storage1 = first[1].data_shared_ptr().get();
  require(storage0 != storage1, "slots alias");
  rejected_requests(stream, enc, true);

  const auto other_stream = new_stream(Device::gpu);
  auto &other_enc = metal::get_command_encoder(other_stream);
  auto other_out = allocation();
  Settle settle_other{other_enc};
  const auto &other = acquire(other_stream, other_enc);
  require(other[0].data_shared_ptr().get() != storage0,
          "streams share mutable workspace");
  fill(other_enc, other_stream, other[0], 900);
  read(other_enc, other_stream, other[0], other_out);

  for (unsigned i = 0; i < 8; ++i) {
    const auto &slots = acquire(stream, enc);
    require(slots[0].data_shared_ptr().get() == storage0 &&
                slots[1].data_shared_ptr().get() == storage1,
            "allocation changed");
    out.push_back(allocation());
    fill(enc, stream, slots[0], 100 * i);
    auto intermediate = slots[1];
    read(enc, stream, slots[0], intermediate);
    read(enc, stream, slots[1], out.back());
    if (i % 2 == 0)
      enc.end_encoding();
    if (i == 2 || i == 5) {
      enc.end_encoding();
      enc.commit(nullptr, "workspace-test-no-wait");
    }
  }

  fill(enc, stream, acquire(stream, enc)[0], 12345);
  try {
    throw std::runtime_error("injected after encode");
  } catch (const std::runtime_error &) {
  }
  out.push_back(allocation());
  fill(enc, stream, acquire(stream, enc)[0], 800);
  read(enc, stream, acquire(stream, enc)[0], out.back());
  enc.synchronize();
  other_enc.synchronize();
  for (unsigned i = 0; i < out.size(); ++i)
    check(out[i], 100 * i);
  check(other_out, 900);
  require(acquire(stream, enc)[0].data_shared_ptr().get() == storage0,
          "sync evicted pool");

  workspace::drain(stream);
  require(!workspace::testing::has_profile(stream), "per-stream drain failed");
  require(workspace::testing::has_profile(other_stream),
          "drain erased another stream");
  workspace::drain(other_stream);
  std::fprintf(
      stderr,
      "[workspace-encoder] repeated=8 commits_without_wait=2 old_outputs=9 "
      "distinct_streams=2 rejects=13 drains=2 PASS\n");
}

array thread_drop_regression(uint32_t seed, int mode) {
  const auto stream = mode == 2 ? new_thread_unsafe_stream(Device::gpu)
                                : new_stream(Device::gpu);
  auto out = allocation();
  {
    auto &enc = metal::get_command_encoder(stream);
    try {
      const auto &slots = acquire(stream, enc);
      fill(enc, stream, slots[0], seed);
      read(enc, stream, slots[0], out);
    } catch (...) {
      try {
        enc.synchronize();
      } catch (...) {
      }
      throw;
    }
  }
  if (mode != 0) {
    workspace::clear_streams_and_workspaces();
    require(!workspace::testing::has_profile(stream),
            "combined clear retained profile");
  }
  return out;
}
} // namespace
#endif

extern "C" bool mlx_test_persistent_workspace_encoder() {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    boundary_regression();
    failure_atomicity();
    retention_failure_ordering_regression();
    encoder_regression();
    for (const auto [seed, mode] :
         {std::pair{777u, 0}, std::pair{888u, 1}, std::pair{999u, 2}}) {
      std::packaged_task<mlx::core::array()> task(
          [seed, mode] { return thread_drop_regression(seed, mode); });
      auto future = task.get_future();
      std::thread worker(std::move(task));
      worker.join();
      auto out = future.get();
      check(out, seed);
    }
    std::packaged_task<void()> clear_task(clear_streams_reaping_regression);
    auto clear_future = clear_task.get_future();
    std::thread clear_worker(std::move(clear_task));
    clear_worker.join();
    clear_future.get();
    std::fprintf(
        stderr, "[workspace-encoder] "
                "natural_local_explicit_local_explicit_global_teardown PASS\n");
    return true;
  } catch (const std::exception &e) {
    std::cerr << "workspace encoder test: " << e.what() << '\n';
    return false;
  }
#else
  return false;
#endif
}
