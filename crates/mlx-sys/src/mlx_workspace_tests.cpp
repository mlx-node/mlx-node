// Bounded real-encoder regression, invoked only by explicit integration tests.
#include "mlx_common.h"
#include "mlx/allocator.h"
#include <future>
#include <limits>
#include <thread>
#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/metal/device.h"
namespace mlx::core::metal {
struct PersistentWorkspaceTestAccess {
  static void failure_atomicity(CommandEncoder& enc) {
    if(enc.workspace_identity_ || !enc.workspace_.empty())
      throw std::runtime_error("publication fault test requires a fresh encoder");
    array sentinel({1},uint32,nullptr,{});sentinel.set_data(allocator::malloc(sentinel.nbytes()));
    enc.pending_retained_.insert(sentinel.data_shared_ptr());
    for(int fail_after=0;fail_after<=2;++fail_after) {
      for(int repeat=0;repeat<3;++repeat) {
        std::vector<array> prepared;std::vector<std::weak_ptr<array::Data>> weak;
        for(int i=0;i<2;++i){array a({32},uint32,nullptr,{});a.set_data(allocator::malloc(a.nbytes()));weak.push_back(a.data_shared_ptr());prepared.push_back(std::move(a));}
        int pins=0;bool failed=false;
        try {
          enc.publish_workspace(17,std::move(prepared),[&](const auto& data){
            if(fail_after==0)throw std::runtime_error("injected before pin");
            enc.pending_retained_.insert(data);
            if(++pins==fail_after)throw std::runtime_error("injected after pin");
          });
        } catch(const std::runtime_error&){failed=true;}
        if(!failed || enc.workspace_identity_ || !enc.workspace_.empty() ||
           enc.pending_retained_.size()!=1 ||
           !enc.pending_retained_.count(sentinel.data_shared_ptr()) ||
           !std::all_of(weak.begin(),weak.end(),[](const auto& p){return p.expired();}))
          throw std::runtime_error("failed first publication retained or published storage");
      }
    }
    enc.pending_retained_.erase(sentinel.data_shared_ptr());
    std::fprintf(stderr,"[workspace-encoder] injected_publication_failures=9 rollback_exact PASS\n");
  }
};
}
namespace {
using namespace mlx::core;
using Spec = metal::CommandEncoder::WorkspaceSpec;
constexpr uint64_t identity = 0x57534b5350414345ULL;
const std::vector<Spec> specs{{{32},uint32},{{32},uint32}};
void require(bool value,const char* message) {
  if(!value) throw std::runtime_error(message);
}
struct Settle {
  metal::CommandEncoder& enc;
  ~Settle(){try{enc.synchronize();}catch(...) {}}
};
array allocation() {
  array a({32},uint32,nullptr,{});a.set_data(allocator::malloc(a.nbytes()));return a;
}
const std::vector<array>& acquire(metal::CommandEncoder& enc) {
  const auto* slots=enc.try_workspace(identity,specs);
  require(slots && slots->size()==2,"workspace acquisition failed");return *slots;
}
void kernel(metal::CommandEncoder& enc,Stream s,const char* name) {
  auto& d=metal::device(s.device);
  auto* lib=d.get_library("persistent_workspace_test_v1",[]{return R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void workspace_fill(device uint* a[[buffer(0)]],constant uint& seed[[buffer(1)]],uint i[[thread_position_in_grid]]) {a[i]=seed+i;}
kernel void workspace_read(device const uint* a[[buffer(0)]],device uint* b[[buffer(1)]],uint i[[thread_position_in_grid]]) {b[i]=a[i];}
)METAL";});
  enc.set_compute_pipeline_state(d.get_kernel(name,lib));
}
void fill(metal::CommandEncoder& enc,Stream s,array a,uint32_t value) {
  kernel(enc,s,"workspace_fill");enc.set_output_array(a,0);enc.set_bytes(value,1);
  enc.dispatch_threads(MTL::Size(32,1,1),MTL::Size(32,1,1));
}
void read(metal::CommandEncoder& enc,Stream s,const array& a,array& out) {
  kernel(enc,s,"workspace_read");enc.set_input_array(a,0);enc.set_output_array(out,1);
  enc.dispatch_threads(MTL::Size(32,1,1),MTL::Size(32,1,1));
}
void check(const array& a,uint32_t seed) {
  for(size_t i=0;i<a.size();++i)require(a.data<uint32_t>()[i]==seed+i,"workspace overwrite/fence mismatch");
}
void rejected_requests(metal::CommandEncoder& enc,bool populated) {
  require(!enc.try_workspace(0,specs),"zero identity accepted");
  require(!enc.try_workspace(identity,{}),"empty profile accepted");
  require(!enc.try_workspace(identity,{{{0},uint32}}),"zero dimension accepted");
  require(!enc.try_workspace(identity,{{{-1},uint32}}),"negative dimension accepted");
  require(!enc.try_workspace(identity,{{{std::numeric_limits<int>::max(),std::numeric_limits<int>::max()},float32}}),"overflow/oversize accepted");
  require(!enc.try_workspace(identity,{{{8*1024*1024+1},uint8}}),"oversize bytes accepted");
  require(!enc.try_workspace(identity,std::vector<Spec>(17,Spec{{1},uint8})),"too many slots accepted");
  if(populated) {
    require(!enc.try_workspace(identity+1,specs),"changed identity accepted");
    require(!enc.try_workspace(identity,{{{64},uint32},{{32},uint32}}),"changed shape accepted");
    require(!enc.try_workspace(identity,{{{32},int32},{{32},uint32}}),"changed dtype accepted");
    require(!enc.try_workspace(identity,{{{32},uint32}}),"changed slot count accepted");
  }
  {auto concurrent=enc.start_concurrent();require(!enc.try_workspace(identity,specs),"concurrent pool reuse accepted");}
}
void encoder_regression() {
  const auto s=new_stream(Device::gpu);auto& enc=metal::get_command_encoder(s);
  // Keep public destinations alive longer than the final error/settlement guard.
  std::vector<array> out;Settle settle{enc};
  metal::PersistentWorkspaceTestAccess::failure_atomicity(enc);
  rejected_requests(enc,false);
  const auto& first=acquire(enc);
  const auto* storage0=first[0].data_shared_ptr().get();const auto* storage1=first[1].data_shared_ptr().get();
  require(storage0!=storage1,"slots alias");rejected_requests(enc,true);
  const auto s2=new_stream(Device::gpu);auto& enc2=metal::get_command_encoder(s2);auto other_out=allocation();Settle settle2{enc2};
  const auto& other=acquire(enc2);
  require(other[0].data_shared_ptr().get()!=storage0,"streams share mutable workspace");
  fill(enc2,s2,other[0],900);read(enc2,s2,other[0],other_out);
  for(unsigned i=0;i<8;++i) {
    const auto& slots=acquire(enc);
    require(slots[0].data_shared_ptr().get()==storage0 && slots[1].data_shared_ptr().get()==storage1,"allocation changed");
    out.push_back(allocation());
    // Two scratch slots and public output use only ordinary tracked bindings.
    fill(enc,s,slots[0],100*i);auto intermediate=slots[1];
    read(enc,s,slots[0],intermediate);read(enc,s,slots[1],out.back());
    if(i%2==0) enc.end_encoding();
    if(i==2 || i==5){enc.end_encoding();enc.commit(nullptr,"workspace-test-no-wait");}
  }
  // Simulated CPU failure after encoding must not expose or reset the pool.
  fill(enc,s,acquire(enc)[0],12345);
  try {throw std::runtime_error("injected after encode");}
  catch(const std::runtime_error&) {}
  out.push_back(allocation());fill(enc,s,acquire(enc)[0],800);read(enc,s,acquire(enc)[0],out.back());
  enc.synchronize();enc2.synchronize();
  for(unsigned i=0;i<out.size();++i)check(out[i],100*i);
  check(other_out,900);require(acquire(enc)[0].data_shared_ptr().get()==storage0,"sync evicted pool");
  std::fprintf(stderr,"[workspace-encoder] repeated=8 commits_without_wait=2 old_outputs=9 distinct_streams=2 rejects=15 PASS\n");
}
array thread_drop_regression() {
  const auto s=new_stream(Device::gpu);auto& enc=metal::get_command_encoder(s);
  auto out=allocation();
  try {const auto& slots=acquire(enc);fill(enc,s,slots[0],777);read(enc,s,slots[0],out);}
  catch(...){try{enc.synchronize();}catch(...){}throw;}
  // Clear only this worker thread's encoders. Their destructor synchronizes;
  // no caller scratch handle remains, and public output must survive teardown.
  clear_streams();return out;
}
} // namespace
#endif
extern "C" bool mlx_test_persistent_workspace_encoder() {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    encoder_regression();
    std::packaged_task<mlx::core::array()> task(thread_drop_regression);
    auto future=task.get_future();std::thread worker(std::move(task));worker.join();
    auto out=future.get();check(out,777);
    std::fprintf(stderr,"[workspace-encoder] thread_drop_and_retention PASS\n");return true;
  } catch(const std::exception& e) {std::cerr<<"workspace encoder test: "<<e.what()<<'\n';return false;}
#else
  return false;
#endif
}
