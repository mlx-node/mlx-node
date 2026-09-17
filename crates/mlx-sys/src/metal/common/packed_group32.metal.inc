R"metal(
// PackedNAXGroup32 / MLXFAST-WVEC8 from upstream 8981cef5, adapted
// to mixed Q4_K/Q5_K codes and F32 scale/min products (MIT notice retained).
template<int BITS>
struct PackedGroup32 {
  uint words[BITS];
  float scale, bias;
  void prefetch(const device uint8_t* src,float s,float b) thread {
    const device uint32_t* p=(const device uint32_t*)src;
    STEEL_PRAGMA_UNROLL
    for(int j=0;j<BITS;++j) words[j]=p[j];
    scale=s;bias=b;
  }
  template<typename T>
  void store(threadgroup T* dst) const thread {
    constexpr int V=sizeof(T)==4?4:8;
    threadgroup vec<T,V>* out=(threadgroup vec<T,V>*)dst;
    STEEL_PRAGMA_UNROLL
    for(int pack=0;pack<32/V;++pack) {
      vec<T,V> value;
      STEEL_PRAGMA_UNROLL
      for(int j=0;j<V;++j) {
        int bit=(pack*V+j)*BITS, word=bit/32, shift=bit%32;
        uint code=words[word]>>shift;
        if(shift+BITS>32) code|=words[word+1]<<(32-shift);
        value[j]=T(scale*float(code&((1u<<BITS)-1))+bias);
      }
      out[pack]=value;
    }
  }
};

)metal"
