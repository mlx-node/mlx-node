R"METAL(
const int col = int(thread_position_in_grid.x);
const int row = int(thread_position_in_grid.y);
const int M = ids_shape[0];
const int V = dims[0];
const int D = dims[1];
if (col >= D || row >= M) return;

uint tok = 0u;
if constexpr (SIGNED_IDS) {
  const int raw = ids[row];
  if (raw < 0 || raw >= V) {
    out[size_t(row) * size_t(D) + size_t(col)] = 0.0f;
    return;
  }
  tok = uint(raw);
} else {
  tok = ids[row];
  if (tok >= uint(V)) {
    out[size_t(row) * size_t(D) + size_t(col)] = 0.0f;
    return;
  }
}

const size_t blk = size_t(col >> 5);
const size_t in_blk = size_t(col & 31);
const size_t row_bytes = size_t(D >> 5) * 34ul;
const device uint8_t* p =
    buf + size_t(byte_off[0]) + size_t(tok) * row_bytes + blk * 34ul;
const half d = *reinterpret_cast<const device half*>(p);
const int8_t q = reinterpret_cast<const device int8_t*>(p + 2)[in_blk];
out[size_t(row) * size_t(D) + size_t(col)] = float(d) * float(q);
)METAL"
