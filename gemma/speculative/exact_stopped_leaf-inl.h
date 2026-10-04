// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Private processed-KV snapshot for one terminal sibling. Include in the target
// namespace with attention.h and model/cache types already available.
#if defined(THIRD_PARTY_GEMMA_CPP_EXACT_STOPPED_LEAF_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_EXACT_STOPPED_LEAF_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_EXACT_STOPPED_LEAF_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_EXACT_STOPPED_LEAF_TOGGLE
#endif

class ExactStoppedLeafKV {
 public:
  // Read-only validation of the existing allocation. Preparation may allocate
  // private snapshot bytes, but never changes a cache or its metadata.
  static bool Supported(const ModelConfig& config, const KVCachePtr& kv,
                        size_t position) {
    namespace hn = hwy::HWY_NAMESPACE;
    if (config.model != Model::GEMMA3_1B || config.is_encoder_decoder ||
        config.num_mtp_layers != 0 || config.ple_dim != 0 ||
        config.layer_configs.empty() || config.layer_configs.size() > 26 ||
        config.num_layers != config.layer_configs.size() ||
        FloatsPerVector() != 8 || hn::Lanes(hn::ScalableTag<float>()) != 8 ||
        position >= 352 || kv.cache == nullptr || !kv.kv_cache.HasPtr())
      return false;
    const auto& owner = *kv.cache;
    const size_t layers = config.layer_configs.size();
    if (owner.num_layers != layers ||
        owner.layer_flat_offsets.size() != layers ||
        owner.layer_k_v_offsets.size() != layers ||
        owner.rounded_qkv_dims.size() != layers)
      return false;
    size_t flat_cols = 0, transposed_cols = 0;
    for (size_t i = 0; i < layers; ++i) {
      const auto& lc = config.layer_configs[i];
      if (lc.type != LayerAttentionType::kGemma || lc.heads != 4 ||
          lc.kv_heads != 1 || lc.qkv_dim != 256 || lc.kv_share_layer_idx >= 0 ||
          owner.layer_flat_offsets[i] != flat_cols ||
          owner.layer_k_v_offsets[i] != transposed_cols ||
          owner.rounded_qkv_dims[i] != 256)
        return false;
      flat_cols += 512;
      transposed_cols += 256;
    }
    const auto valid_flat = [&](const MatPtrT<KV_t>& view,
                                const MatPtrT<KV_t>& owned) {
      if (!view.HasPtr() || !owned.HasPtr() || view.Rows() <= position ||
          view.Cols() != flat_cols || view.Stride() < flat_cols ||
          view.Scale() != 1.0f || view.GetLayout() != MatPtr::Layout::kFlat ||
          !view.SameShape(owned) || view.Stride() != owned.Stride() ||
          view.Row(0) != owned.Row(0))
        return false;
      if (view.GetRowPtrs() != nullptr) {
        for (size_t r = 0; r < view.Rows(); ++r)
          if (view.GetRowPtrs()[r] != view.RowBytes(r)) return false;
      }
      return true;
    };
    // Native M1 writes zero transposed lanes through position+15 inclusive.
    const size_t needed = position + 16;
    const auto valid_transposed = [&](const MatPtrT<KV_t>& view,
                                      const MatPtrT<KV_t>& owned) {
      if (!view.HasPtr() || !owned.HasPtr() || view.Rows() == 0 ||
          !view.IsPacked() || view.Scale() != 1.0f ||
          view.GetLayout() != MatPtr::Layout::kFlat ||
          view.GetRowPtrs() != nullptr || view.Row(0) != owned.Row(0) ||
          view.Cols() == 0 || owned.Cols() == 0 ||
          view.Rows() > size_t(-1) / view.Cols() ||
          owned.Rows() > size_t(-1) / owned.Cols() ||
          view.Rows() * view.Cols() != owned.Rows() * owned.Cols())
        return false;
      if (view.Cols() == transposed_cols)
        return view.Rows() % 16 == 0 && view.Rows() >= needed;
      return view.Cols() == transposed_cols * 16 &&
             view.Rows() >= hwy::DivCeil(needed, size_t{16});
    };
    return owner.KOrVDefaultCols() == transposed_cols &&
           valid_flat(kv.kv_cache, owner.kv_cache) &&
           valid_transposed(kv.k_cache, owner.k_cache) &&
           valid_transposed(kv.v_cache, owner.v_cache);
  }

  bool Prepare(const ModelConfig& config, const KVCachePtr& kv,
               size_t position) {
    ready_ = false;
    captured_layers_ = 0;
    batch_attention_layers_ = 0;
    if (!Supported(config, kv, position)) return false;
    saved_.resize(kv.kv_cache.Cols());
    config_ = &config;
    owner_ = kv.cache;
    position_ = position;
    ready_ = true;
    return true;
  }

  bool CaptureLayer(size_t layer, const KVCachePtr& kv) {
    if (!ready_ || kv.cache != owner_ || layer != captured_layers_ ||
        layer >= config_->layer_configs.size())
      return false;
    const auto& flat = kv.kv_cache;
    const auto& owned = owner_->kv_cache;
    if (!flat.HasPtr() || !flat.SameShape(owned) ||
        flat.Stride() != owned.Stride() || flat.Row(0) != owned.Row(0) ||
        flat.Scale() != 1.0f || flat.GetLayout() != MatPtr::Layout::kFlat ||
        position_ >= flat.Rows() ||
        (flat.GetRowPtrs() != nullptr &&
         flat.GetRowPtrs()[position_] != flat.RowBytes(position_)))
      return false;
    const size_t offset = owner_->layer_flat_offsets[layer];
    hwy::CopyBytes(kv.kv_cache.Row(position_) + offset, saved_.data() + offset,
                   512 * sizeof(KV_t));
    ++captured_layers_;
    return true;
  }

  bool Restore(KVCachePtr& kv) const {
    if (!ready_ || kv.cache != owner_ ||
        captured_layers_ != config_->layer_configs.size() ||
        !Supported(*config_, kv, position_))
      return false;
    hwy::CopyBytes(saved_.data(), kv.kv_cache.Row(position_),
                   saved_.size() * sizeof(KV_t));
    // Raw flat row bytes already contain the native BF16 norm/RoPE result.
    // Address the contiguous allocation directly: native preparation may have
    // reshaped the K/V view after Prepare, without changing its storage.
    const size_t tile_cols = owner_->KOrVDefaultCols() * 16;
    KV_t* const k_row = kv.k_cache.Row(0) + (position_ / 16) * tile_cols;
    KV_t* const v_row = kv.v_cache.Row(0) + (position_ / 16) * tile_cols;
    for (size_t layer = 0; layer < config_->layer_configs.size(); ++layer) {
      const KV_t* const source =
          saved_.data() + owner_->layer_flat_offsets[layer];
      TransposeKVCacheRow(source,
                          k_row + owner_->KOffset(layer, 0, 8, position_),
                          v_row + owner_->VOffset(layer, 0, 8, position_), 256);
    }
    return true;
  }

  void NoteBatchAttention() { ++batch_attention_layers_; }
  size_t BatchAttentionLayers() const { return batch_attention_layers_; }

  size_t SnapshotBytes() const { return saved_.size() * sizeof(KV_t); }
  size_t Position() const { return position_; }

 private:
  const ModelConfig* config_ = nullptr;
  const KVCache* owner_ = nullptr;
  hwy::AlignedVector<KV_t> saved_;
  size_t position_ = 0;
  size_t captured_layers_ = 0;
  size_t batch_attention_layers_ = 0;
  bool ready_ = false;
};
#endif
