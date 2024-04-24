#pragma once

#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "sequence.h"

namespace py = pybind11;


std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           int64_t>
    prepare_decode(std::vector<SequenceMetadata*> seq_metadata_list,
                   int64_t block_size)
{
    int64_t num_seqs = seq_metadata_list.size();
    auto options = torch::TensorOptions().dtype(at::kLong);
    auto input_tokens = torch::empty({num_seqs}, options);
    auto input_positions = torch::empty({num_seqs}, options);
    auto slot_mapping = torch::empty({num_seqs}, options.dtype(at::kInt));

    auto input_tokens_ptr = input_tokens.data_ptr<int64_t>();
    auto input_positions_ptr = input_positions.data_ptr<int64_t>();
    auto slot_mapping_ptr = slot_mapping.data_ptr<int32_t>();
    int64_t max_context_len = 0;

    for (int64_t i = 0; i < num_seqs; i++) {
        auto seq_metadata = seq_metadata_list[i];
        auto &seq_data = seq_metadata->seq_data;
        int64_t generation_token = seq_data->get_last_token_id();
        input_tokens_ptr[i] = generation_token;

        int64_t seq_len = seq_data->get_len();
        int64_t position = seq_len - 1;
        input_positions_ptr[i] = position;

        max_context_len = std::max(max_context_len, seq_len);

        auto &block_table = seq_metadata->block_table;
        TORCH_CHECK(position / block_size < block_table.size());
        int64_t block_number = block_table[position / block_size];
        TORCH_CHECK(block_number >= 0);
        int64_t block_offset = position % block_size;
        int32_t slot = block_number * block_size + block_offset;
        slot_mapping_ptr[i] = slot;
    }

    return {input_tokens,
            input_positions,
            slot_mapping,
            max_context_len};
}
