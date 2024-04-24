#pragma once
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

enum class SequenceStatus
{
    WAITING,
    RUNNING,
    SWAPPED,
    FINISHED_STOPPED,
    FINISHED_LENGTH_CAPPED,
    FINISHED_ABORTED,
    FINISHED_IGNORED,
    FINISHED_PREEMPTED
};

bool is_finished(SequenceStatus status)
{
    return status >= SequenceStatus::FINISHED_STOPPED;
}

std::string get_finished_reason(SequenceStatus status)
{
    switch (status)
    {
    case SequenceStatus::FINISHED_STOPPED:
        return "stop";
    case SequenceStatus::FINISHED_LENGTH_CAPPED:
        return "length";
    case SequenceStatus::FINISHED_ABORTED:
        return "abort";
    case SequenceStatus::FINISHED_IGNORED:
        return "length";
    case SequenceStatus::FINISHED_PREEMPTED:
        return "preempted";
    default:
        return "";
    }
}

struct SequenceData
{
    SequenceData(std::vector<int> &prompt_token_ids)
        : _prompt_token_ids(prompt_token_ids) {}
    void append_token_id(int token_id)
    {
        _output_token_ids.push_back(token_id);
    }
    int get_len()
    {
        return _prompt_token_ids.size() + _output_token_ids.size();
    }
    int get_prompt_len()
    {
        return _prompt_token_ids.size();
    }
    int get_output_len()
    {
        return _output_token_ids.size();
    }
    auto get_prompt_token_ids()
    {
        return _prompt_token_ids;
    }
    auto get_output_token_ids()
    {
        return _output_token_ids;
    }
    int get_last_token_id()
    {
        if (not _output_token_ids.empty())
            return _output_token_ids.back();
        return _prompt_token_ids.back();
    }
    std::vector<int> _prompt_token_ids, _output_token_ids;
};

struct Sequence
{
    Sequence(int seq_id,
             std::string prompt,
             std::vector<int> prompt_token_ids,
             int vocab_size,
             int block_size,
             std::string request_id,
             py::object sampling_params,
             double arrival_time,
             std::vector<int> stop_token_ids,
             int max_tokens,
             bool ignore_eos,
             int sampling_type
             )
        : _seq_id(seq_id),
          _prompt(prompt),
          _data(SequenceData(prompt_token_ids)),
          _vocab_size(vocab_size),
          _status(SequenceStatus::WAITING),
          _block_size(block_size),
          _request_id(request_id),
          _sampling_params(sampling_params),
          _arrival_time(arrival_time),
          _stop_token_ids(stop_token_ids),
          _max_tokens(max_tokens),
          _ignore_eos(ignore_eos),
          _sampling_type(sampling_type) {}
    int num_prompt_blocks()
    {
        return (get_prompt_len() + _block_size - 1) / _block_size;
    }
    int get_num_blocks()
    {
        return (_data.get_len() + _block_size - 1) / _block_size;
    }
    void append_token_id(int token_id)
    {
        _data.append_token_id(token_id);
    }
    int get_len()
    {
        return _data.get_len();
    }
    int get_prompt_len()
    {
        return _data.get_prompt_len();
    }
    int get_output_len()
    {
        return _data.get_output_len();
    }
    int get_last_token_id()
    {
        return _data.get_last_token_id();
    }
    int64_t get_output_bin_count_ptr()
    {
        if (_output_bin_count.numel() == 0)
            _output_bin_count = torch::zeros({_vocab_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        return reinterpret_cast<int64_t>(_output_bin_count.data_ptr());
    }
    bool is_finished()
    {
        return ::is_finished(_status);
    }
    int _seq_id;
    std::string _prompt;
    SequenceData _data;
    int _vocab_size;
    torch::Tensor _output_bin_count;
    SequenceStatus _status;
    int _block_size;
    std::string _request_id;
    py::object _sampling_params;
    double _arrival_time;

    // for check_stop
    std::vector<int> _stop_token_ids;
    int _max_tokens;
    bool _ignore_eos;
    int _sampling_type;
};


struct SequenceMetadata
{
    SequenceMetadata(std::string &request_id,
                     int64_t seq_id,
                     bool is_prompt,
                     SequenceData *seq_data,
                     py::object &sampling_params,
                     std::vector<int> &block_table,
                     int sampling_type)
                : request_id(request_id),
                  seq_id(seq_id),
                  is_prompt(is_prompt),
                  seq_data(seq_data),
                  sampling_params(sampling_params),
                  block_table(block_table),
                  sampling_type(sampling_type) {}

    std::string request_id;
    int64_t seq_id;
    bool is_prompt;
    SequenceData *seq_data;
    py::object sampling_params;
    std::vector<int> block_table;
    int sampling_type;
};
