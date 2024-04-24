#include "sequence.h"
#include "model_runner.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<SequenceData>(m, "SequenceData")
        .def(py::init<std::vector<int> &>())
        .def("get_prompt_token_ids", &SequenceData::get_prompt_token_ids)
        .def("get_output_token_ids", &SequenceData::get_output_token_ids)
        .def("get_last_token_id", &SequenceData::get_last_token_id)
        .def("append_token_id", &SequenceData::append_token_id)
        .def("get_len", &SequenceData::get_len)
        .def("get_output_len", &SequenceData::get_output_len);

    py::class_<SequenceMetadata>(m, "SequenceMetadata")
        .def(py::init<std::string &, int64_t, bool, SequenceData *, py::object &, std::vector<int> &, int>())
        .def_readwrite("request_id", &SequenceMetadata::request_id)
        .def_readwrite("seq_id", &SequenceMetadata::seq_id)
        .def_readwrite("seq_data", &SequenceMetadata::seq_data)
        .def_readwrite("is_prompt", &SequenceMetadata::is_prompt)
        .def_readwrite("block_table", &SequenceMetadata::block_table)
        .def_readwrite("sampling_params", &SequenceMetadata::sampling_params);

    m.def("prepare_decode", &prepare_decode);
}
