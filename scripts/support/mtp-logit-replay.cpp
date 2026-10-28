// Standalone diagnostic, deliberately outside the production build. It compares
// raw target logits under an identical forced prefix: autoregressive, chunked
// verification, and rejected speculative suffixes with recurrent rollback.
#include "llama.h"
#include "llama-cpp.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static std::vector<int> top(const std::vector<float> & logits, int count = 8) {
    std::vector<int> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + count, ids.end(), [&](int a, int b) {
        return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
    });
    ids.resize(count);
    return ids;
}

static json difference(const std::vector<float> & reference, const float * logits, int position) {
    const size_t size = reference.size();
    std::vector<float> candidate(logits, logits + size);
    const auto ta = top(reference), tb = top(candidate);
    double square = 0, max_abs = 0, sum_a = 0, sum_b = 0;
    for (size_t i = 0; i < size; ++i) {
        require(std::isfinite(reference[i]) && std::isfinite(candidate[i]), "nonfinite raw logits");
        const double delta = double(candidate[i]) - reference[i];
        square += delta*delta;
        max_abs = std::max(max_abs, std::abs(delta));
        sum_a += std::exp(double(reference[i]) - reference[ta[0]]);
        sum_b += std::exp(double(candidate[i]) - candidate[tb[0]]);
    }
    const double za = reference[ta[0]] + std::log(sum_a);
    const double zb = candidate[tb[0]] + std::log(sum_b);
    double tv = 0, max_logprob_delta = 0;
    for (size_t i = 0; i < size; ++i) {
        const double la = reference[i] - za, lb = candidate[i] - zb;
        tv += std::abs(std::exp(la) - std::exp(lb));
        max_logprob_delta = std::max(max_logprob_delta, std::abs(la-lb));
    }
    auto evidence = [](const std::vector<float> & values, const std::vector<int> & ids) {
        json rows = json::array();
        for (int id : ids) rows.push_back({{"id",id},{"logit",values[id]}});
        return json{{"argmax", ids[0]}, {"top_two_logit_gap", values[ids[0]]-values[ids[1]]},
                    {"top", rows}};
    };
    return {{"after_forced_token", position}, {"greedy_equal", ta[0] == tb[0]},
            {"raw_max_abs", max_abs}, {"raw_rms", std::sqrt(square/size)},
            {"log_partition_delta", zb-za}, {"max_logprob_delta",max_logprob_delta},
            {"softmax_total_variation", tv*.5},
            {"reference", evidence(reference,ta)}, {"candidate", evidence(candidate,tb)}};
}

static void decode(llama_context * context, llama_batch_ext * batch,
                   const std::vector<llama_token> & tokens, size_t begin, size_t count,
                   int position, bool output_all) {
    llama_batch_ext_clear(batch);
    for (size_t i=0; i<count; ++i) {
        int index = llama_batch_ext_add_token(batch, 0, tokens.at(begin+i));
        llama_pos pos = position + int(i);
        require(llama_batch_ext_set_pos(batch,index,&pos), "cannot set position");
        if (output_all || i+1 == count) {
            require(llama_batch_ext_set_output_logits(batch,index,true), "cannot mark logits");
        }
    }
    require(llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch) == 0, "forced decode failed");
}

static llama_model_ptr tiny_model() {
    const int dim=32, vocab=64;
    gguf_context_ptr metadata(gguf_init_empty());
    gguf_set_val_str(metadata.get(), "general.architecture", "qwen35");
    gguf_set_val_str(metadata.get(), "general.name", "mtp-rollback-diagnostic-fixture");
    auto u32=[&](const char * key, uint32_t value) {
        gguf_set_val_u32(metadata.get(), (std::string("qwen35.")+key).c_str(),value);
    };
    u32("context_length",512); u32("embedding_length",dim); u32("block_count",3);
    u32("feed_forward_length",64); u32("attention.head_count",2);
    u32("attention.head_count_kv",1); u32("rope.dimension_count",16);
    u32("nextn_predict_layers",1); u32("full_attention_interval",2);
    u32("ssm.conv_kernel",4); u32("ssm.inner_size",128);
    u32("ssm.state_size",128); u32("ssm.time_step_rank",1); u32("ssm.group_count",1);
    gguf_set_val_f32(metadata.get(),"qwen35.attention.layer_norm_rms_epsilon",1e-6f);
    const uint32_t sections[]={4,4,0,0};
    gguf_set_arr_data(metadata.get(),"qwen35.rope.dimension_sections",GGUF_TYPE_UINT32,sections,4);
    gguf_set_val_str(metadata.get(),"tokenizer.ggml.model","test");
    std::vector<std::string> strings;
    std::vector<const char *> names;
    std::vector<float> scores(vocab,0);
    for (int i=0;i<vocab;++i) strings.push_back("tok_"+std::to_string(i));
    for (auto & s:strings) names.push_back(s.c_str());
    gguf_set_arr_str(metadata.get(),"tokenizer.ggml.tokens",names.data(),names.size());
    gguf_set_arr_data(metadata.get(),"tokenizer.ggml.scores",GGUF_TYPE_FLOAT32,scores.data(),scores.size());
    auto initialize=[](ggml_tensor * tensor, void *) {
        require(tensor->type == GGML_TYPE_F32,"unexpected tiny fixture type");
        std::vector<float> values(ggml_nelements(tensor));
        uint32_t state=2166136261u;
        for (const char * p=tensor->name;*p;++p) state=(state^uint8_t(*p))*16777619u;
        const std::string name(tensor->name);
        for (auto & value:values) {
            state^=state<<13; state^=state>>17; state^=state<<5;
            value=(float(state&65535)/65535.f-.5f)*.12f;
        }
        if (name.find("norm")!=std::string::npos) std::fill(values.begin(),values.end(),1.f);
        if (name.find("ssm_a")!=std::string::npos) std::fill(values.begin(),values.end(),-1.f);
        if (name.find("ssm_dt.bias")!=std::string::npos) std::fill(values.begin(),values.end(),-1.f);
        ggml_backend_tensor_set(tensor,values.data(),0,values.size()*sizeof(float));
    };
    auto params=llama_model_default_params();
    ggml_backend_dev_t devices[]={nullptr};
    params.devices=devices; params.n_gpu_layers=0; params.load_mode=LLAMA_LOAD_MODE_NONE;
    params.use_extra_bufts=false; params.no_host=true; params.load_mtp=true;
    llama_model_ptr model(llama_model_init_from_user(metadata.get(),initialize,nullptr,params));
    require(bool(model),"cannot initialize nonzero tiny recurrent fixture");
    return model;
}

int main(int argc, char ** argv) {
    try {
        std::string input, output, model_path;
        int context_size=131072, layers=-1;
        uint64_t quota=13312;
        bool fixture=false;
        for (int i=1;i<argc;++i) {
            const std::string key(argv[i]);
            if (key=="--tiny") fixture=true;
            else if (key=="--input" && i+1<argc) input=argv[++i];
            else if (key=="--output" && i+1<argc) output=argv[++i];
            else if (key=="--model" && i+1<argc) model_path=argv[++i];
            else if (key=="--context" && i+1<argc) context_size=std::stoi(argv[++i]);
            else if (key=="--gpu-layers" && i+1<argc) layers=std::stoi(argv[++i]);
            else if (key=="--vram-mib" && i+1<argc) quota=std::stoull(argv[++i]);
            else throw std::runtime_error("unexpected/missing diagnostic argument");
        }
        require(!output.empty(),"--output is required");
        require(!std::ifstream(output).good(),"output already exists");
        std::vector<llama_token> prompt, forced;
        if (fixture) {
            require(input.empty() && model_path.empty(),"tiny mode must not read model data");
            context_size=512; layers=0;
            prompt={3,4,5,6,7,8,9};
            for (int i=0;i<64;++i) forced.push_back((i*17+11)%64);
        } else {
            require(!input.empty() && !model_path.empty(),"--input and --model are required");
            json data; std::ifstream(input)>>data;
            prompt=data.at("prompt_tokens").get<std::vector<llama_token>>();
            forced=data.at("response").at("tokens").get<std::vector<llama_token>>();
        }
        require(!prompt.empty() && !forced.empty(),"empty forced prefix");
        require(forced.size()<=512,"forced replay is limited to 512 tokens to bound diagnostic RAM");
        require(prompt.size()<=512 && prompt.size()+forced.size()+3<size_t(context_size),"diagnostic input bound exceeded");
        ggml_backend_load_all(); llama_backend_init();
        auto mparams=llama_model_default_params();
        mparams.n_gpu_layers=layers; mparams.load_mtp=true; mparams.load_mode=LLAMA_LOAD_MODE_NONE;
        mparams.gpu_exclusive=layers!=0; mparams.gpu_vram_limit_mib=quota;
        llama_model_ptr model=fixture ? tiny_model() : llama_model_ptr(llama_model_load_from_file(model_path.c_str(),mparams));
        require(bool(model),"cannot load diagnostic model");
        const int vocab=llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        for (auto t:prompt) require(t>=0 && t<vocab,"invalid prompt token");
        for (auto t:forced) require(t>=0 && t<vocab,"invalid forced token");
        std::vector<std::vector<float>> reference;
        json report={{"schema_version",1},{"tiny",fixture},{"context",context_size},
                     {"forced_token_count",forced.size()},
                     {"scope","raw target logits under identical prefixes; no quality-equivalence claim"},
                     {"modes",json::array()}};
        const std::vector<std::string> modes={"autoregressive","autoregressive_rs2","autoregressive_mtp_output",
                                              "chunk3","rollback2","rollback1"};
        for (const auto & mode:modes) {
            std::fprintf(stderr,"mtp-logit-replay: %s\n",mode.c_str());
            auto cp=llama_context_default_params();
            cp.n_ctx=context_size; cp.n_seq_max=1; cp.n_batch=512; cp.n_ubatch=128;
            cp.n_threads=cp.n_threads_batch=fixture?1:4;
            cp.n_rs_seq=mode=="autoregressive"?0:2;
            cp.type_k=cp.type_v=fixture?GGML_TYPE_F16:GGML_TYPE_Q8_0;
            cp.flash_attn_type=fixture?LLAMA_FLASH_ATTN_TYPE_DISABLED:LLAMA_FLASH_ATTN_TYPE_ENABLED;
            // Match the server's bounded verification output contract.
            cp.n_outputs_max=cp.n_outputs_max_per_seq=3;
            llama_context_ptr ctx(llama_init_from_model(model.get(),cp));
            require(bool(ctx),"cannot create sequential diagnostic context");
            const bool mtp_output=mode!="autoregressive" && mode!="autoregressive_rs2";
            if (mtp_output) llama_set_embeddings_nextn(ctx.get(),true,false);
            llama_batch_ext_ptr batch(llama_batch_ext_init(ctx.get()));
            require(bool(batch),"cannot allocate diagnostic batch");
            decode(ctx.get(),batch.get(),prompt,0,prompt.size(),0,false);
            json rows=json::array();
            auto observe=[&](size_t index,int row) {
                const float * logits=llama_get_logits_ith(ctx.get(),row);
                require(logits!=nullptr,"missing forced logits row");
                if (mode=="autoregressive") reference.emplace_back(logits,logits+vocab);
                else rows.push_back(difference(reference.at(index),logits,int(index)-1));
            };
            observe(0,-1);
            size_t done=0;
            while (done<forced.size()) {
                size_t count=1,accepted=1;
                if (mode=="chunk3") count=accepted=std::min(size_t(3),forced.size()-done);
                if (mode=="rollback2" || mode=="rollback1") {
                    count=std::min(size_t(3),forced.size()-done);
                    accepted=std::min(mode=="rollback1"?size_t(2):size_t(1),count);
                }
                auto tokens=forced;
                for (size_t j=accepted;j<count;++j) tokens[done+j]=(tokens[done+j]+1)%vocab;
                decode(ctx.get(),batch.get(),tokens,done,count,int(prompt.size()+done),true);
                for (size_t j=0;j<accepted;++j) observe(done+j+1,int(j));
                done+=accepted;
                if (count>accepted) {
                    require(llama_memory_seq_rm(llama_get_memory(ctx.get()),0,int(prompt.size()+done),-1),
                            "partial recurrent rollback failed");
                }
            }
            if (mode!="autoregressive") {
                double max_abs=0,max_tv=0;
                int unequal=0;
                for (const auto & row:rows) {
                    max_abs=std::max(max_abs,row.at("raw_max_abs").get<double>());
                    max_tv=std::max(max_tv,row.at("softmax_total_variation").get<double>());
                    if (!row.at("greedy_equal").get<bool>()) ++unequal;
                }
                report["modes"].push_back({{"name",mode},{"max_raw_abs",max_abs},
                                          {"max_softmax_total_variation",max_tv},
                                          {"argmax_differences",unequal},{"rows",rows}});
                std::fprintf(stderr,"mtp-logit-replay: %s max_abs=%.9g max_tv=%.9g argmax_changes=%d\n",
                             mode.c_str(),max_abs,max_tv,unequal);
            }
            std::ofstream(output)<<report.dump(2)<<'\n';
        }
        model.reset();
        llama_backend_free();
        return 0;
    } catch(const std::exception & error) {
        std::fprintf(stderr,"mtp-logit-replay: %s\n",error.what());
        return 1;
    }
}
