#pragma once

#include <cstdint>
#include <string>

namespace mit2_server {

int run_server(const std::string& host,
               uint16_t port,
               const std::string& model_bundle_dir,
               const std::string& voice_store_dir,
               uint32_t queue_size,
               uint32_t tts_concurrency,
               uint32_t clone_concurrency,
               bool web_enabled,
               const std::string& web_key,
               bool verbose = false);

}  // namespace mit2_server
