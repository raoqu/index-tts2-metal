#pragma once

#include <filesystem>
#include <string>
#include <vector>

void DownloadModel(const std::vector<std::string>& urls,
                   const std::filesystem::path& target_path = {},
                   bool force_update = false);
