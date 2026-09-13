#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace llama_tokenizer_vendor {
struct BpeVocabulary;
}  // namespace llama_tokenizer_vendor

namespace engine::models::yue2 {

class Yue2TextTokenizer {
public:
    explicit Yue2TextTokenizer(const std::filesystem::path & vocab_path);

    std::vector<int32_t> encode(const std::string & text) const;

    // The inverse. Needed because the model writes an ABC score before it sings,
    // and that score is the thing a user edits to change the music: without a way
    // back to text it is generated, used, and thrown away.
    std::string decode(const std::vector<int32_t> & ids, bool skip_special = true) const;

private:
    std::shared_ptr<llama_tokenizer_vendor::BpeVocabulary> vocab_;
};

}  // namespace engine::models::yue2
