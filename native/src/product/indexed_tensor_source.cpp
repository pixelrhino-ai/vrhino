#include "vrhino/product/converter.h"
#include "vrhino/error.h"
#include <limits>
#include <set>

namespace vrhino::product {
namespace {
void logical_shard(const std::string& s) {
    require(!s.empty() && s != "." && s != "..", "Invalid logical shard identity");
    for (unsigned char c : s)
        require(c >= 0x20 && c != '/' && c != '\\' && c != ':' && c != 0x7f,
                "Invalid logical shard identity");
}
}
IndexedTensorSource::IndexedTensorSource(const std::string& text,
    const std::map<std::string, IndexedSourceArtifact>& artifacts,
    const WorkProgressCallback& verification_progress) {
    try {
        const Json index=Json::parse(text, {64*1024*1024, 16, 1000000, 4000000, 65536});
        require(index.object().size()==2 && index.find("weight_map") && index.find("metadata"),
                "Invalid indexed source field set");
        const auto& metadata=index.at("metadata");
        require(metadata.object().size()==1 && metadata.at("total_size").integer()>=0,
                "Invalid indexed source total size");
        const auto& names=index.at("weight_map").object();
        require(!names.empty(), "Empty indexed source");
        std::set<std::string> required;
        for (const auto& [name, shard]:names) {
            require(!name.empty(), "Empty tensor identity");
            logical_shard(shard.string()); required.insert(shard.string());
        }
        require(required.size()==artifacts.size(), "Indexed shard catalog coverage mismatch");
        uint64_t total=0;
        for (const auto& shard:required) {
            const auto found=artifacts.find(shard);
            require(found!=artifacts.end(), "Missing indexed shard");
            const auto& artifact=found->second;
            require(std::filesystem::is_regular_file(std::filesystem::symlink_status(artifact.path)),
                    "Shard must be a regular immutable artifact, not a symlink");
            auto reader=std::make_unique<SafeTensorReader>(artifact.path);
            reader->verify_identity(artifact.size,artifact.sha256,verification_progress);
            for (const auto& [name, original]:reader->tensors()) {
                const auto owner=names.find(name);
                require(owner!=names.end() && owner->second.string()==shard,
                        "Shard/header tensor ownership mismatch");
                auto tensor=original; tensor.source_index=readers_.size();
                require(tensors_.emplace(name,tensor).second, "Duplicate shard tensor");
                require(tensor.byte_length<=std::numeric_limits<uint64_t>::max()-total,
                        "Indexed source byte count overflow");
                total+=tensor.byte_length;
            }
            readers_.push_back(std::move(reader));
        }
        require(tensors_.size()==names.size(), "Missing indexed tensor");
        require(total==static_cast<uint64_t>(metadata.at("total_size").integer()),
                "Indexed source total size mismatch");
    } catch (const ModelPackageError&) { throw; }
      catch (const std::exception& e) {
        throw ModelPackageError(ModelPackageErrorCode::PackageInvalid,e.what());
    }
}
void IndexedTensorSource::read_tensor(const SourceTensorDescriptor& tensor, uint64_t offset,
                                      void* destination, size_t bytes) const {
    const auto found=tensors_.find(tensor.name);
    require(found!=tensors_.end(), "Unknown indexed tensor descriptor");
    const auto& admitted=found->second;
    require(tensor.source_index==admitted.source_index && tensor.data_offset==admitted.data_offset &&
        tensor.byte_length==admitted.byte_length && tensor.dtype==admitted.dtype &&
        tensor.source_dtype==admitted.source_dtype && tensor.shape==admitted.shape,
        "Indexed tensor descriptor was modified");
    require(bytes==0 || destination!=nullptr, "Missing streaming destination");
    // data_offset is relative to the selected shard's data section. Only the
    // owning reader adds its header size, exactly once.
    readers_.at(admitted.source_index)->read_tensor(admitted,offset,destination,bytes);
}
} // namespace vrhino::product
