#include <torch/extension.h>
#include <vector>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <map>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include "json.hpp"

using json = nlohmann::json;

// --- Helper for mmap ---
struct MappedFile {
    void* ptr;
    size_t size;
    int fd;

    MappedFile(const std::string& path) {
        fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Could not open file: " + path);
        }
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            throw std::runtime_error("Could not stat file: " + path);
        }
        size = sb.st_size;
        ptr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Could not mmap file: " + path);
        }
    }

    ~MappedFile() {
        if (ptr != MAP_FAILED) {
            munmap(ptr, size);
        }
        if (fd != -1) {
            close(fd);
        }
    }
};

// --- Loader Class ---

class StarLoader {
public:
    StarLoader(std::vector<std::string> file_paths, int batch_size, int queue_size)
        : file_paths_(file_paths), batch_size_(batch_size), queue_size_(queue_size), stop_requested_(false) {

        // Start background thread
        worker_thread_ = std::thread(&StarLoader::worker_loop, this);
    }

    ~StarLoader() {
        stop_requested_ = true;
        queue_not_full_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    // Python API: Get next batch
    std::map<std::string, torch::Tensor> next() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        // Wait until queue has item or stopped
        queue_not_empty_.wait(lock, [this] { return !queue_.empty() || stop_requested_ || finished_; });

        if (queue_.empty() && finished_) {
            // Signal StopIteration
            throw std::out_of_range("StopIteration");
        }

        if (queue_.empty() && stop_requested_) {
             throw std::out_of_range("StopIteration");
        }

        auto batch = queue_.front();
        queue_.pop();

        lock.unlock();
        queue_not_full_.notify_one();

        return batch;
    }

private:
    std::vector<std::string> file_paths_;
    int batch_size_;
    int queue_size_;

    std::queue<std::map<std::string, torch::Tensor>> queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_not_empty_;
    std::condition_variable queue_not_full_;

    std::thread worker_thread_;
    std::atomic<bool> stop_requested_;
    std::atomic<bool> finished_{false};

    void worker_loop() {
        for (const auto& path : file_paths_) {
            if (stop_requested_) break;

            try {
                process_file(path);
            } catch (const std::exception& e) {
                // In a real lib, we should log this or signal error to main thread
                // For now, print to stderr
                fprintf(stderr, "[StarLoader C++] Error processing %s: %s\n", path.c_str(), e.what());
            }
        }

        std::lock_guard<std::mutex> lock(queue_mutex_);
        finished_ = true;
        queue_not_empty_.notify_all();
    }

    void process_file(const std::string& path) {
        MappedFile mm(path);
        char* data_ptr = static_cast<char*>(mm.ptr);

        // 1. Read Header Size (u64)
        if (mm.size < 8) return;
        uint64_t header_len = *reinterpret_cast<uint64_t*>(data_ptr);

        // 2. Parse Header
        if (mm.size < 8 + header_len) return;
        std::string header_str(data_ptr + 8, header_len);
        json header = json::parse(header_str);

        size_t data_start = 8 + header_len;

        // 3. Create full tensors
        std::map<std::string, torch::Tensor> full_tensors;
        size_t num_rows = 0;

        for (auto& element : header.items()) {
            std::string name = element.key();
            auto meta = element.value();

            std::string dtype_str = meta["dtype"];
            std::vector<int64_t> shape = meta["shape"].get<std::vector<int64_t>>();
            size_t offset = meta["offset"];

            torch::ScalarType type = torch::kFloat32;
            bool cast_from_uint16 = false;

            // Map types safely
            if (dtype_str == "int64" || dtype_str == "<i8") {
                type = torch::kInt64;
            } else if (dtype_str == "int32" || dtype_str == "<i4") {
                type = torch::kInt32;
            } else if (dtype_str == "float32" || dtype_str == "<f4") {
                type = torch::kFloat32;
            } else if (dtype_str == "uint32" || dtype_str == "<u4") {
                 // PyTorch doesn't have native UInt32. Load as Int32 (same bits) and user can cast if needed.
                 // For tokens (0-4B), Int32 covers up to 2B. If tokens > 2B, it will wrap to negative in Int32.
                 // This is acceptable for indices usually as long as we treat them as bits,
                 // but ideally we should cast to Int64 if we want true values > 2B.
                 // For now, loading as Int32 is the most memory efficient 'safe' load for bits.
                 type = torch::kInt32;
            } else if (dtype_str == "uint16" || dtype_str == "<u2") {
                // PyTorch doesn't have UInt16. Load as Int16 (Short).
                // 0-32767 is fine. 32768-65535 becomes negative.
                // We will load as Int16, then cast to Int32 and correct the sign?
                // Or simply load as Int16 and let user handle it.
                // Better: Load as Int16, then .to(kInt32) and add 65536 where negative?
                // Simpler for performance: Load as Int16 (matches 2 bytes).
                // We'll mark flag to cast to Int32 and fix values if needed.
                // Actually, let's just use Int16 type for the blob view to match stride.
                type = torch::kInt16;
                cast_from_uint16 = true;
            } else {
                 fprintf(stderr, "Warning: Unknown dtype %s, defaulting to float32\n", dtype_str.c_str());
                 type = torch::kFloat32;
            }

            if (num_rows == 0 && shape.size() > 0) num_rows = shape[0];

            void* src_ptr = data_ptr + data_start + offset;

            auto options = torch::TensorOptions().dtype(type).device(torch::kCPU);

            // 1. View raw data
            torch::Tensor tensor_view = torch::from_blob(src_ptr, shape, options);

            // 2. Clone to own memory
            torch::Tensor tensor = tensor_view.clone();

            // 3. Post-process (Cast uint16 -> int32 properly)
            if (cast_from_uint16) {
                // Convert Int16 view to Int32
                tensor = tensor.to(torch::kInt32);
                // Fix negative values (reinterpret uint16 as int16)
                // x & 0xFFFF gives the unsigned value in int32 space
                tensor = tensor.bitwise_and(0xFFFF);
            }

            full_tensors[name] = tensor;
        }

        // 4. Batching Loop
        for (size_t start = 0; start < num_rows; start += batch_size_) {
            if (stop_requested_) break;

            size_t end = std::min(start + (size_t)batch_size_, num_rows);

            std::map<std::string, torch::Tensor> batch;
            for (auto& kv : full_tensors) {
                batch[kv.first] = kv.second.slice(0, start, end);
            }

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_not_full_.wait(lock, [this] { return queue_.size() < queue_size_ || stop_requested_; });

                if (stop_requested_) break;

                queue_.push(batch);
            }
            queue_not_empty_.notify_one();
        }
    }
};

// --- Bindings ---

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<StarLoader>(m, "StarLoader")
        .def(pybind11::init<std::vector<std::string>, int, int>())
        .def("next", &StarLoader::next);
}
