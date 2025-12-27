#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <filesystem>

/*
 * This utility inspects a binary dataset file and verifies its integrity.
 * It reads the header, checks the file size, verifies padding bytes,
 * and dumps some sample data for inspection.  This dataset format is assumed
 * to have a specific binary layout as defined by the DatasetHeader struct and
 * is intended to hold the data tensors and labels for a machine learning task.
 * 
 * It can be compiled using:
 * g++ -std=c++23 ./src/inspect_dataet.cpp -o bin/inspect_dataset
 * and run as:
 * ./bin/inspect_dataset <dataset.bin>
 */

struct DatasetHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_samples;
    uint32_t logical_channels;
    uint32_t physical_channels;
    uint32_t timesteps;
};

static constexpr uint32_t EXPECTED_MAGIC = 0x44544E49; // 'INTD'

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: inspect_dataset <dataset.bin>\n";
        return 1;
    }

    const std::filesystem::path path = argv[1];
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << path << "\n";
        return 1;
    }

    // ----------------------------
    // Read header
    // ----------------------------
    DatasetHeader hdr{};
    file.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!file) {
        std::cerr << "Failed to read header\n";
        return 1;
    }

    std::cout << "Header:\n";
    std::cout << "  magic            = 0x" << std::hex << hdr.magic << std::dec << "\n";
    std::cout << "  version          = " << hdr.version << "\n";
    std::cout << "  num_samples      = " << hdr.num_samples << "\n";
    std::cout << "  logical_channels = " << hdr.logical_channels << "\n";
    std::cout << "  physical_channels= " << hdr.physical_channels << "\n";
    std::cout << "  timesteps        = " << hdr.timesteps << "\n";

    if (hdr.magic != EXPECTED_MAGIC) {
        std::cerr << "ERROR: bad magic\n";
        return 1;
    }

    // ----------------------------
    // Compute expected file size
    // ----------------------------
    const uint64_t samples_bytes =
        uint64_t(hdr.num_samples) *
        hdr.physical_channels *
        hdr.timesteps *
        sizeof(int8_t);

    const uint64_t labels_bytes =
        uint64_t(hdr.num_samples) * sizeof(uint8_t);

    const uint64_t expected_size =
        sizeof(DatasetHeader) + samples_bytes + labels_bytes;

    const uint64_t actual_size = std::filesystem::file_size(path);

    std::cout << "Expected file size = " << expected_size << " bytes\n";
    std::cout << "Actual file size   = " << actual_size   << " bytes\n";

    if (expected_size != actual_size) {
        std::cerr << "ERROR: file size mismatch\n";
        return 1;
    }

    // ----------------------------
    // Read first sample
    // ----------------------------
    const size_t sample_stride =
        hdr.physical_channels * hdr.timesteps;

    std::vector<int8_t> sample(sample_stride);
    file.read(reinterpret_cast<char*>(sample.data()), sample.size());
    if (!file) {
        std::cerr << "Failed to read first sample\n";
        return 1;
    }

    // ----------------------------
    // Check padded channels are zero
    // ----------------------------
    bool padding_ok = true;
    for (uint32_t c = hdr.logical_channels; c < hdr.physical_channels; ++c) {
        for (uint32_t t = 0; t < hdr.timesteps; ++t) {
            int8_t v = sample[c * hdr.timesteps + t];
            if (v != 0) {
                padding_ok = false;
                std::cerr << "Non-zero padding at channel "
                          << c << ", timestep " << t
                          << ": " << int(v) << "\n";
                break;
            }
        }
    }

    std::cout << "Padding check: " << (padding_ok ? "OK" : "FAILED") << "\n";

    // ----------------------------
    // Dump first channel, first 16 timesteps
    // ----------------------------
    std::cout << "First sample, channel 0, timesteps [0..15]:\n  ";
    for (int i = 0; i < 16; ++i) {
        int8_t v = sample[i];
        std::cout << std::setw(4) << int(v) << " ";
    }
    std::cout << "\n";

    // ----------------------------
    // Read first label
    // ----------------------------
    file.seekg(sizeof(DatasetHeader) + samples_bytes);
    uint8_t label = 0;
    file.read(reinterpret_cast<char*>(&label), sizeof(label));

    std::cout << "First label = " << unsigned(label) << "\n";

    std::cout << "Dataset inspection PASSED\n";
    return 0;
}