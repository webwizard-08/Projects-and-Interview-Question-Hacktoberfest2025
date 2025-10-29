#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <iomanip>

class FileCrypto {
private:
    std::string key;

    // XOR encryption/decryption (symmetric)
    std::vector<char> xorEncryptDecrypt(const std::vector<char>& data) {
        std::vector<char> result(data.size());
        size_t keyLen = key.length();
        
        for (size_t i = 0; i < data.size(); ++i) {
            result[i] = data[i] ^ key[i % keyLen];
        }
        return result;
    }

    // Caesar cipher encryption
    std::vector<char> caesarEncrypt(const std::vector<char>& data, int shift) {
        std::vector<char> result(data.size());
        
        for (size_t i = 0; i < data.size(); ++i) {
            result[i] = static_cast<char>((static_cast<unsigned char>(data[i]) + shift) % 256);
        }
        return result;
    }

    // Caesar cipher decryption
    std::vector<char> caesarDecrypt(const std::vector<char>& data, int shift) {
        std::vector<char> result(data.size());
        
        for (size_t i = 0; i < data.size(); ++i) {
            result[i] = static_cast<char>((static_cast<unsigned char>(data[i]) - shift + 256) % 256);
        }
        return result;
    }

    // Read file into vector
    bool readFile(const std::string& filepath, std::vector<char>& data) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Cannot open file '" << filepath << "' for reading\n";
            return false;
        }

        file.seekg(0, std::ios::end);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        data.resize(size);
        if (!file.read(data.data(), size)) {
            std::cerr << "Error: Failed to read file '" << filepath << "'\n";
            return false;
        }

        return true;
    }

    // Write vector to file
    bool writeFile(const std::string& filepath, const std::vector<char>& data) {
        std::ofstream file(filepath, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Cannot open file '" << filepath << "' for writing\n";
            return false;
        }

        if (!file.write(data.data(), data.size())) {
            std::cerr << "Error: Failed to write to file '" << filepath << "'\n";
            return false;
        }

        return true;
    }

public:
    FileCrypto(const std::string& encryptionKey) : key(encryptionKey) {
        if (key.empty()) {
            key = "default_key_12345";
        }
    }

    // Encrypt file using XOR
    bool encryptFileXOR(const std::string& inputPath, const std::string& outputPath) {
        std::vector<char> data;
        if (!readFile(inputPath, data)) return false;

        std::vector<char> encrypted = xorEncryptDecrypt(data);
        
        if (!writeFile(outputPath, encrypted)) return false;

        std::cout << "✓ File encrypted successfully (XOR): " << outputPath << "\n";
        std::cout << "  Original size: " << data.size() << " bytes\n";
        return true;
    }

    // Decrypt file using XOR (same as encrypt for XOR)
    bool decryptFileXOR(const std::string& inputPath, const std::string& outputPath) {
        return encryptFileXOR(inputPath, outputPath);
    }

    // Encrypt file using Caesar cipher
    bool encryptFileCaesar(const std::string& inputPath, const std::string& outputPath, int shift) {
        std::vector<char> data;
        if (!readFile(inputPath, data)) return false;

        std::vector<char> encrypted = caesarEncrypt(data, shift);
        
        if (!writeFile(outputPath, encrypted)) return false;

        std::cout << "✓ File encrypted successfully (Caesar, shift=" << shift << "): " << outputPath << "\n";
        std::cout << "  Original size: " << data.size() << " bytes\n";
        return true;
    }

    // Decrypt file using Caesar cipher
    bool decryptFileCaesar(const std::string& inputPath, const std::string& outputPath, int shift) {
        std::vector<char> data;
        if (!readFile(inputPath, data)) return false;

        std::vector<char> decrypted = caesarDecrypt(data, shift);
        
        if (!writeFile(outputPath, decrypted)) return false;

        std::cout << "✓ File decrypted successfully (Caesar, shift=" << shift << "): " << outputPath << "\n";
        std::cout << "  Original size: " << data.size() << " bytes\n";
        return true;
    }
};

void printUsage(const char* programName) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║     File Encryption-Decryption Tool v1.0            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << programName << " <mode> <method> <input_file> <output_file> [options]\n\n";
    std::cout << "Modes:\n";
    std::cout << "  encrypt    Encrypt the input file\n";
    std::cout << "  decrypt    Decrypt the input file\n\n";
    std::cout << "Methods:\n";
    std::cout << "  xor        XOR encryption (requires key)\n";
    std::cout << "  caesar     Caesar cipher (requires shift value)\n\n";
    std::cout << "Options:\n";
    std::cout << "  -k <key>   Encryption key (for XOR method)\n";
    std::cout << "  -s <num>   Shift value (for Caesar method, default: 3)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << " encrypt xor secret.txt secret.enc -k mypassword\n";
    std::cout << "  " << programName << " decrypt xor secret.enc secret.txt -k mypassword\n";
    std::cout << "  " << programName << " encrypt caesar data.txt data.enc -s 13\n";
    std::cout << "  " << programName << " decrypt caesar data.enc data.txt -s 13\n\n";
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        printUsage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    std::string method = argv[2];
    std::string inputFile = argv[3];
    std::string outputFile = argv[4];

    // Parse optional arguments
    std::string key = "";
    int shift = 3;

    for (int i = 5; i < argc; i += 2) {
        if (i + 1 < argc) {
            std::string arg = argv[i];
            if (arg == "-k") {
                key = argv[i + 1];
            } else if (arg == "-s") {
                try {
                    shift = std::stoi(argv[i + 1]);
                } catch (...) {
                    std::cerr << "Error: Invalid shift value\n";
                    return 1;
                }
            }
        }
    }

    // Validate mode
    if (mode != "encrypt" && mode != "decrypt") {
        std::cerr << "Error: Invalid mode. Use 'encrypt' or 'decrypt'\n";
        printUsage(argv[0]);
        return 1;
    }

    // Validate method
    if (method != "xor" && method != "caesar") {
        std::cerr << "Error: Invalid method. Use 'xor' or 'caesar'\n";
        printUsage(argv[0]);
        return 1;
    }

    // Check for key if using XOR
    if (method == "xor" && key.empty()) {
        std::cerr << "Error: XOR method requires a key (-k option)\n";
        return 1;
    }

    FileCrypto crypto(key);

    bool success = false;

    if (method == "xor") {
        if (mode == "encrypt") {
            success = crypto.encryptFileXOR(inputFile, outputFile);
        } else {
            success = crypto.decryptFileXOR(inputFile, outputFile);
        }
    } else if (method == "caesar") {
        if (mode == "encrypt") {
            success = crypto.encryptFileCaesar(inputFile, outputFile, shift);
        } else {
            success = crypto.decryptFileCaesar(inputFile, outputFile, shift);
        }
    }

    return success ? 0 : 1;
}