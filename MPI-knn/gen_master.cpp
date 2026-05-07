#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <cstdlib>

int main(int argc, char* argv[]) {
    // 1. Parse command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " [number of datapoints] [dimension size]\n";
        std::cerr << "Example: " << argv[0] << " 1000000 500\n";
        return 1;
    }

    size_t num_points = std::stoull(argv[1]);
    int dimensions = std::stoi(argv[2]);

    std::cout << "Generating " << num_points << " points with " << dimensions 
              << " dimensions into 'master.txt'...\n";

    // 2. Open file with a massive write buffer for speed
    std::ofstream outfile("inputs/master.txt", std::ios::out | std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open inputs/master.txt for writing.\n";
        return 1;
    }
    
    // Provide a 1MB buffer to prevent constant disk thrashing
    std::vector<char> buffer(1024 * 1024);
    outfile.rdbuf()->pubsetbuf(buffer.data(), buffer.size());

    // 3. Fast Random Setup
    // Using minstd_rand instead of mt19937 because it is significantly faster 
    // and perfectly fine for generating dummy benchmark coordinates.
    std::random_device rd;
    std::minstd_rand gen(rd()); 
    std::uniform_real_distribution<float> coord_dist(-1000.0f, 1000.0f);

    // 4. Generation Loop
    for (size_t i = 0; i < num_points; ++i) {
        // Write the D coordinates
        for (int d = 0; d < dimensions; ++d) {
            outfile << coord_dist(gen) << " ";
        }
        
        // Strictly use \n instead of std::endl to avoid flushing the buffer early
        outfile << "\n"; 
        
        // Progress tracker for huge files
        if ((i + 1) % 100000 == 0) {
            std::cout << "  ... " << (i + 1) << " points generated.\n";
        }
    }

    outfile.close();
    std::cout << "Done! Saved to inputs/master.txt\n";

    return 0;
}