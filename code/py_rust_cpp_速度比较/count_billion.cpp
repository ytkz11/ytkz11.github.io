#include <iostream>
#include <chrono>

int main() {
    size_t counter = 0;
    auto start = std::chrono::high_resolution_clock::now();
    while (counter < 1000000000) {
        counter++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Counter: " << counter << ", Time taken: " << duration.count() << " milliseconds" << std::endl;
    return 0;
}