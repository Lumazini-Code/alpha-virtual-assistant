rm -rf build
cmake -S . -B build -DBUILD_PLATFORM=LINUX
cmake --build build -j$(nproc)
