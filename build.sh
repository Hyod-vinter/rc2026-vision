rm -rf build
rm -rf logs
source ~/rc2026/install/setup.bash
cmake -B build
make -C build/ -j`nproc`
