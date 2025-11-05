nvcc -x cu -O3 $2 --std=c++20 --expt-relaxed-constexpr -arch sm_80 -Isrc/common $1 -o a.out
