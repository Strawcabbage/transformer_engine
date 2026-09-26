#!/bin/bash

g++ -std=c++20 -O0 -g -Wall -Wextra -fsanitize=address,undefined \
      src/tensor.cpp -o t && ./t