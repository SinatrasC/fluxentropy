#!/bin/bash

python analyzer.py \
    --input-csv "entropy_avarentropy_string_correctly_answered.csv" \
    --output-dir "analysis_results" \
    --univariate-metrics entropy varentropy \
    --multivariate-pairs entropy varentropy \
    --multivariate-pairs entropy sqrt_varentropy 