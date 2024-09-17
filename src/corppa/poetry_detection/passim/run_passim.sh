#!/usr/bin/env bash

export SPARK_SUBMIT_ARGS='--master local[4] --driver-memory 8G --executor-memory 8G'

IN=$1
OUT=$2


if [ $# -ne 2 ]; then
  echo "Usage: [input] [output]"
  exit 1
fi

passim  $IN $OUT --fields "corpus" --filterpairs "corpus <> 'ppa' AND corpus2 = 'ppa'" --pairwise
