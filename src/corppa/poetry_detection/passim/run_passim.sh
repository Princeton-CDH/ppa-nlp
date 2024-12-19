#!/usr/bin/env bash
#####
# Script for running passim. Requires Java 8*/11/17 because of Spark dependency.
#
# Examples:
#   ./run_passim.sh corpus.jsonl output-dir-path
#   ./run_passim.sh "input/*.jsonl" output-dir-path
#   ./run_passim.sh "{ppa_texts.jsonl,ref_texts.jsonl}" output-dir-path
#####

# Spark settings
export SPARK_LOCAL_IP="127.0.0.1"
export SPARK_SUBMIT_ARGS='--master local[6] --driver-memory 8G --executor-memory 8G'

IN=$1
OUT=$2


if [ $# -ne 2 ]; then
  echo "Usage: [input] [output]"
  exit 1
fi

# Check for supported java version (via major version number)
major_version=`javap -verbose java.lang.String | grep "major version" | rev | cut -d' ' -f 1 | rev`
if [ "$major_version" == "61" ]; then
  # Supported: Java 17
  echo "DEBUG: Running passim with Java 17"
elif [ "$major_version" == "55" ]; then
  # Supported: Java 11
  echo "DEBUG: Running passim with Java 11"
elif [ "$major_version" == "52" ]; then
  # Supported: Java 8, update 371 and higher
  java_version=`java -version 2>&1 | grep -o '"[^"]*"' | cut -d'"' -f 2`
  # Check update number
  update_num=`echo $java_version | cut -d_ -s -f 2`
  if [[ -z "$update_num" || "$update_num" < 371 ]]; then
    # Fail if no update number found, or below 371
    echo "ERROR: Java 8u$update_version not supported." \
         "Only versions 8u371 and higher are supported."
    exit 1
  fi
  echo "DEBUG: Running passim with Java 8u$update_num"
elif [ "$major_version" == "65" ]; then
  # Unsupported: Java 21
  echo "ERROR: Java 21 is not supported, run with Java 8*/11/17"
  exit 1
elif [ "$major_version" == "67" ]; then
  # Unsupported: Java 23
  echo "ERROR: Java 23 is not supported, run with Java 8*/11/17"
  exit 1
else
  echo "ERROR: Must run with Java 8*/11/17."
  exit 1
fi

passim "$IN" "$OUT" --fields "corpus" --filterpairs "corpus <> 'ppa' AND corpus2 = 'ppa'" --pairwise


align_json="$OUT/align.json"
out_json="$OUT/out.json"

if [[ -f "$align_json/_SUCCESS" && -f "$out_json/_SUCCESS" ]]; then
  echo "DEBUG: passim run completed successfully."
else
  echo "ERROR: passim run failed."
  exit 1
fi
