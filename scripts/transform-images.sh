#! /bin/sh

# For the images specified in the input jsonl, copy and transform images
# in from the input directory to the output directory, according to the
# mode specified.

mode=$1
in_jsonl=$2
in_dir=$3
out_dir=$4


# Arg validation
if [ $# -ne 4 ]; then
  echo "Usage: [mode] [jsonl] [in dir] [out dir]"
  exit 1
fi
# Check that jsonl file exists
if [ ! -f "$in_jsonl" ]; then
  echo "ERROR: File $in_jsonl does not exist!"
  exit 1
fi
# Check input dir exists
if [ ! -d "$in_dir" ]; then
  echo "ERROR: Directory $in_dir does not exist!"
  exit 1
fi
# Check output dir exists
if [ ! -d "$out_dir" ]; then
  echo "ERROR: Directory $out_dir does not exist!"
  exit 1
fi
# Check that the mode is valid
if [ "$mode" != "copy" ]; then
  echo "ERROR: Invalid mode '$mode'"
  exit 1
fi

for path_str in `jq ".image_path" $in_jsonl`; do
  # strip double quotes
  img_path=${path_str#'"'}
  img_path=${img_path%'"'}
  echo "$img_path"

  # Check image exists
  in_path="$in_dir/$img_path"
  if [ ! -f "$in_path" ]; then
    "WARNING: Image $in_path does not exist!"
  fi

  out_path="$out_dir/$img_path"
  out_subdir=`dirname "$out_path"`
  if [ ! -d "$out_subdir" ]; then
    mkdir -p "$out_subdir"
  fi
  
  if [ $mode == "copy" ]; then
    # For now just make copies
    cp "$in_path" "$out_path"
  else
    echo "ERROR: Unkown mode '$mode'"
    exit 1
  fi
done
