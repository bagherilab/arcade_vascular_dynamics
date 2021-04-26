#!/bin/bash

# List of JSON file extensions (comment out to ignore).
JSONS=(
    ".SEEDS.GROWTH"
    ".SEEDS.SYMMETRY"
    ".SEEDS.CYCLES"
    ".SEEDS.ACTIVITY"
    ".SEEDS.TYPES"
    ".SEEDS.POPS"
    ".SEEDS.COUNTS"
    ".SEEDS.VOLUMES"
    ".SEEDS.DIAMETERS"
    ".METRICS.GROWTH"
    ".METRICS.SYMMETRY"
    ".METRICS.CYCLES"
    ".METRICS.ACTIVITY"
    ".METRICS.TYPES"
    ".METRICS.POPS"
    ".METRICS.COUNTS"
    ".METRICS.VOLUMES"
    ".METRICS.DIAMETERS"
    ".CONCENTRATIONS"
    ".CENTERS"
)

# List of CSV file extensions (comment out to ignore).
# Note additional GRAPH extensions are added below.
CSVS=(
    ".DISTRIBUTION"
    ".LOCATIONS"
    ".OUTLINES"
)

# Do nothing if no argument is passed.
if [[ $# -eq 0 ]] ; then
    exit 0
fi

NAME=$1

# Add additional CSV files for GRAPH simulations.
if [[ "$NAME" == "EXACT_HEMODYNAMICS" || "$NAME" == "VASCULAR_FUNCTION" ]]; then
    CSVS+=(".GRAPH")
    CSVS+=(".MEASURES")
fi

# Iterate through all JSON files to compress.
for JSON in ${JSONS[@]}; do
    JSON_DIRECTORY=$NAME/_$NAME${JSON}

    if [[ -d "$JSON_DIRECTORY" ]]
    then
        echo "$JSON_DIRECTORY already exists. Please move contents out and remove."
        continue
    fi

    mkdir $JSON_DIRECTORY
    mv $NAME/$NAME*${JSON}.json $JSON_DIRECTORY
    cd $JSON_DIRECTORY
    COPYFILE_DISABLE=1 tar cJvf ../$NAME${JSON}.tar.xz *.json
    cd ../..
done

# Iterate through all CSV files to compress.
for CSV in ${CSVS[@]}; do
    CSV_DIRECTORY=$NAME/_$NAME${CSV}

    if [[ -d "$CSV_DIRECTORY" ]]
    then
        echo "$CSV_DIRECTORY already exists. Please move contents out and remove."
        continue
    fi

    mkdir $CSV_DIRECTORY
    mv $NAME/$NAME*${CSV}*.csv $CSV_DIRECTORY
    cd $CSV_DIRECTORY
    COPYFILE_DISABLE=1 tar cJvf ../$NAME${CSV}.tar.xz *.csv
    cd ../..
done
