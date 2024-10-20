#!usr/bin/bash

# Check for the presence of 2 arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_dir> <output_dir>"
    exit 1
fi

# Move to model dir
cd $1 || exit 1

zip_subfolders() {
    for dir in */; do
        if [ -d "$dir" ]; then
            zip -r "${dir%/}.zip" "$dir"
        fi
    done
}

zip_subfolders()
mv *.zip $2