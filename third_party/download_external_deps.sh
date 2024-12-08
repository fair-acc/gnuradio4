#!/bin/bash

# list of git locations, their target folders, and content type
declare -A git_locations=(
    ["https://api.github.com/repos/Neargye/magic_enum/contents/include?ref=v0.9.3"]="magic_enum multiple"
    ["https://raw.githubusercontent.com/ArashPartow/exprtk/7b993904a21639304edd4db261f6e2cdcf6d936b/exprtk.hpp"]="exprtk single"
)

# Create a download function that can be used with parallel
download() {
    url="$1"
    folder="$2"
    wget -N -q --show-progress -P "$folder" "$url"
}

export -f download

for url in "${!git_locations[@]}"; do
    read -r folder content_type <<< "${git_locations[$url]}"
    folder="./$folder"
    [ ! -d "$folder" ] && mkdir -p "$folder"

    if [ "$content_type" == "single" ]; then # download a single file
        download "$url" "$folder"
    elif [ "$content_type" == "multiple" ]; then # use wget to get the list of files
        files_urls=$(wget -qO- "$url" | grep "download_url" | cut -d '"' -f 4)
        echo "$files_urls" | xargs -P 4 -I {} bash -c 'download "$@"' _ {} "$folder"
    fi
done
echo "Downloads completed."
