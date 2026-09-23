#!/bin/bash
# fetch N frames starting at frame K from a y4m on xiph: fetch.sh name K N
name=$1; K=$2; N=$3
url=https://media.xiph.org/video/derf/y4m/$name.y4m
hdr=$(curl -s --max-time 30 -r 0-400 "$url" | head -1)
hlen=$(( ${#hdr} + 1 ))
W=$(echo "$hdr" | grep -oE ' W[0-9]+' | tr -d ' W'); H=$(echo "$hdr" | grep -oE ' H[0-9]+' | tr -d ' H')
C=$(echo "$hdr" | grep -oE ' C[0-9a-z]+' | tr -d ' C')
case "$C" in 422*) fs=$((W*H*2));; 444*) fs=$((W*H*3));; *) fs=$((W*H*3/2));; esac
fl=$((fs+6))
start=$((hlen + K*fl)); end=$((start + N*fl - 1))
{ echo "$hdr"; curl -s --max-time 600 -r $start-$end "$url"; } > $name.y4m
ls -la $name.y4m
