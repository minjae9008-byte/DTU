#!/bin/bash
# Build DVD-like degraded clips + ground truth from Xiph 1080p sources.
set -e
cd "$(dirname "$0")"
mkdir -p gt dvd
FF="ffmpeg -v error -y"
# MPEG-2 settings that mimic a decent DVD encoder
M2V="-c:v mpeg2video -b:v 6000k -maxrate 9000k -bufsize 1835k -g 15 -bf 2 -trellis 1 -mbd rd -cmp 2 -subcmp 2 -dia_size 2 -dc 9 -qmin 1"
# HD 709 -> SD 601 (NTSC) like a mastering downconversion
DOWN480="zscale=w=720:h=480:filter=spline36:matrixin=709:transferin=709:primariesin=709:rangein=limited:matrix=170m:transfer=601:primaries=170m:range=limited,format=yuv420p"
DOWN576="zscale=w=720:h=576:filter=spline36:matrixin=709:transferin=709:primariesin=709:rangein=limited:matrix=470bg:transfer=601:primaries=bt470bg:range=limited,format=yuv420p"

for n in old_town_cross park_joy crowd_run ducks_take_off; do
  src=${n}_1080p50.y4m
  # ground truth: even frames -> 60 (or 50) frame 24p film-like clip at 1080p
  $FF -i $src -vf "select='not(mod(n\,2))',setpts=N/(24000/1001)/TB,format=yuv420p" -r 24000/1001 -c:v ffv1 gt/${n}_1080p24.mkv
  # progressive DVD (clean): anamorphic 16:9 720x480
  $FF -i gt/${n}_1080p24.mkv -vf "$DOWN480" $M2V -aspect 16:9 -r 24000/1001 -f vob dvd/${n}_clean.mpg
  # progressive DVD (grainy): add temporal grain before encoding
  $FF -i gt/${n}_1080p24.mkv -vf "$DOWN480,noise=c0s=10:c0f=t+u:c1s=5:c1f=t+u:c2s=5:c2f=t+u" $M2V -aspect 16:9 -r 24000/1001 -f vob dvd/${n}_grain.mpg
done

# Sintel (24p film, letterboxed) -> telecined NTSC 29.97i + progressive
$FF -i sintel_trailer_2k_1080p24.y4m -vf "setpts=N/(24000/1001)/TB,format=yuv420p" -r 24000/1001 -c:v ffv1 gt/sintel_1080p24.mkv
$FF -i gt/sintel_1080p24.mkv -vf "$DOWN480" -c:v ffv1 gt/sintel_480p24.mkv
$FF -i gt/sintel_480p24.mkv $M2V -aspect 16:9 -r 24000/1001 -f vob dvd/sintel_clean.mpg
$FF -i gt/sintel_480p24.mkv -vf "telecine=first_field=top:pattern=23" $M2V -b:v 7000k -flags +ilme+ildct -alternate_scan 1 -aspect 16:9 -r 30000/1001 -f vob dvd/sintel_telecine.mpg

# Interlaced video: park_joy 1080p50 -> 576p50 GT -> 576i25 (PAL video camera)
$FF -i park_joy_1080p50.y4m -vf "$DOWN576" -c:v ffv1 gt/park_joy_576p50.mkv
$FF -i gt/park_joy_576p50.mkv -vf "interlace=scan=tff:lowpass=linear" $M2V -b:v 7000k -flags +ilme+ildct -alternate_scan 1 -aspect 16:9 -r 25 -f vob dvd/park_joy_576i.mpg
# crowd_run 1080p50 -> 480p "59.94" GT -> 480i (NTSC video camera)
$FF -i crowd_run_1080p50.y4m -vf "$DOWN480,setpts=N/(60000/1001)/TB" -r 60000/1001 -c:v ffv1 gt/crowd_run_480p60.mkv
$FF -i gt/crowd_run_480p60.mkv -vf "interlace=scan=tff:lowpass=linear" $M2V -b:v 7000k -flags +ilme+ildct -alternate_scan 1 -aspect 16:9 -r 30000/1001 -f vob dvd/crowd_run_480i.mpg
# SD progressive 50p GT for frame-interpolation tests
$FF -i crowd_run_1080p50.y4m -vf "$DOWN480" -c:v ffv1 gt/crowd_run_480p50.mkv
$FF -i old_town_cross_1080p50.y4m -vf "$DOWN480" -c:v ffv1 gt/old_town_cross_480p50.mkv
ls -la gt dvd
