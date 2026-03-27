# Praat script for pitch detection
# Usage: praat --run pitch_detect.praat <audio_file> <start> <duration> <step> <min_pitch> <max_pitch>
#
# Arguments are passed via command line, but Praat requires them to be set in the script.
# This script expects the audio file path to be hardcoded or passed via form.

form Pitch_Detection
    text audio_file /path/to/audio.wav
    real start_time 0
    real duration 3
    real time_step 0.01
    real min_pitch 75
    real max_pitch 1500
endform

# Kalimba note frequencies (17-key C-tuned)
# C4=261.6, D4=293.7, E4=329.6, F4=349.2, G4=392.0, A4=440.0, B4=493.9
# C5=523.3, D5=587.3, E5=659.3, F5=698.5, G5=784.0, A5=880.0, B5=987.8
# C6=1046.5, D6=1174.7, E6=1318.5

Read from file: audio_file$
sound = selected("Sound")

# Extract pitch
To Pitch: time_step, min_pitch, max_pitch
pitch = selected("Pitch")

writeInfoLine: "time_s", tab$, "pitch_hz", tab$, "note", tab$, "deviation_pct"

# Iterate through time points
num_frames = Get number of frames
for i from 1 to num_frames
    time = Get time from frame number: i
    if time >= start_time and time <= start_time + duration
        p = Get value at time: time, "Hertz", "Linear"
        if p <> undefined
            # Find closest kalimba note
            note$ = "?"
            min_dev = 100

            # Check each kalimba note
            if abs(p - 261.6) / 261.6 * 100 < min_dev
                note$ = "C4"
                min_dev = abs(p - 261.6) / 261.6 * 100
            endif
            if abs(p - 293.7) / 293.7 * 100 < min_dev
                note$ = "D4"
                min_dev = abs(p - 293.7) / 293.7 * 100
            endif
            if abs(p - 329.6) / 329.6 * 100 < min_dev
                note$ = "E4"
                min_dev = abs(p - 329.6) / 329.6 * 100
            endif
            if abs(p - 349.2) / 349.2 * 100 < min_dev
                note$ = "F4"
                min_dev = abs(p - 349.2) / 349.2 * 100
            endif
            if abs(p - 392.0) / 392.0 * 100 < min_dev
                note$ = "G4"
                min_dev = abs(p - 392.0) / 392.0 * 100
            endif
            if abs(p - 440.0) / 440.0 * 100 < min_dev
                note$ = "A4"
                min_dev = abs(p - 440.0) / 440.0 * 100
            endif
            if abs(p - 493.9) / 493.9 * 100 < min_dev
                note$ = "B4"
                min_dev = abs(p - 493.9) / 493.9 * 100
            endif
            if abs(p - 523.3) / 523.3 * 100 < min_dev
                note$ = "C5"
                min_dev = abs(p - 523.3) / 523.3 * 100
            endif
            if abs(p - 587.3) / 587.3 * 100 < min_dev
                note$ = "D5"
                min_dev = abs(p - 587.3) / 587.3 * 100
            endif
            if abs(p - 659.3) / 659.3 * 100 < min_dev
                note$ = "E5"
                min_dev = abs(p - 659.3) / 659.3 * 100
            endif
            if abs(p - 698.5) / 698.5 * 100 < min_dev
                note$ = "F5"
                min_dev = abs(p - 698.5) / 698.5 * 100
            endif
            if abs(p - 784.0) / 784.0 * 100 < min_dev
                note$ = "G5"
                min_dev = abs(p - 784.0) / 784.0 * 100
            endif
            if abs(p - 880.0) / 880.0 * 100 < min_dev
                note$ = "A5"
                min_dev = abs(p - 880.0) / 880.0 * 100
            endif
            if abs(p - 987.8) / 987.8 * 100 < min_dev
                note$ = "B5"
                min_dev = abs(p - 987.8) / 987.8 * 100
            endif
            if abs(p - 1046.5) / 1046.5 * 100 < min_dev
                note$ = "C6"
                min_dev = abs(p - 1046.5) / 1046.5 * 100
            endif
            if abs(p - 1174.7) / 1174.7 * 100 < min_dev
                note$ = "D6"
                min_dev = abs(p - 1174.7) / 1174.7 * 100
            endif
            if abs(p - 1318.5) / 1318.5 * 100 < min_dev
                note$ = "E6"
                min_dev = abs(p - 1318.5) / 1318.5 * 100
            endif

            appendInfoLine: fixed$(time, 3), tab$, fixed$(p, 1), tab$, note$, tab$, fixed$(min_dev, 1)
        endif
    endif
endfor

# Cleanup
select sound
plus pitch
Remove
