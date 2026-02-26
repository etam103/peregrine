#!/bin/bash
# Memory watchdog for training process.
# Monitors RSS of the given PID every 2 seconds.
# Kills the process if RSS exceeds 80% of 64GB (51.2GB).

PID=$1
THRESHOLD_KB=$((51 * 1024 * 1024))  # ~51.2 GB in KB
LOGFILE="/Users/evantam/Desktop/rustorch/mem_log.txt"

echo "=== Memory Watchdog ===" > "$LOGFILE"
echo "Monitoring PID $PID, threshold: ${THRESHOLD_KB} KB (~51.2 GB)" >> "$LOGFILE"
echo "Started at $(date)" >> "$LOGFILE"
echo "" >> "$LOGFILE"

PEAK_KB=0

while kill -0 "$PID" 2>/dev/null; do
    RSS_KB=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ')
    if [ -z "$RSS_KB" ]; then
        break
    fi

    if [ "$RSS_KB" -gt "$PEAK_KB" ]; then
        PEAK_KB=$RSS_KB
    fi

    RSS_MB=$((RSS_KB / 1024))
    PEAK_MB=$((PEAK_KB / 1024))
    TIMESTAMP=$(date '+%H:%M:%S')

    echo "$TIMESTAMP  RSS: ${RSS_MB} MB  Peak: ${PEAK_MB} MB" >> "$LOGFILE"

    if [ "$RSS_KB" -gt "$THRESHOLD_KB" ]; then
        echo "" >> "$LOGFILE"
        echo "!!! MEMORY THRESHOLD EXCEEDED !!!" >> "$LOGFILE"
        echo "$TIMESTAMP  RSS ${RSS_MB} MB > threshold ~51200 MB" >> "$LOGFILE"
        echo "Killing PID $PID..." >> "$LOGFILE"
        kill "$PID" 2>/dev/null
        sleep 1
        kill -9 "$PID" 2>/dev/null
        echo "Process killed." >> "$LOGFILE"
        break
    fi

    sleep 2
done

echo "" >> "$LOGFILE"
echo "=== Watchdog finished at $(date) ===" >> "$LOGFILE"
echo "Peak RSS: ${PEAK_MB} MB" >> "$LOGFILE"
