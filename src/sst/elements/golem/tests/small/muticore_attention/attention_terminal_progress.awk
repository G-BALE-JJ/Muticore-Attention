{
    line = $0
    gsub(/\r/, "", line)
    if (line !~ /^[[:space:]]*SIM[[:space:]]+[0-9]+%/) {
        next
    }

    compact = line
    sub(/^[[:space:]]*SIM[[:space:]]+/, "", compact)
    percent = compact
    sub(/%.*/, "%", percent)
    phase = compact
    sub(/^[0-9]+%[[:space:]]+/, "", phase)
    sub(/[[:space:]]+elapsed[[:space:]].*$/, "", phase)
    sub(/[[:space:]]+$/, "", phase)

    if (phase != last_phase) {
        printf("  SST %4s  %s\n", percent, phase)
        fflush()
        last_phase = phase
    }
}
