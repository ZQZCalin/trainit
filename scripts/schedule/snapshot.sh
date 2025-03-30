snapshot() {
    local dest="$1"

    # Store config.sh
	printf '%.s-' {1..100} > "$dest"
	echo -e "\n# config.sh\n" >> "$dest"
	cat scripts/schedule/config.sh >> "$dest"

    # Store get_next_lr.py
	echo -e "\n\n\n" >> "$dest"
	printf '%.s-' {1..100} >> "$dest"
	echo -e "\n# get_next_lr.py\n" >> "$dest"
	cat scripts/schedule/get_next_lr.py >> "$dest"
}