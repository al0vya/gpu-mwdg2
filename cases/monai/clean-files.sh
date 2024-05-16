find . -type d -name "eps-*" | grep -v '\-0$' | while read dir; do
    find "$dir" -type f \( -name "*.dem" -o -name "*.wd" -o -name "*.elev" -o -name "*.stage" \) -exec rm {} \;
done

