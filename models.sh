for b in {0..20..2}; do
    for knn in {1..3}; do
        for sel in {1..3}; do
            for mod in {1..3..2}; do
                python3 models.py "${b}" "${knn}" "${sel}" "${mod}"
            done
        done
    done
done
