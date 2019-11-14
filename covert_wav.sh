#!/usr/bin


cd dataset_tmp
for d in */;
    do
    echo 111$d
    for i in $d*/;
        do
        echo 222$i
        mkdir -p "../dataset/$i"
        cd "$i"
        for f in *.wav; 
            do
            # echo 333"../../../dataset/$i$f"
            sox "$f" -e float -b 32 "../../../dataset/$i$f" rate -v -I 16000
            done
        cd ../..
        done
    done