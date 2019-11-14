#!/usr/bin

cd dataset_tmp/__background__/

mkdir -p "../../dataset/__background__/"
echo "process __background__"
for f in *.wav;
    do
    sox "$f" -e float -b 32 "../../dataset/__background__/$f" rate -v -I 16000
    done
cd ../..

cd dataset_tmp/trainset_clean/
for i in */;
    do
    echo $i
    mkdir -p "../../dataset/trainset_clean/$i"
    cd "$i"
    for f in *.wav;
        do
        sox "$f" -e float -b 32 "../../../dataset/trainset_clean/$i$f" rate -v -I 16000
        done
    cd ..
    done


cd ../..
cd dataset_tmp/valset_clean/
for i in */;
    do
    echo $i
    mkdir -p "../../dataset/valset_clean/$i"
    cd "$i"
    for f in *.wav;
        do
        sox "$f" -e float -b 32 "../../../dataset/valset_clean/$i$f" rate -v -I 16000
        done
    cd ..
    done

cd ../..
cd dataset_tmp/testset_clean/
for i in */;
    do
    echo $i
    mkdir -p "../../dataset/testset_clean/$i"
    cd "$i"
    for f in *.wav;
        do
        sox "$f" -e float -b 32 "../../../dataset/testset_clean/$i$f" rate -v -I 16000
        done
    cd ..
    done