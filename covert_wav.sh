#!/usr/bin

cd dataset_tmp/__background__/

mkdir -p "../../dataset/__background__/"
echo "process __background__"
for f in *.wav;
    do
    sox "$f" -r 16000 -b 32 -e float "../../dataset/__background__/$f"
    done
cd ../..


#cd dataset_tmp/trainset_clean/
#for i in */;
#    do
#    echo "trainset_clean/$i"
#    mkdir -p "../../dataset/trainset_clean/$i"
#    cd "$i"
#    for f in *.wav;
#        do
#        sox "$f" -r 16000 -b 32 -e float "../../../dataset/trainset_clean/$i$f"
#        done
#    cd ..
#    done
#
#
#cd ../..
#cd dataset_tmp/valset_clean/
#for i in */;
#    do
#    echo "valset_clean/$i"
#    mkdir -p "../../dataset/valset_clean/$i"
#    cd "$i"
#    for f in *.wav;
#        do
#        sox "$f" -r 16000 -b 32 -e float "../../../dataset/valset_clean/$i$f"
#        done
#    cd ..
#    done
#
#cd ../..
#cd dataset_tmp/testset_clean/
#for i in */;
#    do
#    echo "testset_clean/$i"
#    mkdir -p "../../dataset/testset_clean/$i"
#    cd "$i"
#    for f in *.wav;
#        do
#        sox "$f" -r 16000 -b 32 -e float "../../../dataset/testset_clean/$i$f"
#        done
#    cd ..
#    done