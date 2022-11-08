#!/bin/bash

# Parsear MNIST dataset al formato deseado

ORI=MNISTjpg/
DIR=MNIST/
IMGDIR=MNIST/Images
EXT=jpg

# Estructura base de carpetas:
# NOMBRE_DATASET
#    Images
#        Ficheros de imagenes con labels en el nombre
#        ...
if [ ! -d "$DIR" ]; then
    mkdir $DIR
else
    rm -r $DIR
    mkdir $DIR
fi
mkdir $IMGDIR

# Mover las imagenes todas a la misma carpeta
num=0
for d in $ORI*/ ; do
    echo -n "D[$d]..."
    epoch=0
    for img in $d/* ; do
        # Nombre de archivo: label_datetime.jpg
        newpath="$IMGDIR/${num}_${epoch}.$EXT"
        cp $img $newpath
        ((epoch=epoch+1))
    done
    echo "Done!"
    ((num=num+1))
done

echo -n "Testdir..."
epoch=0
for img in $ORItest/ ; do
    # Nombre de archivo: label_datetime.jpg
    newpath="$IMGDIR/${num}_${epoch}.$EXT"
    cp $img $newpath
    ((epoch=epoch+1))
done
echo "Done!"


echo "\nAll files were successfully copied!"
