for dir in task_*/
do
    echo $dir
    cd $dir
    echo 0 | gmx trjconv -f shear.xtc -s shear.tpr -o shear_whole.xtc -pbc whole
    wait
    cd ..
done
