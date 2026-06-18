
echo "Now Testing ICON-P..."
python main/test.py \
    --model 'ICON-P' \
    --task 'SOD' \
    --ckpt 'checkpoint/ICON/ICON-P/PVT.weight'

echo "Now Testing ICON-R..."
python main/test.py \
    --model 'ICON-R' \
    --task 'SOD' \
    --ckpt 'checkpoint/ICON/ICON-R/Res.weight'




