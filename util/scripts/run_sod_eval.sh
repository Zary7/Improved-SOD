
method_=ICON-P

python util/evaltool/sod_eval.py   \
    --method  $method_ \
    --dataset  'ECSSD' 

python util/evaltool/sod_eval.py   \
    --method  $method_ \
    --dataset  'PASCAL-S' 

python util/evaltool/sod_eval.py   \
    --method  $method_ \
    --dataset  'DUTS' 

python util/evaltool/sod_eval.py   \
    --method  $method_ \
    --dataset  'HKU-IS' 

python util/evaltool/sod_eval.py   \
    --method  $method_ \
   --dataset  'DUT-OMRON' 

python util/evaltool/sod_eval.py   \
    --method  $method_ \
    --dataset  'SOD' 


