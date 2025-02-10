# Cantonese Punctuation Restoration

Download all videos in a youtube playlist in MP3 format.

```
yt-dlp -x --audio-format mp3 [LINK TO PLAYLIST]
```

- Showbiz娛樂專訪: https://www.youtube.com/playlist?list=PL3GvCTq2j37-QDZ6fznA3IzCyCSiOsmYc
- 健康快訊: https://www.youtube.com/playlist?list=PL3GvCTq2j37-0j-2Rrfrszo-ze4qceY59
- 開市追揸沽 & ET開市直擊: https://www.youtube.com/playlist?list=PLw1D7LF5fHU2RzADjQHSguco_x7_xlVpC
- 央廣節目：自由廣場: https://www.youtube.com/playlist?list=PL4obi-HkLgFLCjWSpEqPVwRX6kE9KKgsU
- 桑普對談: https://www.youtube.com/playlist?list=PL4obi-HkLgFLogVGpzPi7wVj0OrsS_VCu
- Thailand 泰國: https://www.youtube.com/playlist?list=PLHX2wryodhB3-c7bXt9LNq4EcIr1NK1SL
- Indonesia 印尼: https://www.youtube.com/playlist?list=PLHX2wryodhB37iu76R1x3vNFl1ysFPDwj
- 行山好去處: https://www.youtube.com/playlist?list=PLp9AUTyGC_soitTBKlpBVyIhofNcEwinW

Train.

```
CUDA_VISIBLE_DEVICES=5 python train.py \
    --dataset_name both \
    --model_path /mnt/nas2/Pretrained_LM/bert-base-multilingual-uncased \
    --seed 18 \
    --output_dir outputs \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --dataloader_num_workers 1 \
    --learning_rate 3e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_safetensors False \
    --save_only_model True \
    --metric_for_best_model macro_f1 \
    --load_best_model_at_end True \
    --do_train \
    --do_predict \
    --report_to none
```
