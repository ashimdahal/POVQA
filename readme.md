# Video Question Answering with motion blurred frames from video and chain of thought reasoning on cc

reference video atm : https://www.youtube.com/watch?v=nIvePkFzaMw


To visualize the test:
```
python scripts/frames_to_videos.py how_things_move --fps 1
```

To run on tvqa:

```
python scripts/tvqa_processing.py dir_to_tvqa_data output_dir 
```

To generate synthetic CoT:
```
python scripts/generate_synthetic_cot.py processed_tvqa_data tvqa_qa.jsonl output_cot.jsonl --model_name Qwen/Qwen-2.5-VL-3B-Instruct --use_4bit True
```
